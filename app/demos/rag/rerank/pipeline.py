import time
from typing import Literal

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_typesafe import Noul, NoulCriteria
from pymongo.collection import Collection
from sentence_transformers import CrossEncoder

from core.config import get_settings
from core.embeddings import LocalEmbeddings
from core.llm import get_chat_model
from core.mongo import clear_doc, ensure_indexes, get_collection
from core.pdf import extract_text
from core.tracing import get_callbacks, is_enabled, observe
from core.typesafe import get_classifier
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_rerank"

PROMPT_TEMPLATE = """Answer the question using only the numbered context below. \
Cite the page number(s) you relied on in square brackets, e.g. [p.3].

{context}

Question: {question}
Answer:"""


# One yes/no question, asked once per (question, passage) pair. Jev answers with the
# probability that it's "yes"; the thresholds that act on it live in code, not in this text.
JEV_QUESTION = Noul(
    instructions="Does the passage contain information that helps answer the question?",
    criteria=NoulCriteria(
        true=(
            "The passage states facts, definitions, figures or explanations that answer "
            "the question or a necessary part of it."
        ),
        false=(
            "The passage is only on a related topic, or repeats the question's terms "
            "without supplying anything usable in an answer."
        ),
    ),
)
JEV_PRICE_PER_M_INPUT_TOKENS = 0.042  # USD; output tokens are free (docs.typesafe.ai/models)

Reranker = Literal["cross_encoder", "jev"]


@st.cache_resource(show_spinner="Loading re-ranker model...")
def _get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.rerank_model)


def _get_vector_store(collection: Collection) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection,
        LocalEmbeddings(),
        index_name="vector_index",
        auto_create_index=False,
    )


def ingest(pdf_bytes: bytes, doc_id: str) -> IngestStats:
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    clear_doc(collection, doc_id)

    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rerank_chunk_size, chunk_overlap=settings.rerank_chunk_overlap
    )

    texts: list[str] = []
    metadatas: list[dict] = []
    for page_num, text in enumerate(parsed.pages, start=1):
        if not text.strip():
            continue
        for chunk_text in splitter.split_text(text):
            texts.append(chunk_text)
            metadatas.append({"doc_id": doc_id, "page": page_num})

    if texts:
        vector_store = _get_vector_store(collection)
        vector_store.add_texts(texts, metadatas=metadatas)
        ensure_indexes(collection)

    return IngestStats(doc_id=doc_id, chunks=len(texts), latency_ms=(time.monotonic() - start) * 1000)


def _cross_encoder_scores(question: str, texts: list[str]) -> tuple[list[float], str]:
    pairs = [(question, t) for t in texts]
    scores = _get_reranker().predict(pairs) if pairs else []
    return [float(x) for x in scores], f"cross-encoder ({get_settings().rerank_model})"


def _jev_scores(question: str, texts: list[str], pages: list[int]) -> tuple[list[float], str, int]:
    """One Jev request per candidate, fanned out in parallel; each shows up as its own
    observation in Langfuse. Returns the probabilities, the model ID Jev reports having
    answered with (proof of the pin) and the input tokens used."""
    requests = [
        {
            "state": {"question": question, "passage": {"page": page, "text": text}},
            "questions": {"relevant": JEV_QUESTION},
        }
        for text, page in zip(texts, pages)
    ]
    if not requests:
        return [], get_settings().typesafe_model, 0
    max_concurrency = get_settings().rerank_jev_concurrency
    responses = get_classifier().batch(
        requests,
        config=[
            {
                "callbacks": get_callbacks(),
                "run_name": f"jev_candidate_{i}",
                "max_concurrency": max_concurrency,
            }
            for i in range(len(requests))
        ],
    )
    probabilities = [r.nouls["relevant"].noul for r in responses]
    tokens = sum(r.usage.input_tokens or 0 for r in responses)
    return probabilities, responses[0].model, tokens


def _push_jev_scores(rows: list[dict], kept: list[dict], model: str) -> None:
    """Jev's probabilities as Langfuse scores on this run's trace, so they can be charted
    and evaluated -- not just read in the trace. Names are stable on purpose: a saved
    dashboard matches on them. Never raises; scoring is optional."""
    if not is_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        trace_id = client.get_current_trace_id()
        if trace_id is None:
            return
        kept_ids = {id(r) for r in kept}
        for row in rows:
            outcome = f"{row['post_rank']}" if id(row) in kept_ids else "cut"
            client.create_score(
                name="jev_relevance",
                value=row["rerank_score"],
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=f"p.{row['page']} rank {row['pre_rank']} -> {outcome}",
                metadata={"page": row["page"], "pre_rank": row["pre_rank"], "model": model},
            )
        if kept:
            client.create_score(
                name="jev_top1_relevance",
                value=kept[0]["rerank_score"],
                trace_id=trace_id,
                data_type="NUMERIC",
            )
            client.create_score(
                name="jev_mean_kept_relevance",
                value=sum(r["rerank_score"] for r in kept) / len(kept),
                trace_id=trace_id,
                data_type="NUMERIC",
            )
    except Exception:
        pass


@observe(name="rerank_ask")
def ask_detailed(
    question: str,
    doc_id: str,
    *,
    candidate_k: int = 20,
    top_k: int = 5,
    reranker: Reranker = "cross_encoder",
    min_probability: float = 0.0,
    **_: object,
) -> tuple[RagResult, list[dict], dict]:
    """Runs the full pipeline and also returns the before/after ranking rows and a small
    info dict (reranker used, how many candidates the Jev threshold dropped) for the page.

    `min_probability` only applies to Jev: a calibrated probability can be cut at a threshold,
    which a cross-encoder's unbounded logit can't.
    """
    start = time.monotonic()
    steps: list[str] = []
    collection = get_collection(COLLECTION_NAME)
    vector_store = _get_vector_store(collection)

    candidate_hits = vector_store.similarity_search_with_score(
        question, k=candidate_k, pre_filter={"doc_id": doc_id}
    )
    steps.append(f"Retriever (k={candidate_k}) on rag_rerank returned {len(candidate_hits)} candidate(s).")

    texts = [doc.page_content for doc, _ in candidate_hits]
    pages = [doc.metadata["page"] for doc, _ in candidate_hits]
    rerank_start = time.monotonic()
    jev_model, jev_tokens = "", 0
    if reranker == "jev":
        scores, jev_model, jev_tokens = _jev_scores(question, texts, pages)
    else:
        scores, label = _cross_encoder_scores(question, texts)
    rerank_ms = (time.monotonic() - rerank_start) * 1000

    rows = [
        {
            "text": text,
            "page": page,
            "pre_rank": pre_rank,
            "pre_score": pre_score,
            "rerank_score": score,
        }
        for pre_rank, (text, page, (_, pre_score), score) in enumerate(
            zip(texts, pages, candidate_hits, scores), start=1
        )
    ]
    rows.sort(key=lambda r: r["rerank_score"], reverse=True)

    eligible = [r for r in rows if r["rerank_score"] >= min_probability] if reranker == "jev" else rows
    dropped = len(rows) - len(eligible)
    kept = eligible[:top_k]
    for post_rank, row in enumerate(kept, start=1):
        row["post_rank"] = post_rank

    if reranker == "jev":
        cost = jev_tokens * JEV_PRICE_PER_M_INPUT_TOKENS / 1_000_000
        steps.append(
            f"Jev ({jev_model}) judged {len(candidate_hits)} candidate(s) in {rerank_ms:.0f} ms "
            f"(1 request each, {jev_tokens} input tokens, ~${cost:.5f}). "
            f"{dropped} fell below the {min_probability:.2f} probability threshold; kept the top {len(kept)}."
        )
        _push_jev_scores(rows, kept, jev_model)
    else:
        steps.append(
            f"{label} re-scored {len(candidate_hits)} candidate(s) "
            f"in {rerank_ms:.0f} ms, kept the top {len(kept)}."
        )

    info = {"reranker": reranker, "dropped": dropped, "min_probability": min_probability}

    if not kept:
        steps.append("No passage cleared the Jev threshold, so the chat model was not called.")
        result = RagResult(
            answer=(
                "No retrieved passage was judged relevant enough to answer this question "
                f"(Jev probability threshold {min_probability:.2f}). Lower the threshold or rephrase."
            ),
            contexts=[],
            steps=steps,
            llm_calls=0,
            tokens=0,
            latency_ms=(time.monotonic() - start) * 1000,
        )
        return result, [], info

    context = "\n\n".join(f"[{i + 1}] (p.{r['page']}) {r['text']}" for i, r in enumerate(kept))
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"context": context, "question": question},
        config={"callbacks": get_callbacks(), "run_name": "rerank_answer"},
    )
    steps.append("Answered from the re-ranked context via an LCEL chain (prompt | chat model | parser).")

    latency_ms = (time.monotonic() - start) * 1000
    result = RagResult(
        answer=answer,
        contexts=[Passage(text=r["text"], score=r["rerank_score"], page=r["page"]) for r in kept],
        steps=steps,
        llm_calls=1,
        tokens=(len(context) + len(question) + len(answer)) // 4,
        latency_ms=latency_ms,
    )
    return result, kept, info


def ask(
    question: str,
    doc_id: str,
    *,
    candidate_k: int = 20,
    top_k: int = 5,
    **settings: object,
) -> RagResult:
    result, _rows, _info = ask_detailed(question, doc_id, candidate_k=candidate_k, top_k=top_k, **settings)
    return result
