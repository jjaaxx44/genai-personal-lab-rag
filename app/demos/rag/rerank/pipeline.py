import time

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection
from sentence_transformers import CrossEncoder

from core.config import get_settings
from core.embeddings import LocalEmbeddings
from core.llm import get_chat_model
from core.mongo import clear_doc, ensure_indexes, get_collection
from core.pdf import extract_text
from core.tracing import get_callbacks
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_rerank"

PROMPT_TEMPLATE = """Answer the question using only the numbered context below. \
Cite the page number(s) you relied on in square brackets, e.g. [p.3].

{context}

Question: {question}
Answer:"""


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


def ask_detailed(
    question: str,
    doc_id: str,
    *,
    candidate_k: int = 20,
    top_k: int = 5,
    **_: object,
) -> tuple[RagResult, list[dict]]:
    """Runs the full pipeline and also returns the before/after ranking rows for the page."""
    start = time.monotonic()
    steps: list[str] = []
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    vector_store = _get_vector_store(collection)

    candidate_hits = vector_store.similarity_search_with_score(
        question, k=candidate_k, pre_filter={"doc_id": doc_id}
    )
    steps.append(f"Retriever (k={candidate_k}) on rag_rerank returned {len(candidate_hits)} candidate(s).")

    rerank_start = time.monotonic()
    pairs = [(question, doc.page_content) for doc, _ in candidate_hits]
    cross_scores = _get_reranker().predict(pairs) if pairs else []
    rerank_ms = (time.monotonic() - rerank_start) * 1000

    rows = [
        {
            "text": doc.page_content,
            "page": doc.metadata["page"],
            "pre_rank": pre_rank,
            "pre_score": pre_score,
            "cross_score": float(cross_score),
        }
        for pre_rank, ((doc, pre_score), cross_score) in enumerate(zip(candidate_hits, cross_scores), start=1)
    ]
    rows.sort(key=lambda r: r["cross_score"], reverse=True)
    rows = rows[:top_k]
    for post_rank, row in enumerate(rows, start=1):
        row["post_rank"] = post_rank
    steps.append(
        f"cross-encoder ({settings.rerank_model}) re-scored {len(candidate_hits)} candidate(s) "
        f"in {rerank_ms:.0f} ms, kept the top {len(rows)}."
    )

    context = "\n\n".join(f"[{i + 1}] (p.{r['page']}) {r['text']}" for i, r in enumerate(rows))
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"context": context, "question": question},
        config={"callbacks": get_callbacks(), "run_name": "rerank_ask"},
    )
    steps.append("Answered from the re-ranked context via an LCEL chain (prompt | chat model | parser).")

    latency_ms = (time.monotonic() - start) * 1000
    result = RagResult(
        answer=answer,
        contexts=[Passage(text=r["text"], score=r["cross_score"], page=r["page"]) for r in rows],
        steps=steps,
        llm_calls=1,
        tokens=(len(context) + len(question) + len(answer)) // 4,
        latency_ms=latency_ms,
    )
    return result, rows


def ask(question: str, doc_id: str, *, candidate_k: int = 20, top_k: int = 5, **settings: object) -> RagResult:
    result, _rows = ask_detailed(question, doc_id, candidate_k=candidate_k, top_k=top_k, **settings)
    return result
