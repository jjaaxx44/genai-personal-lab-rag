import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pymongo.collection import Collection

from core.config import get_settings
from core.embeddings import LocalEmbeddings
from core.llm import get_chat_model
from core.mongo import clear_doc, ensure_indexes, get_collection
from core.pdf import extract_text
from core.tracing import get_callbacks, observe
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_contextual"
VARIANTS = ["plain", "contextual"]

# Contexts survive a re-ingest here, keyed by passage hash, so a retry after a rate
# limit only pays for the passages that failed. data/ is the compose volume.
CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "contextual_cache"

# Calls sent in parallel per .batch() wave. Kept low: every call carries the whole
# document, and free tiers cap tokens per minute as well as requests.
MAX_CONCURRENCY = 2

CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """<document>
{document}
</document>

Here are passages taken from the document above:

{passages}

For each passage, write 1-2 short sentences that situate it within the overall document, \
to improve search retrieval of the passage. Name the section, subject, product, person or \
time period the passage belongs to whenever the passage itself leaves it implicit. \
Return one entry per passage, using the passage's index. Write only the context, \
do not repeat the passage."""
)

ANSWER_PROMPT = """Answer the question using only the numbered context below. \
Cite the page number(s) you relied on in square brackets, e.g. [p.3].

{context}

Question: {question}
Answer:"""


class PassageContext(BaseModel):
    index: int = Field(description="The index attribute of the passage this context is for.")
    context: str = Field(description="1-2 sentences situating the passage within the document.")


class PassageContexts(BaseModel):
    # min_length becomes minItems in the JSON schema. Without it, a local 7B model given a
    # long document answers with a schema-valid but empty list for most groups.
    contexts: list[PassageContext] = Field(min_length=1)


class ContextualIngestStats(IngestStats):
    """IngestStats plus what enrichment cost, so the page can show it."""

    generated: int = 0
    cached: int = 0
    failed: int = 0
    over_cap: int = 0
    llm_calls: int = 0
    tokens: int = 0
    doc_truncated: bool = False
    timings: dict[str, float] = {}


def _get_vector_store(collection: Collection) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection,
        LocalEmbeddings(),
        index_name="vector_index",
        auto_create_index=False,
    )


def _passage_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(doc_id: str) -> Path:
    return CACHE_DIR / f"{doc_id}.json"


def _load_cache(doc_id: str) -> dict[str, str]:
    path = _cache_path(doc_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(doc_id: str, cache: dict[str, str]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(doc_id).write_text(json.dumps(cache))
    except OSError:
        pass  # The cache is an optimisation; losing it only costs a re-run's LLM calls.


def clear_cache(doc_id: str) -> None:
    _cache_path(doc_id).unlink(missing_ok=True)


def _format_passages(texts: list[str]) -> str:
    return "\n\n".join(f'<passage index="{n}">\n{text}\n</passage>' for n, text in enumerate(texts, start=1))


@observe(name="contextual_ingest")
def ingest(
    pdf_bytes: bytes,
    doc_id: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> ContextualIngestStats:
    """Stores every passage twice: as-is (variant=plain) and with an LLM-written context
    prepended (variant=contextual). Only the prepended context differs between the two."""
    start = time.monotonic()
    timings: dict[str, float] = {}
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    clear_doc(collection, doc_id)

    stage_start = time.monotonic()
    parsed = extract_text(pdf_bytes)
    timings["parse"] = time.monotonic() - stage_start

    stage_start = time.monotonic()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.contextual_chunk_size, chunk_overlap=settings.contextual_chunk_overlap
    )
    chunks: list[dict] = [
        {"page": page_num, "text": chunk_text}
        for page_num, text in enumerate(parsed.pages, start=1)
        if text.strip()
        for chunk_text in splitter.split_text(text)
    ]
    timings["chunk"] = time.monotonic() - stage_start

    if not chunks:
        return ContextualIngestStats(doc_id=doc_id, chunks=0, latency_ms=(time.monotonic() - start) * 1000)

    # Contextualize: cached passages first, then the rest in groups of
    # contextual_chunks_per_call, one LLM call per group, sent as .batch() waves.
    stage_start = time.monotonic()
    document = "\n\n".join(parsed.pages).strip()
    doc_truncated = len(document) > settings.contextual_doc_max_chars
    document = document[: settings.contextual_doc_max_chars]

    cache = _load_cache(doc_id)
    enrich_count = min(len(chunks), settings.contextual_max_chunks)
    for chunk in chunks[enrich_count:]:
        chunk["context"], chunk["context_status"] = "", "over_cap"

    pending: list[int] = []
    for i, chunk in enumerate(chunks[:enrich_count]):
        cached = cache.get(_passage_hash(chunk["text"]))
        if cached:
            chunk["context"], chunk["context_status"] = cached, "cached"
        else:
            pending.append(i)

    per_call = max(1, settings.contextual_chunks_per_call)
    groups = [pending[i : i + per_call] for i in range(0, len(pending), per_call)]
    llm_calls = 0
    tokens = 0
    if on_progress:
        on_progress(0, len(groups))

    if groups:
        # method="json_schema" for the same reason as the evaluation demo: Groq's
        # gpt-oss models sometimes skip a forced tool call; json_schema works everywhere.
        chain = CONTEXT_PROMPT | get_chat_model(fast=True).with_structured_output(
            PassageContexts, method="json_schema"
        )
        for wave_start in range(0, len(groups), MAX_CONCURRENCY):
            wave = groups[wave_start : wave_start + MAX_CONCURRENCY]
            inputs = [
                {"document": document, "passages": _format_passages([chunks[i]["text"] for i in group])}
                for group in wave
            ]
            outputs = chain.batch(
                inputs,
                config={"callbacks": get_callbacks(), "max_concurrency": MAX_CONCURRENCY},
                return_exceptions=True,
            )
            for group, prompt_input, output in zip(wave, inputs, outputs):
                llm_calls += 1
                tokens += (len(prompt_input["document"]) + len(prompt_input["passages"])) // 4
                by_index: dict[int, str] = {}
                if isinstance(output, PassageContexts):
                    by_index = {c.index: c.context.strip() for c in output.contexts}
                    tokens += sum(len(c) for c in by_index.values()) // 4
                for n, i in enumerate(group, start=1):
                    context = by_index.get(n, "")
                    if context:
                        chunks[i]["context"], chunks[i]["context_status"] = context, "generated"
                        cache[_passage_hash(chunks[i]["text"])] = context
                    else:
                        chunks[i]["context"], chunks[i]["context_status"] = "", "failed"
            # Saved per wave, so an interrupted ingest keeps what it already paid for.
            _save_cache(doc_id, cache)
            if on_progress:
                on_progress(min(wave_start + MAX_CONCURRENCY, len(groups)), len(groups))
    timings["contextualize"] = time.monotonic() - stage_start

    stage_start = time.monotonic()
    texts: list[str] = []
    metadatas: list[dict] = []
    for chunk_index, chunk in enumerate(chunks):
        for variant in VARIANTS:
            with_context = variant == "contextual" and chunk["context"]
            # `text` is what gets embedded and BM25-indexed; the original passage and
            # its context are kept apart too, so the page can show them separately.
            texts.append(f"{chunk['context']}\n\n{chunk['text']}" if with_context else chunk["text"])
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "variant": variant,
                    "chunk_index": chunk_index,
                    "page": chunk["page"],
                    "original_text": chunk["text"],
                    "context": chunk["context"] if variant == "contextual" else "",
                    "context_status": chunk["context_status"] if variant == "contextual" else "none",
                }
            )
    _get_vector_store(collection).add_texts(texts, metadatas=metadatas)
    timings["embed"] = time.monotonic() - stage_start

    stage_start = time.monotonic()
    ensure_indexes(collection, filter_fields=("doc_id", "variant"), with_text_index=True)
    timings["index"] = time.monotonic() - stage_start

    statuses = [chunk["context_status"] for chunk in chunks]
    return ContextualIngestStats(
        doc_id=doc_id,
        chunks=len(chunks),
        latency_ms=(time.monotonic() - start) * 1000,
        generated=statuses.count("generated"),
        cached=statuses.count("cached"),
        failed=statuses.count("failed"),
        over_cap=statuses.count("over_cap"),
        llm_calls=llm_calls,
        tokens=tokens,
        doc_truncated=doc_truncated,
        timings=timings,
    )


def enrichment_summary(doc_id: str) -> dict[str, int]:
    """Passage counts per context_status, read back from MongoDB so it survives reruns."""
    collection = get_collection(COLLECTION_NAME)
    pipeline = [
        {"$match": {"doc_id": doc_id, "variant": "contextual"}},
        {"$group": {"_id": "$context_status", "count": {"$sum": 1}}},
    ]
    return {row["_id"]: row["count"] for row in collection.aggregate(pipeline)}


def sample_enrichment(doc_id: str, n: int = 3) -> list[dict]:
    """The first few passages that received a context, for inspecting what was added."""
    collection = get_collection(COLLECTION_NAME)
    cursor = (
        collection.find(
            {"doc_id": doc_id, "variant": "contextual", "context": {"$ne": ""}},
            {"_id": 0, "page": 1, "chunk_index": 1, "original_text": 1, "context": 1},
        )
        .sort("chunk_index", 1)
        .limit(n)
    )
    return list(cursor)


@observe(name="contextual_ask")
def ask_detailed(
    question: str,
    doc_id: str,
    *,
    variant: str = "contextual",
    top_k: int = 5,
    **_: object,
) -> tuple[RagResult, list[dict], dict[str, float]]:
    """Runs the full pipeline and also returns per-passage score rows and stage timings."""
    start = time.monotonic()
    steps: list[str] = []
    collection = get_collection(COLLECTION_NAME)

    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=_get_vector_store(collection),
        search_index_name="text_index",
        k=top_k,
        pre_filter={"doc_id": {"$eq": doc_id}, "variant": {"$eq": variant}},
        auto_create_index=False,
    )
    docs = retriever.invoke(question, config={"callbacks": get_callbacks()})
    retrieve_s = time.monotonic() - start
    steps.append(
        f"MongoDBAtlasHybridSearchRetriever (k={top_k}) on rag_contextual, variant='{variant}': "
        f"$vectorSearch and $search (BM25) fused by reciprocal rank in one aggregation, "
        f"returned {len(docs)} passage(s)."
    )

    rows = [
        {
            "rank": rank,
            "chunk_index": doc.metadata["chunk_index"],
            "page": doc.metadata["page"],
            "text": doc.metadata["original_text"],
            "context": doc.metadata.get("context", ""),
            "score": float(doc.metadata.get("score", 0.0)),
            "vector_score": float(doc.metadata.get("vector_score", 0.0)),
            "fulltext_score": float(doc.metadata.get("fulltext_score", 0.0)),
        }
        for rank, doc in enumerate(docs, start=1)
    ]

    # The answer prompt gets the original passages in both variants, so any difference
    # between the two answers comes from which passages were retrieved, not from the
    # extra context sentences.
    answer_start = time.monotonic()
    context = "\n\n".join(f"[{i + 1}] (p.{r['page']}) {r['text']}" for i, r in enumerate(rows))
    chain = ChatPromptTemplate.from_template(ANSWER_PROMPT) | get_chat_model() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}, config={"callbacks": get_callbacks()})
    answer_s = time.monotonic() - answer_start
    steps.append("Answered from the original passage text (without the added context) via an LCEL chain.")

    result = RagResult(
        answer=answer,
        contexts=[Passage(text=r["text"], score=r["score"], page=r["page"]) for r in rows],
        steps=steps,
        llm_calls=1,
        tokens=(len(context) + len(question) + len(answer)) // 4,
        latency_ms=(time.monotonic() - start) * 1000,
    )
    return result, rows, {"retrieve": retrieve_s, "answer": answer_s}


def ask(question: str, doc_id: str, *, variant: str = "contextual", top_k: int = 5, **settings: object) -> RagResult:
    result, _rows, _timings = ask_detailed(question, doc_id, variant=variant, top_k=top_k, **settings)
    return result
