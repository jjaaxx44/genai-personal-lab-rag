import time

from core.config import get_settings
from core.embeddings import embed
from core.llm import complete
from core.mongo import ensure_indexes, get_collection, text_search, vector_search
from core.pdf import extract_text
from core.tracing import observe
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_hybrid"
MODES = ["vector", "keyword", "fused"]

# Order matters: try to split on paragraph breaks before falling back to
# sentences, then words, then raw characters as a last resort.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

PROMPT_TEMPLATE = """Answer the question using only the numbered context below. \
Cite the page number(s) you relied on in square brackets, e.g. [p.3].

{context}

Question: {question}
Answer:"""


def _split_text(text: str, size: int, overlap: int, separators: list[str] = SEPARATORS) -> list[str]:
    """Recursively split on the first separator that fits, then stitch in overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size or not separators:
        return [text]

    sep, rest = separators[0], separators[1:]
    parts = list(text) if sep == "" else text.split(sep)

    pieces: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        piece = part + (sep if sep and i < len(parts) - 1 else "")
        if len(current) + len(piece) <= size:
            current += piece
            continue
        if current:
            pieces.append(current)
        if len(piece) > size:
            pieces.extend(_split_text(piece, size, overlap, rest))
            current = ""
        else:
            current = piece
    if current:
        pieces.append(current)

    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    return [pieces[0]] + [pieces[i - 1][-overlap:] + pieces[i] for i in range(1, len(pieces))]


def _chunk_pages(pages: list[str], size: int, overlap: int) -> list[dict]:
    chunks = []
    for page_num, text in enumerate(pages, start=1):
        for chunk_text in _split_text(text, size, overlap):
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "page": page_num})
    return chunks


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    *,
    vector_weight: float,
    rrf_k: int,
    top_k: int,
) -> list[dict]:
    """Weighted RRF: each side contributes 1/(rrf_k + rank), scaled by vector_weight."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, r in enumerate(vector_results, start=1):
        key = str(r["_id"])
        docs[key] = r
        scores[key] = scores.get(key, 0.0) + vector_weight * (1 / (rrf_k + rank))
    for rank, r in enumerate(keyword_results, start=1):
        key = str(r["_id"])
        docs[key] = r
        scores[key] = scores.get(key, 0.0) + (1 - vector_weight) * (1 / (rrf_k + rank))

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    fused = []
    for key in ranked_keys:
        doc = dict(docs[key])
        doc["score"] = scores[key]
        fused.append(doc)
    return fused


@observe(name="hybrid_ingest")
def ingest(pdf_bytes: bytes, doc_id: str) -> IngestStats:
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)

    parsed = extract_text(pdf_bytes)
    chunks = _chunk_pages(parsed.pages, settings.hybrid_chunk_size, settings.hybrid_chunk_overlap)

    collection.delete_many({"doc_id": doc_id})
    if chunks:
        vectors = embed([c["text"] for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk["doc_id"] = doc_id
            chunk["embedding"] = vector
        collection.insert_many(chunks)
        ensure_indexes(collection, vector_dims=len(vectors[0]), with_text_index=True)

    return IngestStats(doc_id=doc_id, chunks=len(chunks), latency_ms=(time.monotonic() - start) * 1000)


@observe(name="hybrid_ask")
def ask(
    question: str,
    doc_id: str,
    *,
    mode: str = "fused",
    top_k: int = 5,
    vector_weight: float = 0.5,
    **_: object,
) -> RagResult:
    start = time.monotonic()
    steps: list[str] = []
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)

    # Fusion needs a wider candidate pool than top_k from each side to have
    # something worth re-ranking; vector/keyword-only modes just truncate it.
    pool = max(top_k * 4, 20)

    query_vector = embed([question])[0]
    vector_results = vector_search(collection, query_vector, doc_id=doc_id, top_k=pool)
    steps.append(f"$vectorSearch on rag_hybrid returned {len(vector_results)} candidate(s).")

    keyword_results = text_search(collection, question, doc_id=doc_id, top_k=pool)
    steps.append(f"$search (BM25) on rag_hybrid returned {len(keyword_results)} candidate(s).")

    if mode == "vector":
        results = vector_results[:top_k]
        steps.append(f"Mode 'vector': kept the top {len(results)} by vector score alone.")
    elif mode == "keyword":
        results = keyword_results[:top_k]
        steps.append(f"Mode 'keyword': kept the top {len(results)} by BM25 score alone.")
    else:
        results = _reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            vector_weight=vector_weight,
            rrf_k=settings.hybrid_rrf_k,
            top_k=top_k,
        )
        steps.append(
            f"Mode 'fused': weighted reciprocal rank fusion (vector_weight={vector_weight:.1f}, "
            f"rrf_k={settings.hybrid_rrf_k}) kept the top {len(results)}."
        )

    context = "\n\n".join(f"[{i + 1}] (p.{r['page']}) {r['text']}" for i, r in enumerate(results))
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    answer = complete(prompt)
    steps.append(f"Answered from the '{mode}' context via core.llm.complete().")

    latency_ms = (time.monotonic() - start) * 1000
    return RagResult(
        answer=answer,
        contexts=[Passage(text=r["text"], score=r["score"], page=r["page"]) for r in results],
        steps=steps,
        llm_calls=1,
        # complete() only returns text, not usage metadata, so this is a
        # rough ~4-chars-per-token estimate for the metrics row, not an exact count.
        tokens=(len(prompt) + len(answer)) // 4,
        latency_ms=latency_ms,
    )
