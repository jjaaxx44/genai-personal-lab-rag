import re
import time

import numpy as np
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pymongo.collection import Collection

from core.config import get_settings
from core.embeddings import LocalEmbeddings, embed
from core.llm import complete
from core.mongo import clear_doc, ensure_indexes, get_collection
from core.pdf import extract_text
from core.tracing import observe
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_chunking"
STRATEGIES = ["fixed", "recursive", "sliding", "semantic"]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

PROMPT_TEMPLATE = """Answer the question using only the numbered context below. \
Cite the page number(s) you relied on in square brackets, e.g. [p.3].

{context}

Question: {question}
Answer:"""


def _nonempty_pages(pages: list[str]) -> tuple[list[str], list[dict]]:
    texts, metadatas = [], []
    for i, text in enumerate(pages, start=1):
        if text.strip():
            texts.append(text)
            metadatas.append({"page": i})
    return texts, metadatas


def _fixed_chunks(pages: list[str], size: int) -> list:
    texts, metadatas = _nonempty_pages(pages)
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=size, chunk_overlap=0)
    return splitter.create_documents(texts, metadatas=metadatas)


def _recursive_chunks(pages: list[str], size: int, overlap: int) -> list:
    texts, metadatas = _nonempty_pages(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.create_documents(texts, metadatas=metadatas)


def _sliding_chunks(pages: list[str], size: int, overlap: int) -> list:
    texts, metadatas = _nonempty_pages(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.create_documents(texts, metadatas=metadatas)


def _distance(a: list[float], b: list[float]) -> float:
    # embed() L2-normalizes, so cosine similarity is a plain dot product.
    return float(1 - np.dot(a, b))


def _semantic_chunks(pages: list[str], percentile: float) -> list[dict]:
    """Split where similarity between neighbouring sentences drops sharply.

    The breakpoint threshold is one distance-percentile pooled across the whole
    document, then applied while walking each page's own sentences -- a single
    page rarely has enough sentences for its own percentile to be meaningful,
    and chunks still never cross a page boundary.
    """
    page_sentences: list[list[str]] = [
        [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()] for text in pages
    ]

    all_sentences = [s for sentences in page_sentences for s in sentences]
    if not all_sentences:
        return []
    all_vectors = embed(all_sentences)

    vectors_by_page: list[list[list[float]]] = []
    cursor = 0
    for sentences in page_sentences:
        vectors_by_page.append(all_vectors[cursor : cursor + len(sentences)])
        cursor += len(sentences)

    pooled_distances = [
        _distance(vectors[i], vectors[i + 1])
        for vectors in vectors_by_page
        for i in range(len(vectors) - 1)
    ]
    threshold = float(np.percentile(pooled_distances, percentile)) if pooled_distances else float("inf")

    chunks: list[dict] = []
    for page_num, (sentences, vectors) in enumerate(zip(page_sentences, vectors_by_page), start=1):
        if not sentences:
            continue
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            if _distance(vectors[i - 1], vectors[i]) > threshold:
                chunks.append({"text": " ".join(current), "page": page_num})
                current = []
            current.append(sentences[i])
        if current:
            chunks.append({"text": " ".join(current), "page": page_num})
    return chunks


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

    strategy_docs = {
        "fixed": _fixed_chunks(parsed.pages, settings.chunking_chunk_size),
        "recursive": _recursive_chunks(
            parsed.pages, settings.chunking_chunk_size, settings.chunking_chunk_overlap
        ),
        "sliding": _sliding_chunks(
            parsed.pages, settings.chunking_chunk_size, settings.chunking_sliding_overlap
        ),
    }

    texts: list[str] = []
    metadatas: list[dict] = []
    for strategy, docs in strategy_docs.items():
        for doc in docs:
            texts.append(doc.page_content)
            metadatas.append({"doc_id": doc_id, "strategy": strategy, "page": doc.metadata["page"]})

    for chunk in _semantic_chunks(parsed.pages, settings.chunking_semantic_breakpoint_percentile):
        texts.append(chunk["text"])
        metadatas.append({"doc_id": doc_id, "strategy": "semantic", "page": chunk["page"]})

    if texts:
        vector_store = _get_vector_store(collection)
        vector_store.add_texts(texts, metadatas=metadatas)
        ensure_indexes(collection, filter_fields=("doc_id", "strategy"))

    return IngestStats(doc_id=doc_id, chunks=len(texts), latency_ms=(time.monotonic() - start) * 1000)


def strategy_stats(doc_id: str) -> dict[str, dict]:
    collection = get_collection(COLLECTION_NAME)
    stats: dict[str, dict] = {}
    for strategy in STRATEGIES:
        lengths = [len(d["text"]) for d in collection.find({"doc_id": doc_id, "strategy": strategy}, {"text": 1})]
        stats[strategy] = {
            "count": len(lengths),
            "avg_size": (sum(lengths) / len(lengths)) if lengths else 0,
        }
    return stats


@observe(name="chunking_ask")
def ask(question: str, doc_id: str, *, strategy: str = "recursive", top_k: int = 5, **_: object) -> RagResult:
    start = time.monotonic()
    steps: list[str] = []
    collection = get_collection(COLLECTION_NAME)
    vector_store = _get_vector_store(collection)

    hits = vector_store.similarity_search_with_score(
        question, k=top_k, pre_filter={"doc_id": doc_id, "strategy": strategy}
    )
    steps.append(f"$vectorSearch on rag_chunking (strategy={strategy}) returned {len(hits)} passage(s).")

    context = "\n\n".join(
        f"[{i + 1}] (p.{doc.metadata['page']}) {doc.page_content}" for i, (doc, _) in enumerate(hits)
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    answer = complete(prompt)
    steps.append(f"Answered from the '{strategy}' strategy's context via core.llm.complete().")

    latency_ms = (time.monotonic() - start) * 1000
    return RagResult(
        answer=answer,
        contexts=[
            Passage(text=doc.page_content, score=score, page=doc.metadata["page"]) for doc, score in hits
        ],
        steps=steps,
        llm_calls=1,
        # complete() only returns text, not usage metadata, so this is a
        # rough ~4-chars-per-token estimate for the metrics row, not an exact count.
        tokens=(len(prompt) + len(answer)) // 4,
        latency_ms=latency_ms,
    )
