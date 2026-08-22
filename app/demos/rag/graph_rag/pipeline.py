import asyncio
import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import LLMGraphTransformer, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_settings
from core.embeddings import LocalEmbeddings
from core.graph_db import clear_graph_doc, get_graph
from core.llm import get_chat_model
from core.pdf import extract_text
from core.tracing import get_callbacks
from core.types import IngestStats, Passage, RagResult

DEMO_TAG = "graph_rag"
CHUNK_LABEL = "Document"
VECTOR_INDEX_NAME = "graph_rag_vector_index"

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using the retrieved passages and the graph relationships \
below, taken from the entities those passages mention. Cite page numbers in square brackets, \
e.g. [p.3]. Prefer a graph relationship over a passage when the passage only implies it.

Retrieved passages:
{chunks}

Graph relationships (from entities mentioned in the passages above):
{graph}

Question: {question}
Answer:"""
)


class GraphRagIngestStats(IngestStats):
    """IngestStats plus what the graph build cost, so the page can show it."""

    extracted: int = 0
    over_cap: int = 0
    failed: int = 0
    nodes: int = 0
    relationships: int = 0
    llm_calls: int = 0
    tokens: int = 0


class GraphRagResult(RagResult):
    """RagResult plus the retrieved chunks and the expanded subgraph, for the page's graph view."""

    triples: list[dict] = []


def _chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}:{index}"


def _chunk_pdf(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> list[dict]:
    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        {"page": page_num, "text": chunk_text}
        for page_num, text in enumerate(parsed.pages, start=1)
        if text.strip()
        for chunk_text in splitter.split_text(text)
    ]


def _get_vector_store(graph) -> Neo4jVector:
    # graph= reuses the shared driver instead of opening a second connection to Aura.
    return Neo4jVector(
        embedding=LocalEmbeddings(),
        graph=graph,
        index_name=VECTOR_INDEX_NAME,
        node_label=CHUNK_LABEL,
        embedding_node_property="embedding",
        text_node_property="text",
        embedding_dimension=384,
    )


def _wait_index_online(graph, index_name: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rows = graph.query("SHOW INDEXES YIELD name, state WHERE name = $name", {"name": index_name})
        if rows and rows[0]["state"] == "ONLINE":
            return
        time.sleep(0.5)


async def _extract_all(transformer: LLMGraphTransformer, documents: list[Document], max_concurrency: int) -> list:
    # LLMGraphTransformer.convert_to_graph_documents() runs one document at a time; this
    # runs a bounded number concurrently instead, since extraction is the slow, costly step.
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(doc: Document):
        async with semaphore:
            return await transformer.aprocess_response(doc)

    return await asyncio.gather(*(_one(d) for d in documents), return_exceptions=True)


def ingest(pdf_bytes: bytes, doc_id: str) -> GraphRagIngestStats:
    start = time.monotonic()
    settings = get_settings()
    graph = get_graph()
    clear_graph_doc(graph, DEMO_TAG, doc_id)

    chunks = _chunk_pdf(pdf_bytes, settings.graph_rag_chunk_size, settings.graph_rag_chunk_overlap)
    if not chunks:
        return GraphRagIngestStats(doc_id=doc_id, chunks=0, latency_ms=(time.monotonic() - start) * 1000)

    extract_count = min(len(chunks), settings.graph_rag_max_chunks)
    over_cap = len(chunks) - extract_count
    to_extract = chunks[:extract_count]

    documents = [
        Document(
            page_content=c["text"],
            metadata={"id": _chunk_id(doc_id, i), "doc_id": doc_id, "demo": DEMO_TAG, "page": c["page"]},
        )
        for i, c in enumerate(to_extract)
    ]

    transformer = LLMGraphTransformer(llm=get_chat_model(fast=True))
    results = asyncio.run(_extract_all(transformer, documents, settings.graph_rag_extract_concurrency))

    graph_documents = []
    failed = 0
    node_count = 0
    rel_count = 0
    tokens = sum(len(d.page_content) for d in documents) // 4
    for doc, result in zip(documents, results):
        if isinstance(result, Exception):
            failed += 1
            continue
        result.source = doc
        for node in result.nodes:
            node.properties["doc_id"] = doc_id
            node.properties["demo"] = DEMO_TAG
        for rel in result.relationships:
            rel.properties["doc_id"] = doc_id
            rel.properties["demo"] = DEMO_TAG
        node_count += len(result.nodes)
        rel_count += len(result.relationships)
        graph_documents.append(result)

    if graph_documents:
        # include_source=True links every extracted entity back to its chunk via MENTIONS,
        # which is what lets ask() walk from a retrieved chunk into the graph around it.
        graph.add_graph_documents(graph_documents, include_source=True)

    vector_store = _get_vector_store(graph)
    vector_store.create_new_index()
    _wait_index_online(graph, VECTOR_INDEX_NAME)
    # Embedded regardless of whether extraction succeeded, so a failed chunk is still
    # retrievable (just without any graph neighbourhood) rather than silently dropped.
    vector_store.add_texts(
        [d.page_content for d in documents],
        metadatas=[d.metadata for d in documents],
        ids=[d.metadata["id"] for d in documents],
    )

    return GraphRagIngestStats(
        doc_id=doc_id,
        chunks=len(chunks),
        latency_ms=(time.monotonic() - start) * 1000,
        extracted=len(graph_documents),
        over_cap=over_cap,
        failed=failed,
        nodes=node_count,
        relationships=rel_count,
        llm_calls=len(documents),
        tokens=tokens,
    )


def _entity_type(labels: list[str]) -> str:
    return next((label for label in labels if label != CHUNK_LABEL), "Entity")


def graph_summary(doc_id: str, limit: int = 200) -> list[dict]:
    """The whole extracted entity graph for this document (no chunk nodes), for the
    'extracted graph' view."""
    graph = get_graph()
    rows = graph.query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.doc_id = $doc_id AND a.demo = $demo AND b.doc_id = $doc_id AND b.demo = $demo
          AND NOT a:Document AND NOT b:Document
        RETURN a.id AS source, labels(a) AS source_labels, type(r) AS rel,
               b.id AS target, labels(b) AS target_labels
        LIMIT $limit
        """,
        {"doc_id": doc_id, "demo": DEMO_TAG, "limit": limit},
    )
    return [
        {
            "source": r["source"],
            "source_type": _entity_type(r["source_labels"]),
            "rel": r["rel"],
            "target": r["target"],
            "target_type": _entity_type(r["target_labels"]),
        }
        for r in rows
    ]


def ask(question: str, doc_id: str, *, top_k: int | None = None, **_: object) -> GraphRagResult:
    start = time.monotonic()
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.graph_rag_top_k
    steps: list[str] = []

    graph = get_graph()
    vector_store = _get_vector_store(graph)
    hits = vector_store.similarity_search_with_score(question, k=top_k, filter={"doc_id": doc_id, "demo": DEMO_TAG})
    chunk_rows = [
        {"id": doc.metadata.get("id"), "text": doc.page_content, "page": doc.metadata.get("page"), "score": float(score)}
        for doc, score in hits
    ]
    steps.append(
        f"Neo4jVector similarity search (index '{VECTOR_INDEX_NAME}', k={top_k}) on :{CHUNK_LABEL} "
        f"nodes for this document returned {len(chunk_rows)} chunk(s)."
    )

    chunk_ids = [r["id"] for r in chunk_rows]
    expansion = (
        graph.query(
            """
            MATCH (c:Document)-[:MENTIONS]->(e)
            WHERE c.id IN $chunk_ids AND e.doc_id = $doc_id AND e.demo = $demo
            OPTIONAL MATCH (e)-[r]-(neighbor)
            WHERE neighbor.doc_id = $doc_id AND neighbor.demo = $demo AND NOT neighbor:Document
            RETURN DISTINCT e.id AS entity, labels(e) AS entity_labels, type(r) AS rel_type,
                   neighbor.id AS neighbor_id, labels(neighbor) AS neighbor_labels
            """,
            {"chunk_ids": chunk_ids, "doc_id": doc_id, "demo": DEMO_TAG},
        )
        if chunk_ids
        else []
    )
    triples = [
        {
            "source": r["entity"],
            "source_type": _entity_type(r["entity_labels"]),
            "rel": r["rel_type"],
            "target": r["neighbor_id"],
            "target_type": _entity_type(r["neighbor_labels"]),
        }
        for r in expansion
        if r["neighbor_id"] is not None
    ]
    mentioned = {r["entity"] for r in expansion}
    steps.append(
        f"Cypher expansion: {len(mentioned)} entit(y/ies) mentioned in those chunks, "
        f"{len(triples)} neighbouring relationship(s) one hop out."
    )

    chunk_context = "\n\n".join(f"[p.{r['page']}] {r['text']}" for r in chunk_rows) or "(no matching chunks)"
    graph_context = (
        "\n".join(f"({t['source']}:{t['source_type']}) -[{t['rel']}]-> ({t['target']}:{t['target_type']})" for t in triples)
        or "(no graph relationships found for the retrieved chunks)"
    )

    chain = ANSWER_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"chunks": chunk_context, "graph": graph_context, "question": question},
        config={"callbacks": get_callbacks(), "run_name": "graph_rag_ask"},
    )
    steps.append("Answered from the retrieved passages and the expanded graph relationships via an LCEL chain.")

    return GraphRagResult(
        answer=answer,
        contexts=[Passage(text=r["text"], score=r["score"], page=r["page"]) for r in chunk_rows],
        steps=steps,
        llm_calls=1,
        tokens=(len(chunk_context) + len(graph_context) + len(question) + len(answer)) // 4,
        latency_ms=(time.monotonic() - start) * 1000,
        triples=triples,
    )
