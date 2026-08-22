import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection

from core.config import get_settings
from core.embeddings import LocalEmbeddings
from core.llm import get_chat_model
from core.mongo import clear_doc, ensure_indexes, get_collection, insert_passages
from core.pdf import extract_text
from core.tracing import get_callbacks
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_agentic"

SYSTEM_PROMPT = """You answer questions about one uploaded document. You have no prior \
knowledge of this specific document's contents -- you cannot see it, guess at it, or \
infer it from its filename. The only way to learn what is in it is to call a tool.

Any question that could plausibly be about the document's subject, content, structure \
or claims requires calling search_document at least once before you answer -- call it \
again with a different query if the first search doesn't cover everything the question \
needs. Use get_page when a retrieved passage needs more of its surrounding context to \
make sense (e.g. a table, a list that continues, or a figure caption).

Skip tool calls only for messages that are not about the document at all: greetings, \
thanks, and other small talk.

Never answer a document question from general knowledge about its topic. If the tools \
don't return enough to answer, say so instead of guessing.

When you answer from the document, cite the page number(s) you relied on in square \
brackets, e.g. [p.3]."""


def _get_vector_store(collection: Collection) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection,
        LocalEmbeddings(),
        index_name="vector_index",
        auto_create_index=False,
    )


def ingest(pdf_bytes: bytes, doc_id: str) -> IngestStats:
    """Stores two kinds of record in rag_agentic: embedded chunks (kind="chunk"), for
    search_document, and whole page texts (kind="page", no embedding), for get_page."""
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    clear_doc(collection, doc_id)

    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.agentic_chunk_size, chunk_overlap=settings.agentic_chunk_overlap
    )

    texts: list[str] = []
    metadatas: list[dict] = []
    page_docs: list[dict] = []
    for page_num, text in enumerate(parsed.pages, start=1):
        if not text.strip():
            continue
        page_docs.append({"doc_id": doc_id, "kind": "page", "page": page_num, "text": text})
        for chunk_text in splitter.split_text(text):
            texts.append(chunk_text)
            metadatas.append({"doc_id": doc_id, "kind": "chunk", "page": page_num})

    if texts:
        _get_vector_store(collection).add_texts(texts, metadatas=metadatas)
    insert_passages(collection, page_docs)
    collection.insert_one({"doc_id": doc_id, "kind": "meta", "page_count": parsed.page_count})

    if texts:
        ensure_indexes(collection)

    return IngestStats(doc_id=doc_id, chunks=len(texts), latency_ms=(time.monotonic() - start) * 1000)


def _build_tools(collection: Collection, vector_store: MongoDBAtlasVectorSearch, doc_id: str, top_k: int):
    """Builds the two tools bound to this question's doc_id, and a shared trace list the
    page can render afterwards: one entry per tool call, plus the passages it surfaced."""
    settings = get_settings()
    trace: list[dict] = []
    passages: dict[tuple[int, str], Passage] = {}

    @tool
    def search_document(query: str) -> str:
        """Search the uploaded document for passages relevant to a query. Returns the
        top matching passages with their page numbers. Call again with a different
        query if the first search doesn't cover everything the question needs."""
        hits = vector_store.similarity_search_with_score(query, k=top_k, pre_filter={"doc_id": doc_id})
        for doc, score in hits:
            page = doc.metadata["page"]
            passages[(page, doc.page_content)] = Passage(text=doc.page_content, score=float(score), page=page)
        summary = ", ".join(f"p.{doc.metadata['page']} ({score:.3f})" for doc, score in hits) or "none"
        trace.append({"tool": "search_document", "input": query, "summary": f"{len(hits)} passage(s): {summary}"})
        if not hits:
            return "No relevant passages found."
        return "\n\n".join(f"[p.{doc.metadata['page']}] {doc.page_content}" for doc, _ in hits)

    @tool
    def get_page(page: int) -> str:
        """Read the full text of one page of the document by page number, when a
        retrieved passage needs more of its surrounding context."""
        record = collection.find_one({"doc_id": doc_id, "kind": "page", "page": page}, {"_id": 0, "text": 1})
        if record is None:
            trace.append({"tool": "get_page", "input": page, "summary": "page not found"})
            meta = collection.find_one({"doc_id": doc_id, "kind": "meta"}, {"_id": 0, "page_count": 1})
            page_count = meta["page_count"] if meta else "unknown"
            return f"Page {page} not found. This document has {page_count} page(s)."
        text = record["text"][: settings.agentic_page_char_cap]
        trace.append({"tool": "get_page", "input": page, "summary": f"{len(text)} char(s) returned"})
        return text

    return [search_document, get_page], trace, passages


def ask_detailed(
    question: str,
    doc_id: str,
    *,
    top_k: int = 5,
    **_: object,
) -> tuple[RagResult, list[dict]]:
    """Runs the agent loop and also returns the raw tool-call trace for the page."""
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    vector_store = _get_vector_store(collection)
    tools, trace, passages = _build_tools(collection, vector_store, doc_id, top_k)

    agent = create_agent(get_chat_model(), tools=tools, system_prompt=SYSTEM_PROMPT)
    state = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={
            "callbacks": get_callbacks(),
            "recursion_limit": settings.agentic_recursion_limit,
            "run_name": "agentic_ask",
        },
    )
    messages = state["messages"]

    llm_calls = sum(1 for m in messages if isinstance(m, AIMessage))
    tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in messages if isinstance(m, AIMessage))
    if not tokens:
        tokens = sum(len(m.text) for m in messages) // 4

    # `trace` (built inside the tool closures) is already in call order, so it doubles
    # as the human-readable step list -- no need to re-derive it from `messages`.
    steps = [f"{entry['tool']}({entry['input']!r}) -> {entry['summary']}" for entry in trace]
    if not trace:
        steps.append("No tool calls -- answered directly from the question, without consulting the document.")

    answer = messages[-1].text if messages else ""

    result = RagResult(
        answer=answer,
        contexts=sorted(passages.values(), key=lambda p: p.score, reverse=True),
        steps=steps,
        llm_calls=llm_calls,
        tokens=tokens,
        latency_ms=(time.monotonic() - start) * 1000,
    )
    return result, trace


def ask(question: str, doc_id: str, *, top_k: int = 5, **settings: object) -> RagResult:
    result, _trace = ask_detailed(question, doc_id, top_k=top_k, **settings)
    return result
