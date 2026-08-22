import operator
import time
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pymongo.collection import Collection

from core.config import get_settings
from core.embeddings import LocalEmbeddings
from core.llm import get_chat_model
from core.mongo import clear_doc, ensure_indexes, get_collection
from core.pdf import extract_text
from core.tracing import get_callbacks
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_multi_hop"

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Break this question into a chain of hops: ordered sub-questions, each answerable by \
retrieving passages from the document, where a later sub-question may depend on an \
earlier one's answer. Use as few hops as the question genuinely needs, up to {max_hops}."""
)

HOP_PROMPT = ChatPromptTemplate.from_template(
    """Original question: {question}

Answers found in earlier hops:
{prior_hops}

Context retrieved for this hop:
{context}

Hop {hop_number} sub-question: {sub_question}

Answer this sub-question using only the context above. Cite page numbers in square \
brackets, e.g. [p.3]. If the context doesn't answer it, say so.
Answer:"""
)

COMBINE_PROMPT = ChatPromptTemplate.from_template(
    """Original question: {question}

Each hop's sub-question and answer, in order:
{hops}

Write one final answer to the original question, combining the hop answers above. \
Keep the page citations from the hop answers you rely on.
Answer:"""
)


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(description="Ordered hop sub-questions, earliest first.")


class MultiHopState(TypedDict):
    question: str
    doc_id: str
    top_k: int
    max_hops: int
    sub_questions: list[str]
    hop_index: int
    hops: Annotated[list[dict], operator.add]
    answer: str
    path: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]
    tokens: Annotated[int, operator.add]


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
        chunk_size=settings.multi_hop_chunk_size, chunk_overlap=settings.multi_hop_chunk_overlap
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
        _get_vector_store(collection).add_texts(texts, metadatas=metadatas)
        ensure_indexes(collection)

    return IngestStats(doc_id=doc_id, chunks=len(texts), latency_ms=(time.monotonic() - start) * 1000)


def _decompose(state: MultiHopState) -> dict:
    chain = DECOMPOSE_PROMPT | get_chat_model(fast=True).with_structured_output(SubQuestions, method="json_schema")
    output = chain.invoke(
        {"question": state["question"], "max_hops": state["max_hops"]}, config={"callbacks": get_callbacks()}
    )
    sub_questions = output.sub_questions[: state["max_hops"]] or [state["question"]]
    tokens = len(state["question"]) // 4
    return {"sub_questions": sub_questions, "hop_index": 0, "path": ["decompose"], "llm_calls": 1, "tokens": tokens}


def _hop(state: MultiHopState) -> dict:
    index = state["hop_index"]
    sub_question = state["sub_questions"][index]

    collection = get_collection(COLLECTION_NAME)
    hits = _get_vector_store(collection).similarity_search_with_score(
        sub_question, k=state["top_k"], pre_filter={"doc_id": state["doc_id"]}
    )
    passages = [{"text": doc.page_content, "page": doc.metadata["page"], "score": float(score)} for doc, score in hits]
    context = "\n\n".join(f"[p.{p['page']}] {p['text']}" for p in passages)

    prior_hops = (
        "\n".join(f"{i}. {h['question']} -> {h['answer']}" for i, h in enumerate(state["hops"], start=1))
        or "(none yet)"
    )
    chain = HOP_PROMPT | get_chat_model(fast=True) | StrOutputParser()
    answer = chain.invoke(
        {
            "question": state["question"],
            "prior_hops": prior_hops,
            "context": context or "(no passages retrieved)",
            "hop_number": index + 1,
            "sub_question": sub_question,
        },
        config={"callbacks": get_callbacks()},
    )
    tokens = (len(context) + len(prior_hops) + len(sub_question) + len(answer)) // 4
    entry = {"question": sub_question, "passages": passages, "context": context, "answer": answer}
    return {"hops": [entry], "hop_index": index + 1, "path": ["hop"], "llm_calls": 1, "tokens": tokens}


def _route_after_hop(state: MultiHopState) -> str:
    return "hop" if state["hop_index"] < len(state["sub_questions"]) else "combine"


def _combine(state: MultiHopState) -> dict:
    hops_text = "\n\n".join(
        f"Hop {i}: {h['question']}\nAnswer: {h['answer']}" for i, h in enumerate(state["hops"], start=1)
    )
    chain = COMBINE_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"question": state["question"], "hops": hops_text}, config={"callbacks": get_callbacks()}
    )
    tokens = (len(hops_text) + len(answer)) // 4
    return {"answer": answer, "path": ["combine"], "llm_calls": 1, "tokens": tokens}


def _build_graph():
    graph = StateGraph(MultiHopState)
    graph.add_node("decompose", _decompose)
    graph.add_node("hop", _hop)
    graph.add_node("combine", _combine)
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "hop")
    graph.add_conditional_edges("hop", _route_after_hop, {"hop": "hop", "combine": "combine"})
    graph.add_edge("combine", END)
    return graph.compile()


GRAPH = _build_graph()


def ask_detailed(
    question: str,
    doc_id: str,
    *,
    top_k: int = 5,
    max_hops: int = 2,
    **_: object,
) -> tuple[RagResult, MultiHopState]:
    """Runs the Multi-hop RAG graph and also returns the full end state for the page."""
    start = time.monotonic()
    state: MultiHopState = GRAPH.invoke(
        {
            "question": question,
            "doc_id": doc_id,
            "top_k": top_k,
            "max_hops": max_hops,
            "sub_questions": [],
            "hop_index": 0,
            "hops": [],
            "answer": "",
            "path": [],
            "llm_calls": 0,
            "tokens": 0,
        },
        config={
            "callbacks": get_callbacks(),
            "recursion_limit": 4 + max_hops * 2,
            "run_name": "multi_hop_ask",
        },
    )

    steps = [f"decompose: {len(state['sub_questions'])} hop(s): {state['sub_questions']}."]
    for i, hop in enumerate(state["hops"], start=1):
        steps.append(f"hop {i}: {hop['question']!r} -> {len(hop['passages'])} passage(s) -> {hop['answer'][:120]}")
    steps.append(f"combine: merged {len(state['hops'])} hop answer(s) into the final answer.")

    contexts = [
        Passage(text=p["text"], score=p["score"], page=p["page"]) for hop in state["hops"] for p in hop["passages"]
    ]

    result = RagResult(
        answer=state["answer"],
        contexts=contexts,
        steps=steps,
        llm_calls=state["llm_calls"],
        tokens=state["tokens"],
        latency_ms=(time.monotonic() - start) * 1000,
    )
    return result, state


def ask(question: str, doc_id: str, *, top_k: int = 5, max_hops: int = 2, **settings: object) -> RagResult:
    result, _state = ask_detailed(question, doc_id, top_k=top_k, max_hops=max_hops, **settings)
    return result
