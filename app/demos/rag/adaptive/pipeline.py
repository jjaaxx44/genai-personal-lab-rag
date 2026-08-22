import operator
import time
from typing import Annotated, Literal, TypedDict

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

COLLECTION_NAME = "rag_adaptive"

ROUTE_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Classify this question into exactly one path:
- "no_retrieval": greetings, small talk, or a general-knowledge question that doesn't \
need the uploaded document at all.
- "single_step": a factual question that one round of retrieving passages from the \
document can answer.
- "multi_step": a question that needs comparing, combining, or connecting information \
from multiple, possibly distant parts of the document -- one retrieval pass is unlikely \
to cover it.

Give one sentence explaining the choice."""
)

NO_RETRIEVAL_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question directly, without consulting any document.

Question: {question}
Answer:"""
)

SINGLE_GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. Cite page numbers in square \
brackets, e.g. [p.3], for anything drawn from it. If the context doesn't contain the \
answer, say so instead of guessing.

Context:
{context}

Question: {question}
Answer:"""
)

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Break this question into 2 to 3 ordered sub-questions that can each be answered by \
retrieving passages from the document, where a later sub-question may depend on an \
earlier one's answer. List them in the order they should be answered."""
)

HOP_PROMPT = ChatPromptTemplate.from_template(
    """Original question: {question}

Sub-answers found so far:
{prior_hops}

Context retrieved for this sub-question:
{context}

Sub-question: {sub_question}

Answer this sub-question using only the context above. Cite page numbers in square \
brackets, e.g. [p.3]. If the context doesn't answer it, say so.
Answer:"""
)

COMBINE_PROMPT = ChatPromptTemplate.from_template(
    """Original question: {question}

Each sub-question was answered in turn:
{hops}

Write one final answer to the original question, combining the sub-answers above. \
Keep the page citations from the sub-answers you rely on.
Answer:"""
)


class RouteDecision(BaseModel):
    route: Literal["no_retrieval", "single_step", "multi_step"]
    reason: str


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(description="2 to 3 ordered sub-questions, earliest first.")


class AdaptiveState(TypedDict):
    question: str
    doc_id: str
    top_k: int
    max_subquestions: int
    route: str
    route_reason: str
    passages: list[dict]
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


def _retrieve(question: str, doc_id: str, top_k: int) -> list[dict]:
    collection = get_collection(COLLECTION_NAME)
    hits = _get_vector_store(collection).similarity_search_with_score(
        question, k=top_k, pre_filter={"doc_id": doc_id}
    )
    return [{"text": doc.page_content, "page": doc.metadata["page"], "score": float(score)} for doc, score in hits]


def ingest(pdf_bytes: bytes, doc_id: str) -> IngestStats:
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    clear_doc(collection, doc_id)

    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.adaptive_chunk_size, chunk_overlap=settings.adaptive_chunk_overlap
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


def _route(state: AdaptiveState) -> dict:
    chain = ROUTE_PROMPT | get_chat_model(fast=True).with_structured_output(RouteDecision, method="json_schema")
    decision = chain.invoke({"question": state["question"]}, config={"callbacks": get_callbacks()})
    tokens = len(state["question"]) // 4
    return {
        "route": decision.route,
        "route_reason": decision.reason,
        "path": ["route"],
        "llm_calls": 1,
        "tokens": tokens,
    }


def _route_after_route(state: AdaptiveState) -> str:
    return state["route"]


def _no_retrieval_generate(state: AdaptiveState) -> dict:
    chain = NO_RETRIEVAL_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke({"question": state["question"]}, config={"callbacks": get_callbacks()})
    tokens = (len(state["question"]) + len(answer)) // 4
    return {"answer": answer, "path": ["no_retrieval_generate"], "llm_calls": 1, "tokens": tokens}


def _single_retrieve(state: AdaptiveState) -> dict:
    passages = _retrieve(state["question"], state["doc_id"], state["top_k"])
    return {"passages": passages, "path": ["single_retrieve"]}


def _single_generate(state: AdaptiveState) -> dict:
    context = "\n\n".join(f"[p.{p['page']}] {p['text']}" for p in state["passages"])
    chain = SINGLE_GENERATE_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"context": context or "(no passages retrieved)", "question": state["question"]},
        config={"callbacks": get_callbacks()},
    )
    tokens = (len(context) + len(state["question"]) + len(answer)) // 4
    return {"answer": answer, "path": ["single_generate"], "llm_calls": 1, "tokens": tokens}


def _decompose(state: AdaptiveState) -> dict:
    chain = DECOMPOSE_PROMPT | get_chat_model(fast=True).with_structured_output(SubQuestions, method="json_schema")
    output = chain.invoke({"question": state["question"]}, config={"callbacks": get_callbacks()})
    sub_questions = output.sub_questions[: state["max_subquestions"]]
    tokens = len(state["question"]) // 4
    return {
        "sub_questions": sub_questions,
        "hop_index": 0,
        "path": ["decompose"],
        "llm_calls": 1,
        "tokens": tokens,
    }


def _hop(state: AdaptiveState) -> dict:
    index = state["hop_index"]
    sub_question = state["sub_questions"][index]
    passages = _retrieve(sub_question, state["doc_id"], state["top_k"])
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
            "sub_question": sub_question,
        },
        config={"callbacks": get_callbacks()},
    )
    tokens = (len(context) + len(prior_hops) + len(sub_question) + len(answer)) // 4
    entry = {"question": sub_question, "passages": passages, "answer": answer}
    return {
        "hops": [entry],
        "hop_index": index + 1,
        "path": ["hop"],
        "llm_calls": 1,
        "tokens": tokens,
    }


def _route_after_hop(state: AdaptiveState) -> str:
    return "hop" if state["hop_index"] < len(state["sub_questions"]) else "combine"


def _combine(state: AdaptiveState) -> dict:
    hops_text = "\n\n".join(
        f"Sub-question {i}: {h['question']}\nAnswer: {h['answer']}" for i, h in enumerate(state["hops"], start=1)
    )
    chain = COMBINE_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"question": state["question"], "hops": hops_text}, config={"callbacks": get_callbacks()}
    )
    tokens = (len(hops_text) + len(answer)) // 4
    return {"answer": answer, "path": ["combine"], "llm_calls": 1, "tokens": tokens}


def _build_graph():
    graph = StateGraph(AdaptiveState)
    graph.add_node("route", _route)
    graph.add_node("no_retrieval_generate", _no_retrieval_generate)
    graph.add_node("single_retrieve", _single_retrieve)
    graph.add_node("single_generate", _single_generate)
    graph.add_node("decompose", _decompose)
    graph.add_node("hop", _hop)
    graph.add_node("combine", _combine)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        _route_after_route,
        {
            "no_retrieval": "no_retrieval_generate",
            "single_step": "single_retrieve",
            "multi_step": "decompose",
        },
    )
    graph.add_edge("no_retrieval_generate", END)
    graph.add_edge("single_retrieve", "single_generate")
    graph.add_edge("single_generate", END)
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
    max_subquestions: int = 3,
    **_: object,
) -> tuple[RagResult, AdaptiveState]:
    """Runs the Adaptive RAG graph and also returns the full end state for the page."""
    start = time.monotonic()
    state: AdaptiveState = GRAPH.invoke(
        {
            "question": question,
            "doc_id": doc_id,
            "top_k": top_k,
            "max_subquestions": max_subquestions,
            "route": "",
            "route_reason": "",
            "passages": [],
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
            "recursion_limit": 6 + max_subquestions * 2,
            "run_name": "adaptive_ask",
        },
    )

    steps = [f"route: {state['route']} ({state['route_reason']})."]
    if state["route"] == "no_retrieval":
        steps.append("generate: answered directly, no document context.")
    elif state["route"] == "single_step":
        steps.append(f"retrieve: found {len(state['passages'])} passage(s) in rag_adaptive.")
        steps.append("generate: answered from the retrieved passages.")
    else:
        steps.append(f"decompose: {len(state['sub_questions'])} sub-question(s): {state['sub_questions']}.")
        for i, hop in enumerate(state["hops"], start=1):
            steps.append(
                f"hop {i}: {hop['question']!r} -> {len(hop['passages'])} passage(s) -> {hop['answer'][:120]}"
            )
        steps.append(f"combine: merged {len(state['hops'])} sub-answer(s) into the final answer.")

    if state["route"] == "single_step":
        contexts = [Passage(text=p["text"], score=p["score"], page=p["page"]) for p in state["passages"]]
    elif state["route"] == "multi_step":
        contexts = [
            Passage(text=p["text"], score=p["score"], page=p["page"])
            for hop in state["hops"]
            for p in hop["passages"]
        ]
    else:
        contexts = []

    result = RagResult(
        answer=state["answer"],
        contexts=contexts,
        steps=steps,
        llm_calls=state["llm_calls"],
        tokens=state["tokens"],
        latency_ms=(time.monotonic() - start) * 1000,
    )
    return result, state


def ask(question: str, doc_id: str, *, top_k: int = 5, max_subquestions: int = 3, **settings: object) -> RagResult:
    result, _state = ask_detailed(question, doc_id, top_k=top_k, max_subquestions=max_subquestions, **settings)
    return result
