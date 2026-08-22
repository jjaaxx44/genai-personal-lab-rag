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

COLLECTION_NAME = "rag_self_rag"

ROUTE_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Decide whether answering this well requires retrieving passages from the uploaded \
document, or whether it can be answered directly without it (greetings, small talk, \
or a general question unrelated to the document)."""
)

FILTER_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Passages retrieved for this question:

{passages}

For each passage, grade whether it is "relevant" or "irrelevant" to answering the \
question. Return one grade per passage, using its index."""
)

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below, if any is given. Cite page \
numbers in square brackets, e.g. [p.3], for anything drawn from it. If the context \
doesn't contain the answer, say so instead of guessing.
{revision_note}
Context:
{context}

Question: {question}
Answer:"""
)

CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Context given to the answer-writer:
{context}

Answer written:
{answer}

Judge two things about the answer:
1. supported: is every claim backed by the context above -- fully_supported, \
partially_supported, or not_supported?
2. useful: does it actually address the question, regardless of support -- \
useful or not_useful?

Give one sentence of feedback the answer-writer could use to improve it."""
)

USEFULNESS_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Answer written (no document context was used for this one):
{answer}

Judge whether the answer actually addresses the question -- useful or not_useful. \
Give one sentence of feedback the answer-writer could use to improve it."""
)


class RetrievalDecision(BaseModel):
    needs_retrieval: bool
    reason: str


class PassageGrade(BaseModel):
    index: int = Field(description="The index attribute of the passage this grade is for.")
    grade: Literal["relevant", "irrelevant"]


class PassageGrades(BaseModel):
    grades: list[PassageGrade]


class CritiqueResult(BaseModel):
    supported: Literal["fully_supported", "partially_supported", "not_supported"]
    useful: Literal["useful", "not_useful"]
    feedback: str


class UsefulnessResult(BaseModel):
    useful: Literal["useful", "not_useful"]
    feedback: str


class SelfRagState(TypedDict):
    question: str
    doc_id: str
    top_k: int
    max_retries: int
    needs_retrieval: bool
    route_reason: str
    passages: list[dict]
    relevant_passages: list[dict]
    context: str
    answer: str
    feedback: str
    decision: str
    attempts: Annotated[int, operator.add]
    attempt_log: Annotated[list[dict], operator.add]
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


def _format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f'<passage index="{i}">\n{p["text"]}\n</passage>' for i, p in enumerate(passages, start=1))


def ingest(pdf_bytes: bytes, doc_id: str) -> IngestStats:
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    clear_doc(collection, doc_id)

    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.self_rag_chunk_size, chunk_overlap=settings.self_rag_chunk_overlap
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


def _route(state: SelfRagState) -> dict:
    chain = ROUTE_PROMPT | get_chat_model(fast=True).with_structured_output(
        RetrievalDecision, method="json_schema"
    )
    decision = chain.invoke({"question": state["question"]}, config={"callbacks": get_callbacks()})
    tokens = len(state["question"]) // 4
    return {
        "needs_retrieval": decision.needs_retrieval,
        "route_reason": decision.reason,
        "path": ["route"],
        "llm_calls": 1,
        "tokens": tokens,
    }


def _route_after_route(state: SelfRagState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate"


def _retrieve(state: SelfRagState) -> dict:
    collection = get_collection(COLLECTION_NAME)
    hits = _get_vector_store(collection).similarity_search_with_score(
        state["question"], k=state["top_k"], pre_filter={"doc_id": state["doc_id"]}
    )
    passages = [{"text": doc.page_content, "page": doc.metadata["page"], "score": float(score)} for doc, score in hits]
    return {"passages": passages, "path": ["retrieve"]}


def _filter_relevant(state: SelfRagState) -> dict:
    passages = state["passages"]
    if not passages:
        return {"relevant_passages": [], "path": ["filter_relevant"]}

    formatted = _format_passages(passages)
    chain = FILTER_PROMPT | get_chat_model(fast=True).with_structured_output(PassageGrades, method="json_schema")
    output = chain.invoke(
        {"question": state["question"], "passages": formatted}, config={"callbacks": get_callbacks()}
    )
    by_index = {g.index: g.grade for g in output.grades}
    relevant = [p for i, p in enumerate(passages, start=1) if by_index.get(i, "irrelevant") == "relevant"]
    tokens = (len(formatted) + len(state["question"])) // 4
    return {"relevant_passages": relevant, "path": ["filter_relevant"], "llm_calls": 1, "tokens": tokens}


def _generate(state: SelfRagState) -> dict:
    context = "\n\n".join(f"[p.{p['page']}] {p['text']}" for p in state["relevant_passages"])
    revision_note = (
        f"\nYour previous answer was rejected: {state['feedback']}\nWrite a better one.\n"
        if state["feedback"]
        else ""
    )
    chain = GENERATE_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"context": context or "(none)", "question": state["question"], "revision_note": revision_note},
        config={"callbacks": get_callbacks()},
    )
    tokens = (len(context) + len(state["question"]) + len(answer)) // 4
    return {"answer": answer, "context": context, "path": ["generate"], "llm_calls": 1, "tokens": tokens, "attempts": 1}


def _critique(state: SelfRagState) -> dict:
    if state["context"]:
        chain = CRITIQUE_PROMPT | get_chat_model(fast=True).with_structured_output(
            CritiqueResult, method="json_schema"
        )
        output = chain.invoke(
            {"question": state["question"], "context": state["context"], "answer": state["answer"]},
            config={"callbacks": get_callbacks()},
        )
        supported, useful, feedback = output.supported, output.useful, output.feedback
    else:
        chain = USEFULNESS_PROMPT | get_chat_model(fast=True).with_structured_output(
            UsefulnessResult, method="json_schema"
        )
        output = chain.invoke(
            {"question": state["question"], "answer": state["answer"]}, config={"callbacks": get_callbacks()}
        )
        supported, useful, feedback = "fully_supported", output.useful, output.feedback

    attempts = state["attempts"]
    good = supported != "not_supported" and useful == "useful"
    budget_left = attempts < state["max_retries"] + 1
    decision = "accept" if good or not budget_left else "retry"

    log_entry = {
        "attempt": attempts,
        "answer": state["answer"],
        "supported": supported,
        "useful": useful,
        "feedback": feedback,
        "decision": decision,
    }
    tokens = (len(state["context"]) + len(state["answer"])) // 4
    return {
        "feedback": feedback,
        "decision": decision,
        "attempt_log": [log_entry],
        "path": ["critique"],
        "llm_calls": 1,
        "tokens": tokens,
    }


def _route_after_critique(state: SelfRagState) -> str:
    return state["decision"]


def _build_graph():
    graph = StateGraph(SelfRagState)
    graph.add_node("route", _route)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("filter_relevant", _filter_relevant)
    graph.add_node("generate", _generate)
    graph.add_node("critique", _critique)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _route_after_route, {"retrieve": "retrieve", "generate": "generate"})
    graph.add_edge("retrieve", "filter_relevant")
    graph.add_edge("filter_relevant", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", _route_after_critique, {"retry": "generate", "accept": END})
    return graph.compile()


GRAPH = _build_graph()


def ask_detailed(
    question: str,
    doc_id: str,
    *,
    top_k: int = 5,
    max_retries: int = 2,
    **_: object,
) -> tuple[RagResult, SelfRagState]:
    """Runs the Self-RAG graph and also returns the full end state for the page."""
    start = time.monotonic()
    state: SelfRagState = GRAPH.invoke(
        {
            "question": question,
            "doc_id": doc_id,
            "top_k": top_k,
            "max_retries": max_retries,
            "needs_retrieval": False,
            "route_reason": "",
            "passages": [],
            "relevant_passages": [],
            "context": "",
            "answer": "",
            "feedback": "",
            "decision": "",
            "attempts": 0,
            "attempt_log": [],
            "path": [],
            "llm_calls": 0,
            "tokens": 0,
        },
        config={
            "callbacks": get_callbacks(),
            "recursion_limit": 6 + (max_retries + 1) * 2,
            "run_name": "self_rag_ask",
        },
    )

    steps = [f"route: needs_retrieval={state['needs_retrieval']} ({state['route_reason']})."]
    if state["needs_retrieval"]:
        steps.append(f"retrieve: found {len(state['passages'])} passage(s) in rag_self_rag.")
        steps.append(f"filter_relevant: kept {len(state['relevant_passages'])} of {len(state['passages'])}.")
    for entry in state["attempt_log"]:
        steps.append(
            f"generate (attempt {entry['attempt']}) -> critique: supported={entry['supported']}, "
            f"useful={entry['useful']}, decision={entry['decision']} ({entry['feedback']})"
        )

    result = RagResult(
        answer=state["answer"],
        contexts=[Passage(text=p["text"], score=p["score"], page=p["page"]) for p in state["relevant_passages"]],
        steps=steps,
        llm_calls=state["llm_calls"],
        tokens=state["tokens"],
        latency_ms=(time.monotonic() - start) * 1000,
    )
    return result, state


def ask(question: str, doc_id: str, *, top_k: int = 5, max_retries: int = 2, **settings: object) -> RagResult:
    result, _state = ask_detailed(question, doc_id, top_k=top_k, max_retries=max_retries, **settings)
    return result
