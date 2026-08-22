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

COLLECTION_NAME = "rag_crag"

GRADE_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

Passages retrieved for this question:

{passages}

For each passage, grade whether it is "relevant" (clearly helps answer the question), \
"irrelevant" (off-topic), or "ambiguous" (on-topic but doesn't clearly answer the \
question by itself). Return one grade per passage, using its index."""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """The document didn't have a clear answer to this question:

{question}

Rewrite it as a short, keyword-focused web search query likely to find an answer \
online. Return only the query, nothing else."""
)

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the numbered context below. Context from the \
uploaded document is marked [doc, p.N]; context from a web search is marked [web]. \
Cite the source marker(s) you relied on. If the context doesn't answer the question, say so.

{context}

Question: {question}
Answer:"""
)


class PassageGrade(BaseModel):
    index: int = Field(description="The index attribute of the passage this grade is for.")
    grade: Literal["relevant", "irrelevant", "ambiguous"]


class PassageGrades(BaseModel):
    grades: list[PassageGrade]


class CragState(TypedDict):
    question: str
    doc_id: str
    top_k: int
    web_results_n: int
    passages: list[dict]
    grades: list[dict]
    decision: str
    rewritten_query: str
    web_results: list[dict]
    context: str
    used_passages: list[dict]
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


def _format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f'<passage index="{i}">\n{p["text"]}\n</passage>' for i, p in enumerate(passages, start=1))


def ingest(pdf_bytes: bytes, doc_id: str) -> IngestStats:
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    clear_doc(collection, doc_id)

    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.crag_chunk_size, chunk_overlap=settings.crag_chunk_overlap
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


def _retrieve(state: CragState) -> dict:
    collection = get_collection(COLLECTION_NAME)
    hits = _get_vector_store(collection).similarity_search_with_score(
        state["question"], k=state["top_k"], pre_filter={"doc_id": state["doc_id"]}
    )
    passages = [{"text": doc.page_content, "page": doc.metadata["page"], "score": float(score)} for doc, score in hits]
    return {"passages": passages, "path": ["retrieve"]}


def _grade(state: CragState) -> dict:
    passages = state["passages"]
    if not passages:
        return {"grades": [], "decision": "insufficient", "path": ["grade"]}

    formatted = _format_passages(passages)
    chain = GRADE_PROMPT | get_chat_model(fast=True).with_structured_output(PassageGrades, method="json_schema")
    output = chain.invoke(
        {"question": state["question"], "passages": formatted}, config={"callbacks": get_callbacks()}
    )
    by_index = {g.index: g.grade for g in output.grades}
    grades = [{**p, "grade": by_index.get(i, "irrelevant")} for i, p in enumerate(passages, start=1)]
    decision = "sufficient" if any(g["grade"] == "relevant" for g in grades) else "insufficient"
    tokens = (len(formatted) + len(state["question"])) // 4
    return {"grades": grades, "decision": decision, "path": ["grade"], "llm_calls": 1, "tokens": tokens}


def _route_after_grade(state: CragState) -> str:
    return state["decision"]


def _rewrite_query(state: CragState) -> dict:
    chain = REWRITE_PROMPT | get_chat_model(fast=True) | StrOutputParser()
    rewritten = chain.invoke({"question": state["question"]}, config={"callbacks": get_callbacks()}).strip()
    tokens = (len(state["question"]) + len(rewritten)) // 4
    return {
        "rewritten_query": rewritten or state["question"],
        "path": ["rewrite_query"],
        "llm_calls": 1,
        "tokens": tokens,
    }


def _web_search(state: CragState) -> dict:
    from ddgs import DDGS

    query = state["rewritten_query"] or state["question"]
    try:
        hits = DDGS().text(query, max_results=state["web_results_n"])
    except Exception:
        hits = []
    web_results = [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")} for h in hits]
    return {"web_results": web_results, "path": ["web_search"]}


def _generate(state: CragState) -> dict:
    if state["decision"] == "sufficient":
        used = [g for g in state["grades"] if g["grade"] == "relevant"]
    else:
        used = [g for g in state["grades"] if g["grade"] == "ambiguous"]

    context_parts = [f"[doc, p.{p['page']}] {p['text']}" for p in used]
    if state["decision"] == "insufficient":
        context_parts += [f"[web] {w['title']}: {w['snippet']}" for w in state["web_results"]]
    context = "\n\n".join(context_parts) or "No relevant information was found."

    chain = GENERATE_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"context": context, "question": state["question"]}, config={"callbacks": get_callbacks()}
    )
    tokens = (len(context) + len(state["question"]) + len(answer)) // 4
    return {"answer": answer, "context": context, "used_passages": used, "path": ["generate"], "llm_calls": 1, "tokens": tokens}


def _build_graph():
    graph = StateGraph(CragState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade", _grade)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("web_search", _web_search)
    graph.add_node("generate", _generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", _route_after_grade, {"sufficient": "generate", "insufficient": "rewrite_query"}
    )
    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


GRAPH = _build_graph()


def ask_detailed(
    question: str,
    doc_id: str,
    *,
    top_k: int = 5,
    web_results_n: int = 3,
    **_: object,
) -> tuple[RagResult, CragState]:
    """Runs the CRAG graph and also returns the full end state for the page."""
    start = time.monotonic()
    state: CragState = GRAPH.invoke(
        {
            "question": question,
            "doc_id": doc_id,
            "top_k": top_k,
            "web_results_n": web_results_n,
            "passages": [],
            "grades": [],
            "decision": "",
            "rewritten_query": "",
            "web_results": [],
            "context": "",
            "used_passages": [],
            "answer": "",
            "path": [],
            "llm_calls": 0,
            "tokens": 0,
        },
        config={"callbacks": get_callbacks(), "run_name": "crag_ask"},
    )

    steps = [f"retrieve: found {len(state['passages'])} passage(s) in rag_crag."]
    if state["grades"]:
        summary = ", ".join(f"p.{g['page']}={g['grade']}" for g in state["grades"])
        steps.append(f"grade: {summary}.")
    else:
        steps.append("grade: no passages to grade.")
    if state["decision"] == "sufficient":
        steps.append("decision: at least one relevant passage -- answering from the document.")
    else:
        steps.append("decision: no relevant passage -- falling back to the web.")
        steps.append(f"rewrite_query: {question!r} -> {state['rewritten_query']!r}.")
        steps.append(f"web_search: {len(state['web_results'])} result(s).")
    steps.append(
        f"generate: answered from {len(state['used_passages'])} document passage(s)"
        + (f" and {len(state['web_results'])} web result(s)." if state["decision"] == "insufficient" else ".")
    )

    result = RagResult(
        answer=state["answer"],
        contexts=[Passage(text=p["text"], score=p["score"], page=p["page"]) for p in state["used_passages"]],
        steps=steps,
        llm_calls=state["llm_calls"],
        tokens=state["tokens"],
        latency_ms=(time.monotonic() - start) * 1000,
    )
    return result, state


def ask(question: str, doc_id: str, *, top_k: int = 5, web_results_n: int = 3, **settings: object) -> RagResult:
    result, _state = ask_detailed(question, doc_id, top_k=top_k, web_results_n=web_results_n, **settings)
    return result
