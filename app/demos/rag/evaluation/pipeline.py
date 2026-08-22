import math
import random
import sys
import time
import types
import uuid
import warnings

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from core.embeddings import LocalEmbeddings
from core.llm import get_chat_model, get_plain_chat_model
from core.mongo import get_collection
from core.pdf import extract_text
from core.tracing import get_callbacks, is_enabled, observe
from demos.rag.contextual import pipeline as contextual_pipeline
from demos.rag.hybrid import pipeline as hybrid_pipeline
from demos.rag.naive import pipeline as naive_pipeline
from demos.rag.rerank import pipeline as rerank_pipeline

# ragas 0.4.3 unconditionally imports langchain_community.chat_models.vertexai, a
# submodule langchain-community 0.4.2 dropped while sunsetting (Vertex AI chat
# models moved out to langchain-google-vertexai). Nothing here uses Vertex AI, so
# the missing symbol is stubbed before ragas imports it, rather than pinning
# either package to a different, unverified version.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness  # noqa: E402

# Only demos that follow the shared ingest()/ask() contract can be compared here.
DEMOS = {
    "Naive RAG": naive_pipeline,
    "Hybrid search": hybrid_pipeline,
    "Re-ranking": rerank_pipeline,
    "Contextual retrieval": contextual_pipeline,
}

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100

QA_PROMPT = ChatPromptTemplate.from_template(
    """Based only on the passage below, write one question a reader of this document might ask, \
and the answer to that question using only information in the passage. Keep the answer short.

Passage:
{chunk}"""
)


class TestCaseDraft(BaseModel):
    question: str
    reference_answer: str


class TestCase(BaseModel):
    question: str
    reference_answer: str


@observe(name="evaluation_generate_test_set")
def generate_test_set(pdf_bytes: bytes, n: int) -> list[TestCase]:
    """Samples random chunks from the PDF and has the chat model write a Q/A pair for each."""
    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    chunks = [c for text in parsed.pages if text.strip() for c in splitter.split_text(text)]
    if not chunks:
        return []

    sampled = random.sample(chunks, k=min(n, len(chunks)))
    # method="json_schema" is required for Groq compatibility: the default
    # ("function_calling") forces a tool call, and openai/gpt-oss-120b on Groq
    # sometimes answers in plain text instead, which LangChain then rejects.
    # json_schema works across all four providers, so it's used unconditionally
    # rather than branching on which one get_chat_model() resolved to.
    structured_model = get_chat_model(fast=True).with_structured_output(TestCaseDraft, method="json_schema")
    chain = QA_PROMPT | structured_model
    drafts = chain.batch(
        [{"chunk": chunk} for chunk in sampled],
        config={"callbacks": get_callbacks()},
        return_exceptions=True,
    )

    test_cases = []
    for draft in drafts:
        if isinstance(draft, Exception):
            continue
        test_cases.append(TestCase(question=draft.question, reference_answer=draft.reference_answer))
    return test_cases


@st.cache_resource(show_spinner=False)
def _get_ragas_llm() -> LangchainLLMWrapper:
    """Ragas mutates fields directly on the LangChain model it's given, which only a plain
    BaseChatModel supports -- the fallback wrapper core.llm.get_chat_model() returns doesn't.
    get_plain_chat_model() hands back the first configured provider unwrapped, so scoring
    runs on one provider with no fallback if that provider is rate-limited."""
    return LangchainLLMWrapper(get_plain_chat_model())


@st.cache_resource(show_spinner=False)
def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(LocalEmbeddings())


def run_evaluation(pdf_bytes: bytes, doc_id: str, demo_name: str, test_cases: list[TestCase]) -> dict:
    """Ingests (if needed) and scores one demo against the test set. Never raises -- failures
    come back as an 'error' key so the page can show a clear message instead of crashing."""
    start = time.monotonic()
    module = DEMOS[demo_name]
    # One session per (demo, run) groups every question's trace together in the Langfuse
    # UI and gives push_scores_to_langfuse() something to attach scores to -- without it,
    # each ask() call is a standalone trace with no link back to this evaluation run.
    session_id = f"eval-{demo_name}-{doc_id[:12]}-{uuid.uuid4().hex[:8]}"
    try:
        collection = get_collection(module.COLLECTION_NAME)
        if collection.count_documents({"doc_id": doc_id}, limit=1) == 0:
            module.ingest(pdf_bytes, doc_id)
    except Exception:
        return {"error": f"Couldn't ingest the document for {demo_name} — MongoDB, the embedding model or (for demos that enrich passages at ingest) the chat model may be unavailable."}

    samples: list[SingleTurnSample] = []
    failed_questions: list[str] = []
    for tc in test_cases:
        try:
            if is_enabled():
                from langfuse import propagate_attributes

                with propagate_attributes(session_id=session_id, tags=["evaluation", demo_name]):
                    result = module.ask(tc.question, doc_id)
            else:
                result = module.ask(tc.question, doc_id)
        except Exception:
            failed_questions.append(tc.question)
            continue
        samples.append(
            SingleTurnSample(
                user_input=tc.question,
                response=result.answer,
                retrieved_contexts=[c.text for c in result.contexts],
                reference=tc.reference_answer,
            )
        )

    if not samples:
        return {"error": f"Every question failed for {demo_name} — the chat model may be rate-limited or unavailable."}

    dataset = EvaluationDataset(samples=samples)
    eval_result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        show_progress=False,
        raise_exceptions=False,
    )
    df = eval_result.to_pandas()
    means: dict[str, float] = {}
    unscored_metrics: list[str] = []
    for metric in METRICS:
        value = float(df[metric].mean(skipna=True)) if metric in df else float("nan")
        means[metric] = value
        if math.isnan(value):
            unscored_metrics.append(metric)

    outcome = {
        "means": means,
        "per_sample": df,
        "failed_questions": failed_questions,
        "latency_ms": (time.monotonic() - start) * 1000,
        "session_id": session_id,
    }
    if len(unscored_metrics) == len(METRICS):
        outcome["warning"] = (
            f"Ragas couldn't score any question for {demo_name} — the evaluator LLM is likely "
            "rate-limited. Scoring runs on the first configured provider with no fallback, and "
            "free-tier daily quotas can be as low as 20 requests/day per model; try again after "
            "it resets, or with fewer demos/questions."
        )
    elif unscored_metrics:
        outcome["warning"] = (
            f"{demo_name}: couldn't score {', '.join(unscored_metrics)} — the evaluator LLM may be "
            "rate-limited. Other metrics above completed."
        )
    return outcome


def push_scores_to_langfuse(results: dict[str, dict]) -> None:
    """Scores land on the session run_evaluation() opened for that demo, so a run's
    questions and its aggregate scores show up together in the Sessions view. Metric
    names stay stable across demos (just "faithfulness", not "eval_naive_faithfulness")
    so a saved dashboard or evaluator keeps matching as demos are added or renamed."""
    if not is_enabled():
        return
    from langfuse import get_client

    client = get_client()
    for outcome in results.values():
        session_id = outcome.get("session_id")
        if not session_id:
            continue
        for metric, value in outcome.get("means", {}).items():
            if math.isnan(value):
                continue
            client.create_score(
                name=metric,
                value=value,
                session_id=session_id,
                data_type="NUMERIC",
            )
