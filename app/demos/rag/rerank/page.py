from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.mongo import get_collection
from core.typesafe import is_configured as jev_configured
from core.ui import clear_data_button, evidence_view, flow_view, metrics_row, pipeline_ribbon, readme_view, upload_widget
from demos.rag.rerank.pipeline import COLLECTION_NAME, ask_detailed, ingest
from langchain_typesafe.client import TypeSafeAuthenticationError, TypeSafeError, TypeSafeRateLimitError

DEMO = "rerank"
STAGES = ["parse", "chunk", "embed", "index", "retrieve", "rerank", "answer"]

st.title("Re-ranking", anchor=False)
st.caption("A second-stage judge re-scores a wide candidate pool before the top few go to the LLM.")

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
candidate_k = st.sidebar.slider("Candidates to retrieve", 5, 30, 20, key=f"{DEMO}_candidate_k")
top_k = st.sidebar.slider("Passages to keep after rerank", 1, 10, 5, key=f"{DEMO}_top_k")
top_k = min(top_k, candidate_k)

reranker_label = st.sidebar.radio(
    "Re-ranker",
    ["Cross-encoder (local)", "Jev (TypeSafe API)"],
    key=f"{DEMO}_reranker",
    help="The cross-encoder runs on this machine's CPU. Jev is a hosted model that returns "
    "a calibrated probability that each passage answers the question.",
)
use_jev = reranker_label.startswith("Jev")
min_probability = 0.0
if use_jev:
    min_probability = st.sidebar.slider(
        "Min Jev probability",
        0.0,
        0.9,
        0.0,
        0.05,
        key=f"{DEMO}_min_probability",
        help="Passages Jev rates below this are dropped even if they'd make the top-k. "
        "A calibrated probability can be cut at a threshold; a cross-encoder logit can't.",
    )
jev_missing = use_jev and not jev_configured()
if jev_missing:
    st.sidebar.info("Set TYPESAFE_API_KEY in your .env file (see .env.example) to use Jev.")

if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        settings = get_settings()
        with st.status("Ingesting document...", expanded=True) as status:
            status.write("Extracting text with PyMuPDF...")
            status.write(
                f"Splitting into chunks (~{settings.rerank_chunk_size} chars, "
                f"{settings.rerank_chunk_overlap} overlap)..."
            )
            status.write("Embedding chunks with bge-small-en-v1.5...")
            status.write("Building the vector index...")
            try:
                stats = ingest(pdf_bytes, uploaded_doc_id)
            except Exception:
                status.update(label="Ingest failed", state="error")
                st.error(
                    "Couldn't ingest this document — the embedding model or MongoDB may be "
                    "unavailable. Check the container logs and try again."
                )
                st.stop()
            status.write(f"Indexed {stats.chunks} chunk(s) in {stats.latency_ms:.0f} ms.")
            if stats.chunks == 0:
                status.update(label="No text found", state="error")
                st.warning(
                    "No extractable text came out of this PDF — it's likely scanned or "
                    "image-only. Try a different file."
                )
                st.stop()
            status.update(label="Ingest complete", state="complete")
    st.session_state[f"{DEMO}_doc_id"] = uploaded_doc_id

doc_id = st.session_state.get(f"{DEMO}_doc_id")

if clear_data_button(DEMO, collection, doc_id):
    st.session_state.pop(f"{DEMO}_doc_id", None)
    st.session_state.pop(f"{DEMO}_result", None)
    st.session_state.pop(f"{DEMO}_rows", None)
    st.session_state.pop(f"{DEMO}_info", None)
    st.session_state.pop(f"{DEMO}_question", None)
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
rows = st.session_state.get(f"{DEMO}_rows")
info = st.session_state.get(f"{DEMO}_info")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else -1)

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask about the document",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None or jev_missing)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving, re-ranking and answering..."):
            try:
                result, rows, info = ask_detailed(
                    question,
                    doc_id,
                    candidate_k=candidate_k,
                    top_k=top_k,
                    reranker="jev" if use_jev else "cross_encoder",
                    min_probability=min_probability,
                )
            except TypeSafeAuthenticationError:
                result, rows, info = None, None, None
                st.error("TypeSafe rejected the API key. Check TYPESAFE_API_KEY in your .env file.")
            except TypeSafeError:
                result, rows, info = None, None, None
                st.error(
                    "Jev is unavailable or rate-limited right now. Try again in a moment, "
                    "or switch to the local cross-encoder."
                )
            except Exception:
                result, rows, info = None, None, None
                st.error(
                    "Couldn't get an answer — the chat model (no provider configured, or all of them "
                    "rate-limited), the re-ranker model or MongoDB may be unavailable right now. "
                    "Try again in a moment."
                )
        st.session_state[f"{DEMO}_result"] = result
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.session_state[f"{DEMO}_rows"] = rows
        st.session_state[f"{DEMO}_info"] = info

if result is not None and rows is not None:
    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Ranking before → after")
    is_jev = info is not None and info["reranker"] == "jev"
    caption = "Only the passages kept after rerank are shown; the rest of the candidate pool is discarded."
    if is_jev and info["dropped"]:
        caption += (
            f" {info['dropped']} candidate(s) fell below the {info['min_probability']:.2f} "
            "probability threshold."
        )
    st.caption(caption)
    if rows:
        st.dataframe(
            [
                {
                    "Rank before": r["pre_rank"],
                    "Rank after": r["post_rank"],
                    "Vector score": round(r["pre_score"], 3),
                    "Jev P(relevant)" if is_jev else "Cross-encoder score": round(r["rerank_score"], 3),
                    "Page": r["page"],
                }
                for r in rows
            ],
            hide_index=True,
        )

    st.subheader("Evidence")
    evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])
elif doc_id is None:
    st.info("Upload a PDF in the sidebar to get started.")

how_tab, trace_tab = st.tabs(
    ["How it works", "Trace"], key=f"{DEMO}_tabs", on_change="rerun"
)
with how_tab:
    flow_view(Path(__file__).parent)
    readme_view(Path(__file__).parent / "README.md")
with trace_tab:
    if result is not None:
        for step in result.steps:
            st.write(f"- {step}")
    else:
        st.caption("Ask a question to see the trace.")

if result is not None:
    metrics_row(result)
