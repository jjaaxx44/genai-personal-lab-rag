from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.ui import evidence_view, metrics_row, pipeline_ribbon, upload_widget
from demos.rag.vectorless.pipeline import ask, clear_doc, has_tree, ingest

DEMO = "vectorless"
STAGES = ["parse", "build tree", "navigate", "read", "answer"]

st.title("Vectorless RAG", anchor=False)
st.caption("No embeddings, no MongoDB — the model navigates the document's own table of contents.")

upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
settings = get_settings()
max_depth = st.sidebar.slider(
    "Max navigation steps", 1, 10, settings.vectorless_max_depth, key=f"{DEMO}_max_depth"
)

if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if not has_tree(uploaded_doc_id):
        with st.status("Building document tree...", expanded=True) as status:
            status.write("Extracting text with PyMuPDF...")
            status.write("Reading the table of contents...")
            try:
                stats = ingest(pdf_bytes, uploaded_doc_id)
            except Exception:
                status.update(label="Ingest failed", state="error")
                st.error(
                    "Couldn't build a tree for this document — no LLM provider may be "
                    "configured for the page-group fallback. Check the container logs and try again."
                )
                st.stop()
            status.write(f"Built a tree of {stats.chunks} section(s) in {stats.latency_ms:.0f} ms.")
            status.update(label="Tree built", state="complete")
    if st.session_state.get(f"{DEMO}_doc_id") != uploaded_doc_id:
        st.session_state.pop(f"{DEMO}_result", None)
    st.session_state[f"{DEMO}_doc_id"] = uploaded_doc_id

doc_id = st.session_state.get(f"{DEMO}_doc_id")

flash_key = f"{DEMO}_cleared"
if flash_key in st.session_state:
    st.sidebar.success(st.session_state.pop(flash_key))
if st.sidebar.button("Clear my data", key=f"{DEMO}_clear", disabled=doc_id is None):
    deleted = clear_doc(doc_id)
    st.session_state[flash_key] = "Cleared the saved tree." if deleted else "Nothing to clear."
    st.session_state.pop(f"{DEMO}_doc_id", None)
    st.session_state.pop(f"{DEMO}_result", None)
    st.session_state.pop(f"{DEMO}_question", None)
    st.session_state[f"{DEMO}_upload_gen"] = st.session_state.get(f"{DEMO}_upload_gen", 0) + 1
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else -1)

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask about the document",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Navigating the tree and answering..."):
            try:
                result = ask(question, doc_id, max_depth=max_depth)
            except Exception:
                result = None
                st.error(
                    "Couldn't get an answer — no LLM provider is configured, or every configured "
                    "provider is rate-limited or unavailable right now. Try again in a moment."
                )
        if result is not None:
            st.session_state[f"{DEMO}_result"] = result
            # A run switches the reader to the trace; the tab is still clickable back.
            st.session_state[f"{DEMO}_tabs"] = "Trace"

if result is not None:
    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Path through the tree")
    for step in result.steps:
        st.write(f"- {step}")

    st.subheader("Pages read")
    evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])
elif doc_id is None:
    st.info("Upload a PDF in the sidebar to get started.")

how_tab, trace_tab = st.tabs(
    ["How it works", "Trace"], key=f"{DEMO}_tabs", on_change="rerun"
)
with how_tab:
    st.markdown((Path(__file__).parent / "README.md").read_text())
with trace_tab:
    if result is not None:
        for step in result.steps:
            st.write(f"- {step}")
    else:
        st.caption("Ask a question to see the trace.")

if result is not None:
    metrics_row(result)
