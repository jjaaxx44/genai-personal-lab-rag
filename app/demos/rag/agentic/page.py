from pathlib import Path

import streamlit as st

from core.mongo import get_collection
from core.ui import clear_data_button, evidence_view, metrics_row, pipeline_ribbon, readme_view, upload_widget
from demos.rag.agentic.pipeline import COLLECTION_NAME, ask_detailed, ingest

DEMO = "agentic"
STAGES = ["parse", "chunk", "embed", "index", "agent loop", "answer"]

st.title("Agentic RAG", anchor=False)
st.caption(
    "The model decides for itself whether to search the document, how many times, and when it "
    "needs a full page instead of a chunk — rather than a fixed retrieve-then-answer pipeline."
)

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages per search_document call", 1, 10, 5, key=f"{DEMO}_top_k")
st.sidebar.caption(
    "The agent can call search_document more than once with different queries, and get_page "
    "to read a whole page — the trace below shows exactly which tools it used and why."
)

if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        with st.status("Ingesting document...", expanded=True) as status:
            status.write("Extracting text with PyMuPDF...")
            status.write("Splitting into chunks and embedding with bge-small-en-v1.5...")
            status.write("Storing whole-page text for get_page, alongside the embedded chunks...")
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
    for key in ("doc_id", "result", "trace", "question"):
        st.session_state.pop(f"{DEMO}_{key}", None)
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
trace = st.session_state.get(f"{DEMO}_trace")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else -1)

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask about the document, or just say hello",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Running the agent loop..."):
            try:
                result, trace = ask_detailed(question, doc_id, top_k=top_k)
            except Exception:
                result, trace = None, None
                st.error(
                    "Couldn't get an answer — the chat model (no provider configured, or all of them "
                    "rate-limited) or MongoDB may be unavailable right now. Try again in a moment."
                )
        st.session_state[f"{DEMO}_result"] = result
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.session_state[f"{DEMO}_trace"] = trace

if result is not None and trace is not None:
    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Agent trace")
    if trace:
        st.caption(f"{len(trace)} tool call(s).")
        for i, entry in enumerate(trace, start=1):
            with st.container(border=True):
                st.markdown(f"`{i}` **{entry['tool']}**`({entry['input']!r})`")
                st.caption(entry["summary"])
    else:
        st.info("No tool calls — the model judged this didn't need the document and answered directly.")

    if result.contexts:
        st.subheader("Evidence")
        st.caption("Every passage any search_document call surfaced, across the whole trace, ranked by score.")
        evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])
elif doc_id is None:
    st.info("Upload a PDF in the sidebar to get started.")

how_tab, trace_tab = st.tabs(
    ["How it works", "Trace"], key=f"{DEMO}_tabs", on_change="rerun"
)
with how_tab:
    readme_view(Path(__file__).parent / "README.md")
with trace_tab:
    if result is not None:
        for step in result.steps:
            st.write(f"- {step}")
    else:
        st.caption("Ask a question to see the trace.")

if result is not None:
    metrics_row(result)
