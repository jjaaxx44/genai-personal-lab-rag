from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.mongo import get_collection
from core.ui import clear_data_button, evidence_view, metrics_row, pipeline_ribbon, upload_widget
from demos.rag.naive.pipeline import COLLECTION_NAME, ask, ingest

DEMO = "naive"
STAGES = ["parse", "chunk", "embed", "retrieve", "answer"]

st.title("Naive RAG", anchor=False)
st.caption("Chunk, embed, retrieve, answer — the baseline every other RAG technique improves on.")

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 5, key=f"{DEMO}_top_k")

if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        settings = get_settings()
        with st.status("Ingesting document...", expanded=True) as status:
            status.write("Extracting text with PyMuPDF...")
            status.write(
                f"Splitting into chunks (~{settings.naive_chunk_size} chars, "
                f"{settings.naive_chunk_overlap} overlap)..."
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
                    "image-only. Naive RAG reads a PDF's text layer only, so there's "
                    "nothing to index. Try a different file."
                )
                st.stop()
            status.update(label="Ingest complete", state="complete")
    st.session_state[f"{DEMO}_doc_id"] = uploaded_doc_id

doc_id = st.session_state.get(f"{DEMO}_doc_id")

if clear_data_button(DEMO, collection, doc_id):
    st.session_state.pop(f"{DEMO}_doc_id", None)
    st.session_state.pop(f"{DEMO}_result", None)
    st.session_state.pop(f"{DEMO}_question", None)
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
        with st.spinner("Retrieving and answering..."):
            try:
                result = ask(question, doc_id, top_k=top_k)
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

    st.subheader("Evidence")
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
