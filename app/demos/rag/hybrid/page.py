from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.mongo import get_collection
from core.types import RagResult
from core.ui import clear_data_button, evidence_view, metrics_row, pipeline_ribbon, readme_view, upload_widget
from demos.rag.hybrid.pipeline import COLLECTION_NAME, MODES, ask, ingest

DEMO = "hybrid"
STAGES = ["parse", "chunk", "embed", "index", "fuse", "answer ×3"]
MODE_LABELS = {"vector": "Vector only", "keyword": "Keyword only", "fused": "Fused (RRF)"}

st.title("Hybrid search", anchor=False)
st.caption("Vector and keyword search run separately, then merged with reciprocal rank fusion.")

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 5, key=f"{DEMO}_top_k")
vector_weight = st.sidebar.slider(
    "Vector weight (fused mode)", 0.0, 1.0, 0.5, step=0.1, key=f"{DEMO}_vector_weight"
)
st.sidebar.caption("Asking a question runs 3 separate LLM calls, one per mode.")

if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        settings = get_settings()
        with st.status("Ingesting document...", expanded=True) as status:
            status.write("Extracting text with PyMuPDF...")
            status.write(
                f"Splitting into chunks (~{settings.hybrid_chunk_size} chars, "
                f"{settings.hybrid_chunk_overlap} overlap)..."
            )
            status.write("Embedding chunks with bge-small-en-v1.5...")
            status.write("Building the vector and keyword (BM25) indexes...")
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
    st.session_state.pop(f"{DEMO}_results", None)
    st.session_state.pop(f"{DEMO}_question", None)
    st.rerun()

results: dict[str, RagResult | None] = st.session_state.get(f"{DEMO}_results", {})

pipeline_ribbon(STAGES, active=len(STAGES) if results else (len(STAGES) - 1 if doc_id is not None else -1))

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Try an exact code or name from the document",
    )
    submitted = st.form_submit_button("Compare", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        results = {}
        failed: list[str] = []
        with st.spinner("Answering from all 3 modes (3 LLM calls)..."):
            for mode in MODES:
                try:
                    results[mode] = ask(
                        question, doc_id, mode=mode, top_k=top_k, vector_weight=vector_weight
                    )
                except Exception:
                    results[mode] = None
                    failed.append(MODE_LABELS[mode])
        st.session_state[f"{DEMO}_results"] = results
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        if failed:
            st.error(
                f"Couldn't get an answer for: {', '.join(failed)} — no LLM provider is configured, "
                "or every configured provider is rate-limited or unavailable right now. "
                "Other modes still completed below."
            )

if results:
    st.subheader("Answer and evidence by mode")
    cols = st.columns(len(MODES))
    for col, mode in zip(cols, MODES):
        with col:
            st.markdown(f"**{MODE_LABELS[mode]}**")
            result = results.get(mode)
            if result is None:
                st.caption("Unavailable — rate-limited or errored.")
                continue
            st.write(result.answer)
            st.caption(f"{result.latency_ms:.0f} ms · {result.tokens:,} tok")
            evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])
elif doc_id is None:
    st.info("Upload a PDF in the sidebar to get started.")

how_tab, trace_tab = st.tabs(
    ["How it works", "Trace"], key=f"{DEMO}_tabs", on_change="rerun"
)
with how_tab:
    readme_view(Path(__file__).parent / "README.md")
with trace_tab:
    if results:
        for mode in MODES:
            result = results.get(mode)
            st.markdown(f"**{MODE_LABELS[mode]}**")
            if result is None:
                st.caption("No trace — this mode's call failed.")
                continue
            for step in result.steps:
                st.write(f"- {step}")
    else:
        st.caption("Ask a question to see the trace.")

if results:
    ok_results = [r for r in results.values() if r is not None]
    if ok_results:
        total = RagResult(
            answer="",
            contexts=[c for r in ok_results for c in r.contexts],
            steps=[],
            llm_calls=sum(r.llm_calls for r in ok_results),
            tokens=sum(r.tokens for r in ok_results),
            latency_ms=sum(r.latency_ms for r in ok_results),
        )
        metrics_row(total)
