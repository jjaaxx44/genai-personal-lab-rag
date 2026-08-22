from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.mongo import get_collection
from core.types import RagResult
from core.ui import clear_data_button, evidence_view, metrics_row, pipeline_ribbon, upload_widget
from demos.rag.chunking.pipeline import COLLECTION_NAME, STRATEGIES, ask, ingest, strategy_stats

DEMO = "chunking"
STAGES = ["parse", "split ×4", "embed", "index", "answer ×4"]
STRATEGY_LABELS = {
    "fixed": "Fixed-size",
    "recursive": "Recursive",
    "sliding": "Sliding window",
    "semantic": "Semantic",
}

st.title("Chunking strategies", anchor=False)
st.caption("Four splitters ingest the same PDF — compare what each one retrieves and answers for one question.")

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 5, key=f"{DEMO}_top_k")
st.sidebar.caption("Asking a question runs 4 separate LLM calls, one per strategy.")

if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        settings = get_settings()
        with st.status("Ingesting document...", expanded=True) as status:
            status.write("Extracting text with PyMuPDF...")
            status.write(
                f"Splitting into 4 strategies (fixed, recursive, sliding, semantic) at "
                f"~{settings.chunking_chunk_size} chars each..."
            )
            status.write("Embedding sentences for semantic breakpoints...")
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
            status.write(f"Indexed {stats.chunks} chunk(s) across 4 strategies in {stats.latency_ms:.0f} ms.")
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

if doc_id is not None:
    st.subheader("Chunk stats")
    stats = strategy_stats(doc_id)
    cols = st.columns(len(STRATEGIES))
    for col, strategy in zip(cols, STRATEGIES):
        with col:
            st.metric(STRATEGY_LABELS[strategy], stats[strategy]["count"])
            st.caption(f"avg {stats[strategy]['avg_size']:.0f} chars")

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask about the document",
    )
    submitted = st.form_submit_button("Compare", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        results = {}
        failed: list[str] = []
        with st.spinner("Answering from all 4 strategies (4 LLM calls)..."):
            for strategy in STRATEGIES:
                try:
                    results[strategy] = ask(question, doc_id, strategy=strategy, top_k=top_k)
                except Exception:
                    results[strategy] = None
                    failed.append(STRATEGY_LABELS[strategy])
        st.session_state[f"{DEMO}_results"] = results
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        if failed:
            st.error(
                f"Couldn't get an answer for: {', '.join(failed)} — no LLM provider is configured, "
                "or every configured provider is rate-limited or unavailable right now. "
                "Other strategies still completed below."
            )

if results:
    st.subheader("Answer and evidence by strategy")
    cols = st.columns(len(STRATEGIES))
    for col, strategy in zip(cols, STRATEGIES):
        with col:
            st.markdown(f"**{STRATEGY_LABELS[strategy]}**")
            result = results.get(strategy)
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
    st.markdown((Path(__file__).parent / "README.md").read_text())
with trace_tab:
    if results:
        for strategy in STRATEGIES:
            result = results.get(strategy)
            st.markdown(f"**{STRATEGY_LABELS[strategy]}**")
            if result is None:
                st.caption("No trace — this strategy's call failed.")
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
