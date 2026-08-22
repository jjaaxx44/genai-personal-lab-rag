from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.mongo import get_collection
from core.types import RagResult
from core.ui import clear_data_button, metrics_row, pipeline_ribbon, upload_widget
from demos.rag.contextual.pipeline import (
    COLLECTION_NAME,
    VARIANTS,
    ask_detailed,
    clear_cache,
    enrichment_summary,
    ingest,
    sample_enrichment,
)

DEMO = "contextual"
STAGES = ["parse", "chunk", "contextualize", "embed", "index", "retrieve ×2", "answer ×2"]
VARIANT_LABELS = {"plain": "Without context", "contextual": "With context"}
PREVIEW_CHARS = 160

st.title("Contextual retrieval", anchor=False)
st.caption(
    "Before indexing, an LLM writes a sentence or two placing each passage in the whole document, "
    "and that context is embedded and keyword-indexed along with the passage."
)

settings = get_settings()
collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 5, key=f"{DEMO}_top_k")
st.sidebar.caption(
    f"Ingest makes one fast-model call per {settings.contextual_chunks_per_call} passages, "
    f"for up to {settings.contextual_max_chunks} passages. Asking runs 2 LLM calls, one per variant."
)


def _run_ingest(pdf_bytes: bytes, doc_id: str) -> None:
    with st.status("Ingesting document...", expanded=True) as status:
        status.write("Extracting text with PyMuPDF...")
        status.write(
            f"Splitting into passages (~{settings.contextual_chunk_size} chars, "
            f"{settings.contextual_chunk_overlap} overlap)..."
        )
        status.write("Writing a context for each passage with the fast chat model...")
        progress = st.progress(0.0)

        def on_progress(done: int, total: int) -> None:
            if total == 0:
                progress.progress(1.0, text="Every passage's context came from the cache — no LLM calls needed.")
            else:
                progress.progress(done / total, text=f"Enrichment calls: {done} of {total}")

        try:
            stats = ingest(pdf_bytes, doc_id, on_progress=on_progress)
        except RuntimeError as exc:
            status.update(label="Ingest failed", state="error")
            st.error(str(exc))
            st.stop()
        except Exception:
            status.update(label="Ingest failed", state="error")
            st.error(
                "Couldn't ingest this document — the embedding model or MongoDB may be "
                "unavailable. Check the container logs and try again."
            )
            st.stop()

        if stats.chunks == 0:
            status.update(label="No text found", state="error")
            st.warning(
                "No extractable text came out of this PDF — it's likely scanned or "
                "image-only. Try a different file."
            )
            st.stop()
        status.write("Embedded both variants of each passage with bge-small-en-v1.5.")
        status.write("Built the vector and keyword (BM25) indexes.")
        status.write(
            f"Indexed {stats.chunks} passage(s) × 2 variants in {stats.latency_ms:,.0f} ms, "
            f"using {stats.llm_calls} enrichment call(s), ≈{stats.tokens:,} tokens."
        )
        status.update(label="Ingest complete", state="complete")
    st.session_state[f"{DEMO}_ingest_stats"] = stats.model_dump()


if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        _run_ingest(pdf_bytes, uploaded_doc_id)
    st.session_state[f"{DEMO}_doc_id"] = uploaded_doc_id

doc_id = st.session_state.get(f"{DEMO}_doc_id")

rerun_disabled = upload is None or doc_id is None
if st.sidebar.button("Re-run enrichment", key=f"{DEMO}_reingest", disabled=rerun_disabled):
    _run_ingest(upload[0], doc_id)
    st.session_state.pop(f"{DEMO}_results", None)
st.sidebar.caption(
    "Re-ingests the document. Contexts already written are reused from the cache, "
    "so only passages that failed cost LLM calls."
)

if clear_data_button(DEMO, collection, doc_id):
    clear_cache(doc_id)
    for key in ("doc_id", "results", "question", "ingest_stats"):
        st.session_state.pop(f"{DEMO}_{key}", None)
    st.rerun()

results: dict[str, dict | None] = st.session_state.get(f"{DEMO}_results", {})
ingest_stats: dict | None = st.session_state.get(f"{DEMO}_ingest_stats")

timings: dict[str, float] = {}
if ingest_stats is not None and ingest_stats["doc_id"] == doc_id:
    timings.update(ingest_stats["timings"])
ok_runs = [r for r in results.values() if r is not None]
if ok_runs:
    timings["retrieve ×2"] = sum(r["timings"]["retrieve"] for r in ok_runs)
    timings["answer ×2"] = sum(r["timings"]["answer"] for r in ok_runs)
pipeline_ribbon(
    STAGES,
    active=len(STAGES) if results else (STAGES.index("retrieve ×2") if doc_id is not None else -1),
    timings=timings,
)

if doc_id is not None:
    summary = enrichment_summary(doc_id)
    total = sum(summary.values())
    with_context = summary.get("generated", 0) + summary.get("cached", 0)
    st.caption(
        f"`{with_context}` of `{total}` passages carry an added context · "
        f"`{summary.get('failed', 0)}` failed · `{summary.get('over_cap', 0)}` over the "
        f"{settings.contextual_max_chunks}-passage cap"
    )
    if summary.get("failed"):
        st.warning(
            f"{summary['failed']} passage(s) got no context — the chat model was rate-limited or "
            "returned a malformed response — and are stored as plain text in both variants. "
            "Use **Re-run enrichment** to retry just those; the rest come from the cache."
        )
    if ingest_stats is not None and ingest_stats["doc_id"] == doc_id and ingest_stats["doc_truncated"]:
        st.caption(
            f"The document is longer than {settings.contextual_doc_max_chars:,} characters, so each "
            "enrichment call saw only its beginning."
        )
    samples = sample_enrichment(doc_id)
    if samples:
        with st.expander("Inspect the added context"):
            for s in samples:
                with st.container(border=True):
                    st.markdown(f"`passage {s['chunk_index']}`  p.{s['page']}")
                    st.markdown(f"**Added context:** {s['context']}")
                    st.caption(s["original_text"][:PREVIEW_CHARS].rstrip() + "…")

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask something the passages don't state outright",
    )
    submitted = st.form_submit_button("Compare", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        results = {}
        with st.spinner("Retrieving and answering for both variants (2 LLM calls)..."):
            for variant in VARIANTS:
                try:
                    result, rows, run_timings = ask_detailed(question, doc_id, variant=variant, top_k=top_k)
                    results[variant] = {"result": result, "rows": rows, "timings": run_timings}
                except Exception:
                    results[variant] = None
        st.session_state[f"{DEMO}_results"] = results
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        # Rerun so the ribbon above picks up this run's timings.
        st.rerun()

failed = [VARIANT_LABELS[v] for v, run in results.items() if run is None]
if failed:
    st.error(
        f"Couldn't get an answer for: {', '.join(failed)} — no LLM provider is configured, "
        "or every configured provider is rate-limited or unavailable right now."
    )


def _evidence(rows: list[dict], other_indexes: set[int]) -> None:
    """Bordered passages with the fused score and its vector/keyword parts. Passages the
    other variant didn't retrieve are marked, and any added context is shown apart from
    the passage it was prepended to."""
    max_score = max((r["score"] for r in rows), default=1.0) or 1.0
    for r in rows:
        with st.container(border=True):
            bar_len = round(20 * (r["score"] / max_score))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            only_here = "  · only in this variant" if r["chunk_index"] not in other_indexes else ""
            st.markdown(f"`{r['score']:.4f}` {bar}  p.{r['page']}{only_here}")
            st.caption(f"vector `{r['vector_score']:.4f}` + keyword `{r['fulltext_score']:.4f}`")
            if r["context"]:
                st.markdown(f"**Added context:** {r['context']}")
            text = r["text"]
            if len(text) <= PREVIEW_CHARS:
                st.write(text)
            else:
                st.write(text[:PREVIEW_CHARS].rstrip() + "…")
                with st.expander(f"View complete passage ({len(text):,} chars)"):
                    st.write(text)


if results:
    st.subheader("Answer and evidence by variant")
    indexes = {
        variant: {r["chunk_index"] for r in run["rows"]} if run is not None else set()
        for variant, run in results.items()
    }
    cols = st.columns(len(VARIANTS))
    for col, variant in zip(cols, VARIANTS):
        with col:
            st.markdown(f"**{VARIANT_LABELS[variant]}**")
            run = results.get(variant)
            if run is None:
                st.caption("Unavailable — rate-limited or errored.")
                continue
            result: RagResult = run["result"]
            st.write(result.answer)
            st.caption(f"{result.latency_ms:,.0f} ms · {result.tokens:,} tok")
            other = next(v for v in VARIANTS if v != variant)
            _evidence(run["rows"], indexes.get(other, set()))

    if all(results.get(v) is not None for v in VARIANTS):
        st.subheader("Rank without → with context")
        st.caption("Every passage either variant retrieved, by its rank in each. A dash means not retrieved.")
        ranks = {v: {r["chunk_index"]: r["rank"] for r in results[v]["rows"]} for v in VARIANTS}
        pages = {r["chunk_index"]: r["page"] for v in VARIANTS for r in results[v]["rows"]}
        order = sorted(pages, key=lambda i: (ranks["contextual"].get(i, 99), ranks["plain"].get(i, 99)))
        st.dataframe(
            [
                {
                    "Passage": i,
                    "Page": pages[i],
                    # Text, not int: a column mixing ints and "—" fails Arrow serialisation.
                    "Rank without": str(ranks["plain"].get(i, "—")),
                    "Rank with": str(ranks["contextual"].get(i, "—")),
                }
                for i in order
            ],
            hide_index=True,
        )
elif doc_id is None:
    st.info("Upload a PDF in the sidebar to get started.")

how_tab, trace_tab = st.tabs(
    ["How it works", "Trace"], key=f"{DEMO}_tabs", on_change="rerun"
)
with how_tab:
    st.markdown((Path(__file__).parent / "README.md").read_text())
with trace_tab:
    if ingest_stats is not None and ingest_stats["doc_id"] == doc_id:
        st.markdown("**Ingest**")
        st.write(
            f"- Split into {ingest_stats['chunks']} passage(s); "
            f"{ingest_stats['generated']} context(s) generated, {ingest_stats['cached']} from the cache, "
            f"{ingest_stats['failed']} failed, {ingest_stats['over_cap']} over the cap."
        )
        st.write(
            f"- {ingest_stats['llm_calls']} enrichment call(s) via `.batch()`, "
            f"≈{ingest_stats['tokens']:,} tokens — each call carries the whole document."
        )
        st.write("- Each passage stored twice (variant `plain` and `contextual`), embedded and BM25-indexed.")
    if results:
        for variant in VARIANTS:
            run = results.get(variant)
            st.markdown(f"**{VARIANT_LABELS[variant]}**")
            if run is None:
                st.caption("No trace — this variant's call failed.")
                continue
            for step in run["result"].steps:
                st.write(f"- {step}")
    else:
        st.caption("Ask a question to see the retrieval trace.")

if ok_runs:
    total_result = RagResult(
        answer="",
        contexts=[c for r in ok_runs for c in r["result"].contexts],
        steps=[],
        llm_calls=sum(r["result"].llm_calls for r in ok_runs),
        tokens=sum(r["result"].tokens for r in ok_runs),
        latency_ms=sum(r["result"].latency_ms for r in ok_runs),
    )
    metrics_row(total_result)
