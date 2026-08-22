from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.mongo import get_collection
from core.ui import clear_data_button, metrics_row, pipeline_ribbon, upload_widget
from demos.rag.multimodal.pipeline import COLLECTION_NAME, ask_detailed, clear_images, image_full_path, ingest

DEMO = "multimodal"
STAGES = ["parse", "caption", "embed", "index", "retrieve", "answer"]
PREVIEW_CHARS = 160

st.title("Multimodal RAG", anchor=False)
st.caption("Embedded images get an LLM-written caption, indexed alongside the document's text.")

settings = get_settings()
collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    f"Images under {settings.multimodal_min_image_px}px on either side are skipped, and only the "
    f"first {settings.multimodal_max_images} distinct images are captioned — captioning costs one "
    "LLM call per image."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, settings.multimodal_top_k, key=f"{DEMO}_top_k")


def _run_ingest(pdf_bytes: bytes, doc_id: str) -> None:
    with st.status("Ingesting document...", expanded=True) as status:
        status.write("Extracting text and embedded images with PyMuPDF...")
        progress = st.progress(0.0)

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / total if total else 1.0, text=f"Captioning images: {done} of {total}")

        try:
            stats = ingest(pdf_bytes, doc_id, on_progress=on_progress)
        except Exception:
            status.update(label="Ingest failed", state="error")
            st.error(
                "Couldn't ingest this document — the embedding model or MongoDB may be "
                "unavailable. Check the container logs and try again."
            )
            st.stop()

        if stats.chunks == 0:
            status.update(label="No content found", state="error")
            st.warning("No extractable text or captionable images came out of this PDF. Try a different file.")
            st.stop()

        status.write(f"Found {stats.images_found} distinct embedded image(s), captioned {stats.images_captioned}.")
        status.write("Embedded text chunks and image captions with bge-small-en-v1.5, in one index.")
        status.write(
            f"Indexed {stats.text_chunks} text chunk(s) + {stats.images_captioned} image caption(s) "
            f"in {stats.latency_ms:,.0f} ms, using {stats.llm_calls} caption call(s), ≈{stats.tokens:,} tokens."
        )
        status.update(label="Ingest complete", state="complete")
    st.session_state[f"{DEMO}_ingest_stats"] = stats.model_dump()


if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        _run_ingest(pdf_bytes, uploaded_doc_id)
    st.session_state[f"{DEMO}_doc_id"] = uploaded_doc_id

doc_id = st.session_state.get(f"{DEMO}_doc_id")

if clear_data_button(DEMO, collection, doc_id):
    clear_images(doc_id)
    for key in ("doc_id", "result", "rows", "question", "ingest_stats"):
        st.session_state.pop(f"{DEMO}_{key}", None)
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
rows = st.session_state.get(f"{DEMO}_rows")
ingest_stats: dict | None = st.session_state.get(f"{DEMO}_ingest_stats")
pipeline_ribbon(
    STAGES, active=len(STAGES) if result is not None else (STAGES.index("retrieve") if doc_id is not None else -1)
)

if ingest_stats is not None and ingest_stats["doc_id"] == doc_id:
    st.caption(
        f"`{ingest_stats['images_captioned']}` of `{ingest_stats['images_found']}` found image(s) captioned "
        f"· `{ingest_stats['images_failed']}` failed · `{ingest_stats['images_over_cap']}` over the "
        f"{settings.multimodal_max_images}-image cap"
    )
    if ingest_stats["images_failed"]:
        st.warning(
            f"{ingest_stats['images_failed']} image(s) got no caption — either the chat model was "
            "rate-limited, or no configured provider supports vision (Groq's and the local Ollama "
            "model here are text-only). Those images aren't retrievable; the document's text still is."
        )

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask about the text or a chart/figure",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving across text and captions, then answering..."):
            try:
                result, rows = ask_detailed(question, doc_id, top_k=top_k)
            except Exception:
                result, rows = None, None
                st.error(
                    "Couldn't get an answer — no LLM provider is configured, every configured "
                    "provider is rate-limited, or MongoDB is unavailable. Try again in a moment."
                )
        st.session_state[f"{DEMO}_result"] = result
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.session_state[f"{DEMO}_rows"] = rows

if result is not None and rows is not None:
    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Evidence")
    max_score = max((r["score"] for r in rows), default=1.0) or 1.0
    for r in rows:
        with st.container(border=True):
            bar_len = round(20 * (r["score"] / max_score))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            kind = "image caption" if r["modality"] == "image" else "text"
            st.markdown(f"`{r['score']:.3f}` {bar}  p.{r['page']}  ·  {kind}")
            text = r["text"]
            if len(text) <= PREVIEW_CHARS:
                st.write(text)
            else:
                st.write(text[:PREVIEW_CHARS].rstrip() + "…")
                with st.expander(f"View complete text ({len(text):,} chars)"):
                    st.write(text)
            if r["modality"] == "image":
                path = image_full_path(r["image_path"])
                if path.exists():
                    st.image(str(path), caption=f"p.{r['page']}", width=320)
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
