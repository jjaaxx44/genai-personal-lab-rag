from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.graph_db import GraphUnavailable, clear_graph_doc, get_graph
from core.ui import evidence_view, graph_triples_dot, metrics_row, pipeline_ribbon, readme_view, upload_widget
from demos.rag.graph_rag.pipeline import DEMO_TAG, ask, graph_summary, ingest

DEMO = "graph_rag"
STAGES = ["parse", "chunk", "extract graph", "embed", "vector search", "expand", "answer"]

st.title("GraphRAG", anchor=False)
st.caption(
    "An LLM extracts entities and relationships into a Neo4j graph; retrieval does a vector "
    "search over chunks, then expands one hop into the graph around what they mention."
)

settings = get_settings()

try:
    graph = get_graph()
    graph_ok = True
except GraphUnavailable as exc:
    graph = None
    graph_ok = False
    st.error(str(exc))

upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Chunks to retrieve", 1, 10, settings.graph_rag_top_k, key=f"{DEMO}_top_k")
st.sidebar.caption(
    f"Ingest extracts entities from up to {settings.graph_rag_max_chunks} chunks "
    f"(one LLM call each, {settings.graph_rag_extract_concurrency} at a time) — building the graph "
    "is the slow, costly step, so it's capped."
)


def _run_ingest(pdf_bytes: bytes, doc_id: str) -> None:
    with st.status("Building the graph...", expanded=True) as status:
        status.write("Extracting text with PyMuPDF and splitting into chunks...")
        status.write(f"Extracting entities and relationships (fast model, up to {settings.graph_rag_max_chunks} chunks)...")
        try:
            stats = ingest(pdf_bytes, doc_id)
        except GraphUnavailable as exc:
            status.update(label="Ingest failed", state="error")
            st.error(str(exc))
            st.stop()
        except RuntimeError as exc:
            status.update(label="Ingest failed", state="error")
            st.error(str(exc))
            st.stop()
        except Exception:
            status.update(label="Ingest failed", state="error")
            st.error(
                "Couldn't build the graph — Neo4j may be unreachable, or APOC may not be "
                "enabled on this database. Check the container logs and try again."
            )
            st.stop()

        if stats.chunks == 0:
            status.update(label="No text found", state="error")
            st.warning(
                "No extractable text came out of this PDF — it's likely scanned or "
                "image-only. Try a different file."
            )
            st.stop()
        status.write(f"Extracted {stats.nodes} node(s) and {stats.relationships} relationship(s) into Neo4j.")
        status.write("Embedded every chunk (bge-small-en-v1.5) into the Neo4j vector index.")
        if stats.failed:
            status.write(f"{stats.failed} chunk(s) failed extraction — rate-limited or malformed output.")
        if stats.over_cap:
            status.write(f"{stats.over_cap} chunk(s) were over the cap and were embedded but not graphed.")
        status.update(label="Graph built", state="complete")
    st.session_state[f"{DEMO}_ingest_stats"] = stats.model_dump()


if graph_ok and upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if st.session_state.get(f"{DEMO}_doc_id") != uploaded_doc_id:
        _run_ingest(pdf_bytes, uploaded_doc_id)
        st.session_state.pop(f"{DEMO}_result", None)
    st.session_state[f"{DEMO}_doc_id"] = uploaded_doc_id

doc_id = st.session_state.get(f"{DEMO}_doc_id")

flash_key = f"{DEMO}_cleared"
if flash_key in st.session_state:
    st.sidebar.success(st.session_state.pop(flash_key))
if st.sidebar.button("Clear my data", key=f"{DEMO}_clear", disabled=doc_id is None or not graph_ok):
    deleted = clear_graph_doc(graph, DEMO_TAG, doc_id)
    st.session_state[flash_key] = f"Cleared {deleted} node(s)." if deleted else "Nothing to clear."
    st.session_state.pop(f"{DEMO}_doc_id", None)
    st.session_state.pop(f"{DEMO}_result", None)
    st.session_state.pop(f"{DEMO}_ingest_stats", None)
    st.session_state.pop(f"{DEMO}_question", None)
    st.session_state[f"{DEMO}_upload_gen"] = st.session_state.get(f"{DEMO}_upload_gen", 0) + 1
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else (2 if doc_id is not None else -1))

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask something that relates two entities",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching and expanding the graph..."):
            try:
                result = ask(question, doc_id, top_k=top_k)
            except GraphUnavailable as exc:
                result = None
                st.error(str(exc))
            except Exception:
                result = None
                st.error(
                    "Couldn't get an answer — no LLM provider is configured, Neo4j is "
                    "unreachable, or every configured provider is rate-limited right now."
                )
        if result is not None:
            st.session_state[f"{DEMO}_result"] = result
            # A run switches the reader to the trace; the tab is still clickable back.
            st.session_state[f"{DEMO}_tabs"] = "Trace"

if result is not None:
    st.subheader("Answer")
    st.write(result.answer)

    left, right = st.columns(2)
    with left:
        st.subheader("Retrieved chunks")
        evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])
    with right:
        st.subheader("Subgraph used for the answer")
        if result.triples:
            mentioned = {t["source"] for t in result.triples}
            st.graphviz_chart(graph_triples_dot(result.triples, highlight=mentioned))
        else:
            st.caption("No graph relationships were found for the retrieved chunks.")
elif doc_id is None:
    st.info("Upload a PDF in the sidebar to get started." if graph_ok else "Neo4j is unavailable — see the message above.")

if doc_id is not None and graph_ok:
    with st.expander("Extracted graph (whole document)"):
        rows = graph_summary(doc_id)
        if rows:
            st.graphviz_chart(graph_triples_dot(rows))
        else:
            st.caption("No relationships were extracted for this document.")

how_tab, trace_tab = st.tabs(
    ["How it works", "Trace"], key=f"{DEMO}_tabs", on_change="rerun"
)
with how_tab:
    readme_view(Path(__file__).parent / "README.md")
with trace_tab:
    ingest_stats = st.session_state.get(f"{DEMO}_ingest_stats")
    if ingest_stats is not None and ingest_stats["doc_id"] == doc_id:
        st.markdown("**Ingest**")
        st.write(
            f"- Split into {ingest_stats['chunks']} chunk(s); extracted graph elements from "
            f"{ingest_stats['extracted']} of them, {ingest_stats['failed']} failed, "
            f"{ingest_stats['over_cap']} over the cap."
        )
        st.write(f"- {ingest_stats['nodes']} node(s), {ingest_stats['relationships']} relationship(s) added to Neo4j.")
    if result is not None:
        st.markdown("**Ask**")
        for step in result.steps:
            st.write(f"- {step}")
    if ingest_stats is None and result is None:
        st.caption("Upload a PDF and ask a question to see the trace.")

if result is not None:
    metrics_row(result)
