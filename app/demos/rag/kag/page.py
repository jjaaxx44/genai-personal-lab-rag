from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import get_settings
from core.graph_db import GraphUnavailable, clear_graph_doc, get_graph
from core.ui import graph_triples_dot, pipeline_ribbon, readme_view, upload_widget
from demos.rag.kag.pipeline import DEMO_TAG, ask, default_schema_json, graph_summary, ingest, parse_schema

DEMO = "kag"
STAGES = ["parse", "chunk", "extract (schema-constrained)", "plan steps", "run Cypher", "answer"]

st.title("KAG", anchor=False)
st.caption(
    "Extraction is restricted to a schema you define. The model breaks the question into steps, "
    "each run as its own read-only Cypher query, then answers from the results."
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

st.sidebar.subheader("Schema")
schema_text = st.sidebar.text_area(
    "Entity and relationship types (JSON)",
    value=st.session_state.get(f"{DEMO}_schema_text", default_schema_json()),
    height=180,
    key=f"{DEMO}_schema_text",
)
schema = None
schema_error = None
try:
    schema = parse_schema(schema_text)
except Exception as exc:
    schema_error = str(exc)
    st.sidebar.error(f"Invalid schema: {schema_error}")

st.sidebar.subheader("Settings")
max_steps = st.sidebar.slider("Max steps", 1, 6, settings.kag_max_steps, key=f"{DEMO}_max_steps")
st.sidebar.caption(
    f"Extraction runs on up to {settings.kag_max_chunks} chunks (one LLM call each). "
    "Entities and relationships outside the schema above are ignored."
)


def _run_ingest(pdf_bytes: bytes, doc_id: str) -> None:
    with st.status("Extracting the schema-constrained graph...", expanded=True) as status:
        status.write("Extracting text with PyMuPDF and splitting into chunks...")
        status.write(f"Extracting only the schema's entity/relationship types (up to {settings.kag_max_chunks} chunks)...")
        try:
            stats = ingest(pdf_bytes, doc_id, schema=schema)
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
        if stats.failed:
            status.write(f"{stats.failed} chunk(s) failed extraction — rate-limited or malformed output.")
        if stats.over_cap:
            status.write(f"{stats.over_cap} chunk(s) were over the cap and were skipped.")
        status.update(label="Graph built", state="complete")
    st.session_state[f"{DEMO}_ingest_stats"] = stats.model_dump()
    st.session_state[f"{DEMO}_ingest_schema"] = schema_text


if graph_ok and schema is not None and upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if st.session_state.get(f"{DEMO}_doc_id") != uploaded_doc_id:
        _run_ingest(pdf_bytes, uploaded_doc_id)
        st.session_state.pop(f"{DEMO}_result", None)
    st.session_state[f"{DEMO}_doc_id"] = uploaded_doc_id
    st.session_state[f"{DEMO}_pdf_bytes"] = pdf_bytes

doc_id = st.session_state.get(f"{DEMO}_doc_id")

rerun_disabled = doc_id is None or schema is None or not graph_ok
if st.sidebar.button("Re-extract with this schema", key=f"{DEMO}_reingest", disabled=rerun_disabled):
    _run_ingest(st.session_state[f"{DEMO}_pdf_bytes"], doc_id)
    st.session_state.pop(f"{DEMO}_result", None)

flash_key = f"{DEMO}_cleared"
if flash_key in st.session_state:
    st.sidebar.success(st.session_state.pop(flash_key))
if st.sidebar.button("Clear my data", key=f"{DEMO}_clear", disabled=doc_id is None or not graph_ok):
    deleted = clear_graph_doc(graph, DEMO_TAG, doc_id)
    st.session_state[flash_key] = f"Cleared {deleted} node(s)." if deleted else "Nothing to clear."
    for key in ("doc_id", "pdf_bytes", "result", "ingest_stats", "ingest_schema", "question"):
        st.session_state.pop(f"{DEMO}_{key}", None)
    st.session_state[f"{DEMO}_upload_gen"] = st.session_state.get(f"{DEMO}_upload_gen", 0) + 1
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else (2 if doc_id is not None else -1))

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask something involving two or more entities",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    elif schema is None:
        st.warning("Fix the schema JSON first.")
    else:
        with st.spinner("Planning steps and running Cypher..."):
            try:
                result = ask(question, doc_id, schema=schema, max_steps=max_steps)
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

    st.subheader("Steps")
    for i, step in enumerate(result.kag_steps, start=1):
        with st.container(border=True):
            st.markdown(f"**Step {i}: {step.description}**")
            if step.cypher:
                st.code(step.cypher, language="cypher")
            if step.blocked_reason:
                st.warning(f"Blocked: {step.blocked_reason}")
            else:
                if step.retried:
                    st.caption("First attempt was rejected — this is the query after a fix-and-retry.")
                st.dataframe(pd.DataFrame(step.rows), width="stretch")
                if step.truncated:
                    st.caption(f"Showing the first {settings.kag_row_limit} row(s) — the query returned more.")
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
    cols = st.columns(4)
    cols[0].metric("Latency", f"{result.latency_ms:.0f} ms")
    cols[1].metric("Tokens", f"{result.tokens:,}")
    cols[2].metric("LLM calls", result.llm_calls)
    cols[3].metric("Steps", len(result.kag_steps))
