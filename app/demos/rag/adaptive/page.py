from pathlib import Path

import streamlit as st

from core.mongo import get_collection
from core.ui import clear_data_button, evidence_view, graph_diagram, metrics_row, pipeline_ribbon, upload_widget
from demos.rag.adaptive.pipeline import COLLECTION_NAME, GRAPH, ask_detailed, ingest

DEMO = "adaptive"
STAGES = ["parse", "chunk", "embed", "index", "route", "retrieve", "generate", "answer"]

EXAMPLES = [
    ("no_retrieval", "Hello, how are you today?"),
    ("single_step", "What is the main topic of this document?"),
    ("multi_step", "Compare what the earliest and the latest sections say, and explain how they connect."),
]

st.title("Adaptive RAG", anchor=False)
st.caption(
    "A router looks at each question before doing anything else, and sends it down one of three "
    "paths: skip retrieval entirely, retrieve once, or break the question into sub-questions and "
    "retrieve for each in turn."
)

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages per retrieval", 1, 10, 5, key=f"{DEMO}_top_k")
max_subquestions = st.sidebar.slider("Max sub-questions (multi-step)", 2, 4, 3, key=f"{DEMO}_max_subquestions")

if upload is not None:
    pdf_bytes, uploaded_doc_id = upload
    if collection.count_documents({"doc_id": uploaded_doc_id}, limit=1) == 0:
        with st.status("Ingesting document...", expanded=True) as status:
            status.write("Extracting text with PyMuPDF...")
            status.write("Splitting into chunks and embedding with bge-small-en-v1.5...")
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
    for key in ("doc_id", "result", "state", "question", "pending"):
        st.session_state.pop(f"{DEMO}_{key}", None)
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
graph_state = st.session_state.get(f"{DEMO}_state")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else -1)

st.caption("Try one example per path:")
example_cols = st.columns(3)
for col, (route, example_question) in zip(example_cols, EXAMPLES):
    if col.button(example_question, key=f"{DEMO}_example_{route}", disabled=doc_id is None, use_container_width=True):
        st.session_state[f"{DEMO}_question"] = example_question
        st.session_state[f"{DEMO}_pending"] = example_question
        st.rerun()

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask about the document, or try an example above",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

pending = st.session_state.pop(f"{DEMO}_pending", None)
if pending is not None:
    question, submitted = pending, True

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Routing, then retrieving and answering..."):
            try:
                result, graph_state = ask_detailed(question, doc_id, top_k=top_k, max_subquestions=max_subquestions)
            except Exception:
                result, graph_state = None, None
                st.error(
                    "Couldn't get an answer — the chat model (no provider configured, or all of them "
                    "rate-limited) or MongoDB may be unavailable right now. Try again in a moment."
                )
        st.session_state[f"{DEMO}_result"] = result
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.session_state[f"{DEMO}_state"] = graph_state

if result is not None and graph_state is not None:
    st.subheader("Answer")
    st.write(result.answer)
    st.caption(f"Path taken: **{graph_state['route']}** — {graph_state['route_reason']}")

    if graph_state["route"] == "multi_step":
        st.subheader("Sub-questions")
        for i, hop in enumerate(graph_state["hops"], start=1):
            with st.container(border=True):
                st.markdown(f"**Hop {i}: {hop['question']}**")
                st.caption(f"{len(hop['passages'])} passage(s) retrieved.")
                st.write(hop["answer"])

    if result.contexts:
        st.subheader("Evidence")
        st.caption("Passages retrieved on the path this question took.")
        evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])
    elif graph_state["route"] == "no_retrieval":
        st.info("No passages retrieved — the router judged this didn't need the document.")

    st.subheader("Graph")
    st.caption("The path this question actually took is outlined.")
    graph_diagram(GRAPH.get_graph().draw_mermaid(), graph_state["path"])
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
