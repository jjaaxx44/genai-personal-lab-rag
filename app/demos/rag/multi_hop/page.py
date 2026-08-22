from pathlib import Path

import streamlit as st

from core.mongo import get_collection
from core.ui import clear_data_button, evidence_view, graph_diagram, metrics_row, pipeline_ribbon, upload_widget
from demos.rag.multi_hop.pipeline import COLLECTION_NAME, GRAPH, ask_detailed, ingest

DEMO = "multi_hop"
STAGES = ["parse", "chunk", "embed", "index", "decompose", "hop", "combine", "answer"]

EXAMPLE_QUESTION = "Compare what the earliest and the latest sections say, and explain how they connect."

st.title("Multi-hop RAG", anchor=False)
st.caption(
    "The question is broken into a chain of sub-questions first. Each hop retrieves and answers its "
    "own sub-question — using earlier hops' answers as context — before the hops are combined into "
    "one final answer."
)

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages per hop", 1, 10, 5, key=f"{DEMO}_top_k")
max_hops = st.sidebar.slider("Max hops", 2, 4, 2, key=f"{DEMO}_max_hops")
st.sidebar.caption(
    "The decompose step can use fewer hops than this if the question doesn't need them."
)

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

if st.button(f'Try an example: "{EXAMPLE_QUESTION}"', key=f"{DEMO}_example", disabled=doc_id is None):
    st.session_state[f"{DEMO}_question"] = EXAMPLE_QUESTION
    st.session_state[f"{DEMO}_pending"] = EXAMPLE_QUESTION
    st.rerun()

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask a question that needs multiple hops",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

pending = st.session_state.pop(f"{DEMO}_pending", None)
if pending is not None:
    question, submitted = pending, True

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Decomposing into hops, then retrieving and answering each..."):
            try:
                result, graph_state = ask_detailed(question, doc_id, top_k=top_k, max_hops=max_hops)
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

    st.subheader("Hops")
    st.caption(f"The question was broken into {len(graph_state['sub_questions'])} hop(s).")
    for i, hop in enumerate(graph_state["hops"], start=1):
        with st.container(border=True):
            st.markdown(f"**Hop {i}: {hop['question']}**")
            st.caption(f"{len(hop['passages'])} passage(s) retrieved as context for this hop.")
            if hop["context"]:
                with st.expander("Context used for this hop"):
                    st.text(hop["context"])
            else:
                st.caption("No passages retrieved for this hop.")
            st.markdown("Partial answer:")
            st.write(hop["answer"])

    if result.contexts:
        st.subheader("Evidence")
        st.caption("Every passage retrieved across all hops.")
        evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])

    st.subheader("Graph")
    st.caption("The path this question actually took is outlined — a longer path means more hops.")
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
