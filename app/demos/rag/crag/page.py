from pathlib import Path

import streamlit as st

from core.mongo import get_collection
from core.ui import clear_data_button, evidence_view, graph_diagram, metrics_row, pipeline_ribbon, readme_view, upload_widget
from demos.rag.crag.pipeline import COLLECTION_NAME, GRAPH, ask_detailed, ingest

DEMO = "crag"
STAGES = ["parse", "chunk", "embed", "index", "retrieve", "grade", "web fallback", "answer"]

st.title("CRAG", anchor=False)
st.caption(
    "Retrieved passages are graded for relevance before they're trusted. When none of them "
    "clearly answer the question, the query is rewritten and answered from a live web search instead."
)

collection = get_collection(COLLECTION_NAME)
upload = upload_widget(DEMO)
st.sidebar.caption(
    "Reads a PDF's text layer only — scanned pages and text inside images aren't extracted."
)

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 5, key=f"{DEMO}_top_k")
web_results_n = st.sidebar.slider("Web results on fallback", 1, 5, 3, key=f"{DEMO}_web_results_n")

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
    for key in ("doc_id", "result", "state", "question"):
        st.session_state.pop(f"{DEMO}_{key}", None)
    st.rerun()

result = st.session_state.get(f"{DEMO}_result")
graph_state = st.session_state.get(f"{DEMO}_state")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else -1)

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question",
        key=f"{DEMO}_question",
        disabled=doc_id is None,
        placeholder="Upload a PDF first" if doc_id is None else "Ask about the document, or something off-topic",
    )
    submitted = st.form_submit_button("Ask", disabled=doc_id is None)

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving, grading, and answering..."):
            try:
                result, graph_state = ask_detailed(question, doc_id, top_k=top_k, web_results_n=web_results_n)
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
    if graph_state["decision"] == "insufficient":
        st.info("Web fallback triggered — no retrieved passage was graded relevant.")

    st.subheader("Grades")
    st.caption("Every retrieved passage, graded for relevance to the question before anything is trusted.")
    if graph_state["grades"]:
        st.dataframe(
            [
                {"Page": g["page"], "Score": round(g["score"], 3), "Grade": g["grade"], "Text": g["text"][:120]}
                for g in graph_state["grades"]
            ],
            hide_index=True,
        )
    else:
        st.caption("No passages were retrieved.")

    if graph_state["decision"] == "insufficient":
        st.subheader("Web fallback")
        st.caption(f"Rewritten query: `{graph_state['rewritten_query']}`")
        if graph_state["web_results"]:
            st.dataframe(
                [
                    {"Title": w["title"], "URL": w["url"], "Snippet": w["snippet"][:160]}
                    for w in graph_state["web_results"]
                ],
                hide_index=True,
            )
        else:
            st.warning("Web search returned nothing, or was unreachable — answered from ambiguous passages alone.")

    if result.contexts:
        st.subheader("Evidence used")
        st.caption("The document passages actually fed to the answer (relevant ones, or ambiguous ones on fallback).")
        evidence_view([{"text": c.text, "score": c.score, "page": c.page} for c in result.contexts])

    st.subheader("Graph")
    st.caption("The path this question actually took is outlined.")
    graph_diagram(GRAPH.get_graph().draw_mermaid(), graph_state["path"])
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
