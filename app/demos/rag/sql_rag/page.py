from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import get_settings
from core.ui import pipeline_ribbon
from demos.rag.sql_rag.pipeline import DB_PATH, ask, describe_schema, get_engine

DEMO = "sql_rag"
STAGES = ["schema", "generate SQL", "run SQL", "answer"]

EXAMPLE_QUESTIONS = [
    "Which 5 artists have the most albums?",
    "What are the top 5 best-selling genres by total invoice amount?",
    "Which employee has supported the most customers?",
    "Delete every customer from the database",  # deliberately a write request, to show the guard rail
]

st.title("SQL RAG", anchor=False)
st.caption("The model writes SQL against a bundled database, runs it read-only, then answers from the result.")
st.sidebar.caption(f"Queries the bundled Chinook sample database ({DB_PATH.name}) — no upload for this demo.")

st.sidebar.subheader("Settings")
settings = get_settings()
row_limit = st.sidebar.slider("Row limit", 5, 100, settings.sql_rag_row_limit, key=f"{DEMO}_row_limit")

with st.expander("Database schema"):
    st.code(describe_schema(get_engine()), language=None)

result = st.session_state.get(f"{DEMO}_result")
pipeline_ribbon(STAGES, active=len(STAGES) if result is not None else -1)

with st.form(key=f"{DEMO}_ask_form"):
    question = st.text_input(
        "Question", key=f"{DEMO}_question", placeholder="Ask a question about the Chinook music store database"
    )
    submitted = st.form_submit_button("Ask")

st.caption("Try: " + " · ".join(f"*{q}*" for q in EXAMPLE_QUESTIONS))

if submitted:
    if not question:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Writing SQL and querying..."):
            try:
                result = ask(question, row_limit=row_limit)
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
    st.subheader("Generated SQL")
    st.code(result.sql, language="sql")
    if result.retried:
        st.caption("The first attempt was rejected — this is the SQL after a fix-and-retry.")

    if result.blocked_reason is not None:
        st.warning(f"Blocked: {result.blocked_reason}")
    else:
        st.subheader("Result")
        st.dataframe(pd.DataFrame(result.rows), width="stretch")
        if result.truncated:
            st.caption(f"Showing the first {row_limit} row(s) — the query returned more.")

    st.subheader("Answer")
    st.write(result.answer)
else:
    st.info("Ask a question about the database to get started.")

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
    cols = st.columns(4)
    cols[0].metric("Latency", f"{result.latency_ms:.0f} ms")
    cols[1].metric("Tokens", f"{result.tokens:,}")
    cols[2].metric("LLM calls", result.llm_calls)
    cols[3].metric("Rows returned", len(result.rows))
