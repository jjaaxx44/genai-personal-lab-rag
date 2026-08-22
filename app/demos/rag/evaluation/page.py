from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import get_settings
from core.ui import upload_widget
from demos.rag.evaluation.pipeline import DEMOS, METRICS, TestCase, generate_test_set, push_scores_to_langfuse, run_evaluation

DEMO = "evaluation"

st.title("RAG evaluation", anchor=False)
st.caption("Ragas scores faithfulness, answer relevancy, context precision and context recall across techniques.")

upload = upload_widget(DEMO)
st.sidebar.caption("The same PDF and test set are used to compare every selected demo.")

if upload is not None:
    pdf_bytes, doc_id = upload
    st.session_state[f"{DEMO}_pdf_bytes"] = pdf_bytes
    st.session_state[f"{DEMO}_doc_id"] = doc_id

pdf_bytes = st.session_state.get(f"{DEMO}_pdf_bytes")
doc_id = st.session_state.get(f"{DEMO}_doc_id")

st.sidebar.subheader("Settings")
settings = get_settings()
test_set_size = st.sidebar.slider("Test questions to generate", 3, 15, settings.eval_test_set_size, key=f"{DEMO}_size")
demo_names = st.sidebar.multiselect(
    "Demos to compare", list(DEMOS.keys()), default=list(DEMOS.keys()), key=f"{DEMO}_demos"
)
st.sidebar.caption(
    "Each question runs once per demo, plus several more chat-model calls per question for the "
    "Ragas metrics — this burns free-tier quota fast. A sample that hits a rate limit shows as "
    "unavailable rather than failing the whole run."
)

if st.sidebar.button("Reset test set and results", key=f"{DEMO}_reset"):
    for key in (f"{DEMO}_test_set", f"{DEMO}_results"):
        st.session_state.pop(key, None)
    st.rerun()

results = st.session_state.get(f"{DEMO}_results")
test_cases: list[TestCase] = []

if pdf_bytes is None:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    st.subheader("Test set")
    st.caption(
        "Generated from random passages of the document. Edit questions and reference answers, or add your own."
    )

    if st.button("Generate test set", disabled=pdf_bytes is None):
        with st.spinner(f"Generating {test_set_size} question/answer pairs..."):
            try:
                generated = generate_test_set(pdf_bytes, test_set_size)
            except Exception:
                generated = []
                st.error(
                    "Couldn't generate a test set — no LLM provider is configured, or every configured "
                    "provider is rate-limited or unavailable right now."
                )
        if generated:
            st.session_state[f"{DEMO}_test_set"] = [tc.model_dump() for tc in generated]
        elif pdf_bytes is not None:
            st.warning("No extractable text came out of this PDF, so no test set could be generated.")

    test_set_rows = st.session_state.get(f"{DEMO}_test_set", [])
    edited = st.data_editor(
        pd.DataFrame(test_set_rows, columns=["question", "reference_answer"]),
        num_rows="dynamic",
        hide_index=True,
        key=f"{DEMO}_editor",
    )
    st.session_state[f"{DEMO}_test_set"] = edited.to_dict("records")

    test_cases = [
        TestCase(question=row["question"], reference_answer=row["reference_answer"])
        for row in st.session_state.get(f"{DEMO}_test_set", [])
        if row.get("question") and row.get("reference_answer")
    ]

    st.divider()
    st.subheader("Run")
    run_disabled = not test_cases or not demo_names
    run = st.button("Run evaluation", type="primary", disabled=run_disabled)
    if not test_cases:
        st.caption("Generate or enter at least one question/answer pair first.")
    elif not demo_names:
        st.caption("Select at least one demo to compare.")

    if run:
        results = {}
        progress = st.progress(0.0)
        for i, demo_name in enumerate(demo_names):
            with st.spinner(f"Evaluating {demo_name} ({len(test_cases)} question(s))..."):
                results[demo_name] = run_evaluation(pdf_bytes, doc_id, demo_name, test_cases)
            progress.progress((i + 1) / len(demo_names))
        progress.empty()
        st.session_state[f"{DEMO}_results"] = results
        # A run switches the reader to the trace; the tab is still clickable back.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        try:
            push_scores_to_langfuse(results)
        except Exception:
            pass

    if results:
        st.subheader("Results")
        for demo_name, outcome in results.items():
            if "error" in outcome:
                st.error(f"**{demo_name}**: {outcome['error']}")
            elif "warning" in outcome:
                st.warning(outcome["warning"])

        summary_rows = {
            demo_name: outcome["means"] for demo_name, outcome in results.items() if "means" in outcome
        }
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).T.reindex(columns=METRICS)
            st.dataframe(summary_df.style.format("{:.2f}", na_rep="—"))
            st.bar_chart(summary_df)

            for demo_name, outcome in results.items():
                if outcome.get("failed_questions"):
                    st.caption(
                        f"{demo_name}: {len(outcome['failed_questions'])} question(s) couldn't be "
                        "answered and were excluded from its scores."
                    )

            st.download_button(
                "Download results as CSV",
                summary_df.to_csv().encode("utf-8"),
                file_name="rag_evaluation_results.csv",
                mime="text/csv",
            )
    else:
        st.caption("Run an evaluation to see results here.")

how_tab, trace_tab = st.tabs(
    ["How it works", "Trace"], key=f"{DEMO}_tabs", on_change="rerun"
)
with how_tab:
    st.markdown((Path(__file__).parent / "README.md").read_text())
with trace_tab:
    if results:
        for demo_name, outcome in results.items():
            st.markdown(f"**{demo_name}**")
            if "error" in outcome:
                st.caption(outcome["error"])
                continue
            st.write(f"- Ingested (if needed) and answered {len(test_cases)} question(s).")
            st.write(f"- Scored with Ragas in {outcome['latency_ms']:.0f} ms.")
            if outcome.get("warning"):
                st.caption(outcome["warning"])
            if outcome["failed_questions"]:
                st.write(f"- {len(outcome['failed_questions'])} question(s) failed and were excluded.")
    else:
        st.caption("Run an evaluation to see the trace.")
