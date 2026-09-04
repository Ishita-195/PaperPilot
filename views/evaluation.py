"""Evaluation dashboard — runs the RAGAS-style LLM-as-judge suite in-app."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backend import evaluation
from views.common import load_agent

BASELINE_PATH = Path(__file__).resolve().parent.parent / "ragas_baseline.json"
TARGETS = {"faithfulness": 0.70, "answer_relevancy": 0.75, "context_precision": 0.70}
LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
}

st.title("Evaluation")
st.caption(
    "A RAGAS-style suite scores the agent on golden questions using an "
    "LLM-as-judge: faithfulness (is the answer grounded?), answer relevancy "
    "(does it address the question?), and context precision (was the "
    "retrieval on-topic?)."
)

run = st.button("Run evaluation", type="primary")
st.caption(
    f"Runs {len(evaluation.EVAL_DATA)} questions through the full agent "
    "pipeline — takes about a minute."
)

if run:
    try:
        from langchain_groq import ChatGroq

        from backend.agent import LLM_MODEL

        with st.spinner("Loading agent..."):
            app, embedder, collection = load_agent([])
        judge = ChatGroq(model=LLM_MODEL, temperature=0)

        def ask_fn(question):
            config = {"configurable": {"thread_id": "eval-dashboard"}}
            return app.invoke({"question": question, "eval_retries": 0}, config)

        bar = st.progress(0.0, text="Starting evaluation...")

        def on_progress(i, total, question):
            bar.progress((i - 1) / total, text=f"Question {i}/{total}: {question}")

        results = evaluation.run_evaluation(ask_fn, judge, progress=on_progress)
        bar.progress(1.0, text="Done")
        st.session_state.eval_results = results
        bar.empty()
    except Exception as e:
        st.error(f"Evaluation failed: {e}")

results = st.session_state.get("eval_results")
source_note = "This session"
if results is None and BASELINE_PATH.exists():
    try:
        results = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        source_note = "Saved baseline (ragas_baseline.json)"
    except Exception:
        results = None

if not results:
    st.info("No results yet — click **Run evaluation** to score the agent.")
else:
    st.markdown(f"**Results — {source_note}**")

    cols = st.columns(3)
    for col, key in zip(cols, TARGETS):
        value = results.get(key)
        if value is None:
            col.metric(LABELS[key], "–")
            continue
        target = TARGETS[key]
        col.metric(
            LABELS[key],
            f"{value:.2f}",
            delta=f"{value - target:+.2f} vs target {target:.2f}",
        )

    per_question = results.get("per_question", [])
    if per_question:
        df = pd.DataFrame(per_question)
        df["question_short"] = df["question"].str.slice(0, 40)

        st.markdown("**Faithfulness by question**")
        st.bar_chart(df.set_index("question_short")["faithfulness"], height=220)

        st.markdown("**Per-question scores**")
        st.dataframe(
            df[["question", "faithfulness", "answer_relevancy", "context_precision"]],
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download results (.json)",
        json.dumps(results, indent=2),
        file_name="paperpilot_evaluation.json",
        mime="application/json",
    )

    method = results.get("evaluation_method")
    if method:
        st.caption(f"Method: {method}")
