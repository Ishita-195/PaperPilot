"""Ablation study: does reranking + the faithfulness-retry gate actually help?

Claims in the README should be backed by a number. This harness runs the SAME
questions through the SAME knowledge base with one component toggled at a time
and reports the delta.

Run:
    python -m backend.ablation

Writes ``ablation_results.json`` to the project root.

Two experiments:

1. Reranking (cross-encoder) ON vs OFF
   Metric: mean CONTEXT PRECISION (LLM-as-judge) over the eval questions.
   Reranking is supposed to push the most relevant passages to the top, so
   this is the metric it should move.

2. Faithfulness-retry gate ON vs OFF
   Metric: mean FAITHFULNESS and the number of answers RESCUED — answers that
   scored below the 0.7 threshold on the first pass but reached >= 0.7 after a
   bounded retry (which widens retrieval). Also reports how often the gate
   fires, so a "coverage is already good, gate rarely triggers" result is
   reported honestly rather than hidden.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from backend.agent import build_app, make_llm, FAITHFULNESS_THRESHOLD
from backend import evaluation
from backend.evaluation import EVAL_DATA

# In-scope questions, including a few that stress single-pass retrieval, so the
# retry gate has a chance to demonstrate value rather than never firing.
STRESS_QUESTIONS = [
    {"question": "How does regularization relate to the bias-variance tradeoff?"},
    {"question": "Why can ensemble methods like boosting reduce overfitting?"},
    {"question": "How is cross-validation used when comparing models?"},
]
ALL_QUESTIONS = EVAL_DATA + STRESS_QUESTIONS


def _run(app, question: str) -> Dict:
    config = {"configurable": {"thread_id": f"ablate-{hash(question) & 0xffff}"}}
    return app.invoke(
        {"question": question, "messages": [], "eval_retries": 0}, config
    )


def rerank_experiment(llm) -> Dict:
    """Context precision with reranking ON vs OFF (retry disabled to isolate)."""
    app_on, *_ = build_app(enable_rerank=True, enable_retry=False)
    app_off, *_ = build_app(enable_rerank=False, enable_retry=False)

    on_scores, off_scores, per_q = [], [], []
    for item in EVAL_DATA:
        q = item["question"]
        r_on = _run(app_on, q)
        r_off = _run(app_off, q)
        p_on = evaluation.score_context_precision(llm, q, r_on.get("retrieved", ""))
        p_off = evaluation.score_context_precision(llm, q, r_off.get("retrieved", ""))
        on_scores.append(p_on)
        off_scores.append(p_off)
        per_q.append({
            "question": q,
            "context_precision_rerank_off": round(p_off, 3),
            "context_precision_rerank_on": round(p_on, 3),
            "delta": round(p_on - p_off, 3),
        })
        print(f"  [rerank] {q[:45]:<45} off={p_off:.2f} on={p_on:.2f}")

    n = len(EVAL_DATA)
    mean_on = sum(on_scores) / n
    mean_off = sum(off_scores) / n
    return {
        "metric": "context_precision (LLM-as-judge)",
        "rerank_off": round(mean_off, 3),
        "rerank_on": round(mean_on, 3),
        "absolute_gain": round(mean_on - mean_off, 3),
        "relative_gain_pct": round((mean_on - mean_off) / mean_off * 100, 1) if mean_off else None,
        "per_question": per_q,
    }


def retry_experiment(llm) -> Dict:
    """Faithfulness with the retry gate ON vs OFF over all questions."""
    app_on, *_ = build_app(enable_rerank=True, enable_retry=True)
    app_off, *_ = build_app(enable_rerank=True, enable_retry=False)

    on_scores, off_scores, per_q = [], [], []
    rescued = 0
    fired = 0
    for item in ALL_QUESTIONS:
        q = item["question"]
        r_off = _run(app_off, q)           # single pass
        r_on = _run(app_on, q)             # gate may retry
        f_off = float(r_off.get("faithfulness", 0.0))
        f_on = float(r_on.get("faithfulness", 0.0))
        retries = int(r_on.get("eval_retries", 0))
        on_scores.append(f_on)
        off_scores.append(f_off)
        if retries > 0:
            fired += 1
        was_rescued = f_off < FAITHFULNESS_THRESHOLD <= f_on
        if was_rescued:
            rescued += 1
        per_q.append({
            "question": q,
            "faithfulness_single_pass": round(f_off, 3),
            "faithfulness_with_retry": round(f_on, 3),
            "retries_used": retries,
            "rescued": was_rescued,
        })
        print(f"  [retry]  {q[:45]:<45} pass={f_off:.2f} final={f_on:.2f} "
              f"retries={retries}{'  RESCUED' if was_rescued else ''}")

    n = len(ALL_QUESTIONS)
    flagged_single_pass = sum(1 for s in off_scores if s < FAITHFULNESS_THRESHOLD)
    return {
        "metric": "faithfulness (LLM-as-judge)",
        "num_questions": n,
        "retry_off_mean": round(sum(off_scores) / n, 3),
        "retry_on_mean": round(sum(on_scores) / n, 3),
        "flagged_on_first_pass": flagged_single_pass,
        "gate_fired": fired,
        "answers_rescued": rescued,
        "min_faithfulness_shipped_off": round(min(off_scores), 3),
        "min_faithfulness_shipped_on": round(min(on_scores), 3),
        "per_question": per_q,
    }


def run() -> Dict:
    llm = make_llm()
    print("\n=== Experiment 1: reranking ON vs OFF (context precision) ===")
    rerank = rerank_experiment(llm)
    print("\n=== Experiment 2: faithfulness-retry gate ON vs OFF ===")
    retry = retry_experiment(llm)
    return {
        "reranking": rerank,
        "faithfulness_retry": retry,
        "note": "Same KB and questions across conditions; only the named "
                "component is toggled.",
    }


if __name__ == "__main__":
    results = run()
    rr, rt = results["reranking"], results["faithfulness_retry"]

    print("\n" + "=" * 56)
    print("HEADLINE RESULTS")
    print("=" * 56)
    print(f"Reranking -> context precision: {rr['rerank_off']} -> {rr['rerank_on']} "
          f"(+{rr['absolute_gain']}"
          + (f", +{rr['relative_gain_pct']}%)" if rr['relative_gain_pct'] is not None else ")"))
    print(f"Retry gate -> mean faithfulness: {rt['retry_off_mean']} -> {rt['retry_on_mean']}")
    print(f"Retry gate -> flagged on first pass: {rt['flagged_on_first_pass']}, "
          f"rescued: {rt['answers_rescued']} "
          f"(worst answer shipped: {rt['min_faithfulness_shipped_off']} without gate "
          f"vs {rt['min_faithfulness_shipped_on']} with gate)")

    out = Path(__file__).resolve().parent.parent / "ablation_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
