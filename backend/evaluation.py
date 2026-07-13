"""RAGAS-style evaluation for the ML Research Assistant.

Provides LLM-as-judge scoring functions (faithfulness, answer relevancy,
context precision) plus a batch ``run_evaluation`` used by both the CLI
(``python -m backend.evaluation``) and the FastAPI ``/evaluate`` endpoint.

All scores are in the range [0.0, 1.0].
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ================= GOLDEN EVALUATION SET =================
EVAL_DATA = [
    {
        "question": "What is XGBoost and why is it efficient?",
        "ground_truth": "XGBoost is an optimized gradient boosting implementation that uses regularization (L1/L2) and parallel processing to achieve high performance.",
    },
    {
        "question": "How do SHAP and LIME differ?",
        "ground_truth": "SHAP uses Shapley values for consistent explanations while LIME uses local linear approximations. SHAP is more robust, LIME is faster.",
    },
    {
        "question": "What metrics evaluate classification models?",
        "ground_truth": "Accuracy, precision, recall, F1-score, and ROC-AUC. F1-score is better for imbalanced datasets.",
    },
    {
        "question": "When should I use Random Forest vs SVM?",
        "ground_truth": "Random Forest works well for high-dimensional data with many features. SVM excels on smaller datasets with clear separation.",
    },
    {
        "question": "What is feature engineering?",
        "ground_truth": "Transforming raw data into meaningful features: handling missing values, encoding categorical vars, scaling, and creating interactions.",
    },
]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_score(text: str, default: float = 0.5) -> float:
    """Extract the first float in [0,1] from an LLM response, robustly."""
    match = _NUM_RE.search(text or "")
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group())))
    except ValueError:
        return default


def score_faithfulness(llm, answer: str, context: str) -> float:
    """How grounded is the answer in the retrieved context? (0-1)"""
    if not answer or not context:
        return 0.0
    prompt = f"""Rate how faithful the answer is to the provided context (0.0-1.0).
0.0 = completely made up, not in context
1.0 = entirely grounded in context, no hallucination

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    return _parse_score(llm.invoke(prompt).content)


def score_answer_relevancy(llm, question: str, answer: str) -> float:
    """How relevant is the answer to the question? (0-1)"""
    if not answer:
        return 0.0
    prompt = f"""Rate how relevant the answer is to the question (0.0-1.0).
0.0 = completely irrelevant, answers a different question
1.0 = perfectly answers the question asked

QUESTION:
{question}

ANSWER:
{answer}

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    return _parse_score(llm.invoke(prompt).content)


def score_context_precision(llm, question: str, context: str) -> float:
    """How precisely does the retrieved context match the question? (0-1)"""
    if not context:
        return 0.0
    prompt = f"""Rate how precisely the retrieved context matches the question (0.0-1.0).
0.0 = context is completely irrelevant to the question
1.0 = all retrieved context is highly relevant

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    return _parse_score(llm.invoke(prompt).content)


def run_evaluation(
    ask_fn: Callable[[str], Dict],
    llm,
    eval_data: Optional[List[Dict]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """Run the full evaluation suite.

    ``ask_fn`` takes a question and returns the agent result dict
    (must contain ``answer``, ``retrieved`` and ``sources``).
    ``llm`` is the judge model. Returns an aggregated metrics dict.
    """
    eval_data = eval_data or EVAL_DATA
    faith, relev, prec = [], [], []
    per_question = []

    for i, item in enumerate(eval_data, 1):
        q = item["question"]
        if progress:
            progress(i, len(eval_data), q)
        result = ask_fn(q)
        answer = result.get("answer", "")
        context = result.get("retrieved", "")

        f = score_faithfulness(llm, answer, context)
        a = score_answer_relevancy(llm, q, answer)
        p = score_context_precision(llm, q, context)
        faith.append(f)
        relev.append(a)
        prec.append(p)
        per_question.append({
            "question": q,
            "faithfulness": round(f, 3),
            "answer_relevancy": round(a, 3),
            "context_precision": round(p, 3),
            "sources": result.get("sources", []),
        })

    n = len(eval_data)
    return {
        "faithfulness": round(sum(faith) / n, 3),
        "answer_relevancy": round(sum(relev) / n, 3),
        "context_precision": round(sum(prec) / n, 3),
        "num_questions": n,
        "evaluation_method": "LLM-as-judge (Groq Llama 3.3 70B)",
        "per_question": per_question,
    }


if __name__ == "__main__":
    # CLI: python -m backend.evaluation
    from backend.agent import build_app, make_llm

    print("[eval] Building agent...")
    app, embedder, collection, reranker = build_app()
    llm = make_llm()

    def ask(question: str) -> Dict:
        config = {"configurable": {"thread_id": "eval"}}
        return app.invoke(
            {"question": question, "messages": [], "eval_retries": 0}, config
        )

    def progress(i, total, q):
        print(f"[Q{i}/{total}] {q}")

    results = run_evaluation(ask, llm, progress=progress)
    print(json.dumps(results, indent=2))

    out = Path(__file__).resolve().parent.parent / "ragas_baseline.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved baseline to {out}")
