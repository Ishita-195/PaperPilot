"""Validate the LLM-as-judge faithfulness score against human labels.

An LLM scoring its own faithfulness is only meaningful if those scores agree
with a human. This harness runs ``evaluation.score_faithfulness`` over the
hand-labeled set in ``backend/human_labels.py`` and reports how well the judge
matches the human verdicts.

Run:
    python -m backend.validate_judge

Outputs a summary table to stdout and writes ``judge_validation.json`` to the
project root. Pure-Python metrics (no scikit-learn) so it runs in the same
lightweight environment as the rest of the backend.

Positive detection class = "hallucination" (human label 0): the whole point of
the judge is to CATCH unfaithful answers, so precision/recall are reported for
that class using the production threshold (score < 0.7 => flagged).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from backend.agent import FAITHFULNESS_THRESHOLD, make_llm
from backend.human_labels import HUMAN_LABELS, counts
from backend import evaluation


# ----------------------------- metrics -----------------------------
def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _auc(scores: List[float], labels: List[int]) -> float:
    """ROC-AUC via the Mann-Whitney U rank formula (positive class = faithful=1).

    Interpreted as: probability the judge gives a random faithful answer a
    higher score than a random unfaithful one. Ties are handled with average
    ranks, so a judge that separates the two groups perfectly scores 1.0.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.0
    # Average-rank of the pooled scores (1-indexed, ascending).
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average of 1-indexed ranks in the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def validate(llm=None) -> Dict:
    llm = llm or make_llm()
    scores: List[float] = []
    labels: List[int] = []
    rows = []

    for item in HUMAN_LABELS:
        s = evaluation.score_faithfulness(llm, item["answer"], item["context"])
        scores.append(s)
        labels.append(item["label"])
        judge_faithful = s >= FAITHFULNESS_THRESHOLD
        human_faithful = item["label"] == 1
        rows.append({
            "human": "faithful" if human_faithful else "unfaithful",
            "judge_score": round(s, 3),
            "judge_verdict": "faithful" if judge_faithful else "flagged",
            "agree": judge_faithful == human_faithful,
            "answer": item["answer"][:70] + ("..." if len(item["answer"]) > 70 else ""),
        })

    n = len(labels)
    # Threshold agreement.
    correct = sum(1 for r in rows if r["agree"])
    accuracy = correct / n

    # Hallucination detection: positive = flagged / human unfaithful.
    tp = sum(1 for s, l in zip(scores, labels) if l == 0 and s < FAITHFULNESS_THRESHOLD)
    fp = sum(1 for s, l in zip(scores, labels) if l == 1 and s < FAITHFULNESS_THRESHOLD)
    fn = sum(1 for s, l in zip(scores, labels) if l == 0 and s >= FAITHFULNESS_THRESHOLD)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    faithful_scores = [s for s, l in zip(scores, labels) if l == 1]
    unfaithful_scores = [s for s, l in zip(scores, labels) if l == 0]

    return {
        "num_examples": n,
        "label_distribution": counts(),
        "threshold": FAITHFULNESS_THRESHOLD,
        "judge_vs_human": {
            "accuracy": round(accuracy, 3),
            "hallucination_precision": round(precision, 3),
            "hallucination_recall": round(recall, 3),
            "hallucination_f1": round(f1, 3),
            "pearson_correlation": round(_pearson(scores, [float(l) for l in labels]), 3),
            "roc_auc": round(_auc(scores, labels), 3),
        },
        "mean_score": {
            "on_faithful": round(sum(faithful_scores) / len(faithful_scores), 3) if faithful_scores else None,
            "on_unfaithful": round(sum(unfaithful_scores) / len(unfaithful_scores), 3) if unfaithful_scores else None,
        },
        "confusion_at_threshold": {
            "true_positive_flagged": tp, "false_positive_flagged": fp,
            "false_negative_missed": fn,
            "true_negative_passed": n - tp - fp - fn,
        },
        "evaluation_method": "LLM-as-judge (Groq Llama 3.3 70B) vs. human labels",
        "per_example": rows,
    }


if __name__ == "__main__":
    print(f"[validate] {counts()}")
    print("[validate] scoring judge against human labels...\n")
    results = validate()

    jvh = results["judge_vs_human"]
    print(f"{'human':<12}{'judge':>8}  verdict")
    print("-" * 40)
    for r in results["per_example"]:
        mark = " " if r["agree"] else "  <-- disagree"
        print(f"{r['human']:<12}{r['judge_score']:>8}  {r['judge_verdict']}{mark}")
    print("-" * 40)
    print(f"accuracy vs human ....... {jvh['accuracy']}")
    print(f"hallucination recall .... {jvh['hallucination_recall']}  "
          f"(precision {jvh['hallucination_precision']}, F1 {jvh['hallucination_f1']})")
    print(f"ROC-AUC ................. {jvh['roc_auc']}")
    print(f"Pearson correlation ..... {jvh['pearson_correlation']}")
    print(f"mean score faithful/unfaithful ... "
          f"{results['mean_score']['on_faithful']} / {results['mean_score']['on_unfaithful']}")

    out = Path(__file__).resolve().parent.parent / "judge_validation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
