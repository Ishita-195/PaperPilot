"""Human-labeled validation set for the faithfulness judge.

Each item pairs a retrieved ``context`` with an ``answer`` and a human
``label``:

    1 = FAITHFUL     — every claim in the answer is supported by the context
    0 = UNFAITHFUL   — the answer adds, invents, or contradicts the context
                       (i.e. a hallucination the judge must catch)

The set is deliberately balanced (8 faithful / 8 unfaithful) and grounded in
the real knowledge base (``backend/kb.py``). It exists so we can measure
whether the LLM-as-judge faithfulness score actually agrees with a human —
run ``python -m backend.validate_judge`` to score the judge against these
labels.

Labels were assigned by hand; the ``rationale`` records why. Unfaithful cases
use plausible-sounding ML statements that are NOT in the given context (or
that contradict it), which is exactly the failure mode the judge exists to
flag before an answer reaches a user.
"""
from __future__ import annotations

# Real KB passages (verbatim from backend/kb.py) reused as retrieval context.
_XGBOOST = "[XGBoost] XGBoost is a highly efficient implementation of gradient boosting. It builds decision trees sequentially and minimizes a loss function using gradient descent. It includes regularization (L1 and L2), supports parallel processing, handles missing values automatically, and is known for high performance in structured data tasks."
_RANDOM_FOREST = "[Random Forest] Random Forest is an ensemble learning method that constructs multiple decision trees using random subsets of data and features. Predictions are made by averaging or voting. It reduces overfitting and improves generalization compared to a single decision tree."
_SHAP = "[SHAP] SHAP (SHapley Additive exPlanations) explains model predictions by assigning each feature an importance value based on cooperative game theory. It provides both global and local interpretability."
_METRICS = "[Evaluation Metrics] Model evaluation uses metrics like accuracy, precision, recall, F1-score, and ROC-AUC. For imbalanced datasets, precision-recall and F1-score are more informative than accuracy."
_TRANSFORMERS = "[Transformers] Transformers are deep learning models based on self-attention mechanisms. They process entire sequences in parallel and are widely used in NLP tasks like translation, summarization, and chatbots."
_CNN = "[CNN] Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data such as images. They use convolutional layers to extract spatial features and are widely used in computer vision."
_OVERFITTING = "[Overfitting] Overfitting occurs when a model learns noise instead of patterns, performing well on training data but poorly on unseen data."
_DIMRED = "[Dimensionality Reduction] Dimensionality reduction reduces the number of features while preserving important information. Techniques include PCA, t-SNE, and UMAP."


HUMAN_LABELS = [
    # ================= FAITHFUL (label = 1) =================
    {
        "context": _XGBOOST,
        "answer": "XGBoost is an efficient implementation of gradient boosting that builds decision trees sequentially and minimizes a loss function with gradient descent. It uses L1 and L2 regularization and supports parallel processing.",
        "label": 1,
        "rationale": "Every claim is directly stated in the context.",
    },
    {
        "context": _RANDOM_FOREST,
        "answer": "Random Forest builds many decision trees on random subsets of the data and features, then averages or votes over them, which reduces overfitting versus a single tree.",
        "label": 1,
        "rationale": "Accurate paraphrase of the context.",
    },
    {
        "context": _SHAP,
        "answer": "SHAP assigns each feature an importance value using ideas from cooperative game theory (Shapley values) and offers both global and local interpretability.",
        "label": 1,
        "rationale": "Grounded; matches context.",
    },
    {
        "context": _METRICS,
        "answer": "Common classification metrics include accuracy, precision, recall, F1-score, and ROC-AUC. For imbalanced data, F1-score and precision-recall are more informative than accuracy.",
        "label": 1,
        "rationale": "Directly supported, including the imbalanced-data caveat.",
    },
    {
        "context": _TRANSFORMERS,
        "answer": "Transformers rely on self-attention and process whole sequences in parallel, which makes them effective for NLP tasks like translation and summarization.",
        "label": 1,
        "rationale": "Faithful to context.",
    },
    {
        "context": _CNN,
        "answer": "CNNs use convolutional layers to extract spatial features from grid-like data such as images, and are widely used in computer vision.",
        "label": 1,
        "rationale": "Faithful to context.",
    },
    {
        "context": _OVERFITTING,
        "answer": "Overfitting is when a model learns noise rather than the underlying pattern, so it does well on training data but poorly on unseen data.",
        "label": 1,
        "rationale": "Faithful to context.",
    },
    {
        "context": _DIMRED,
        "answer": "Dimensionality reduction cuts the number of features while keeping the important information; PCA, t-SNE, and UMAP are common techniques.",
        "label": 1,
        "rationale": "Faithful to context.",
    },

    # ================= UNFAITHFUL (label = 0) =================
    {
        "context": _XGBOOST,
        "answer": "XGBoost was developed by Google in 2020 and internally uses deep neural networks rather than decision trees to achieve its speed.",
        "label": 0,
        "rationale": "Invented origin story; contradicts 'builds decision trees'.",
    },
    {
        "context": _RANDOM_FOREST,
        "answer": "Random Forest always outperforms deep learning on every task and needs no hyperparameter tuning because it can never overfit.",
        "label": 0,
        "rationale": "Overclaims not present in context; 'never overfit' contradicts it.",
    },
    {
        "context": _SHAP,
        "answer": "SHAP predates LIME by a decade and only works on tree-based models, so it cannot explain neural networks.",
        "label": 0,
        "rationale": "Chronology and scope claims are absent from and unsupported by context.",
    },
    {
        "context": _METRICS,
        "answer": "The Matthews correlation coefficient is the only trustworthy classification metric, and accuracy should never be reported.",
        "label": 0,
        "rationale": "MCC is not mentioned; the absolute claim is not in context.",
    },
    {
        "context": _TRANSFORMERS,
        "answer": "Transformers process sequences one token at a time using recurrence and convolution, which is why they are slower than RNNs.",
        "label": 0,
        "rationale": "Contradicts 'self-attention' and 'process entire sequences in parallel'.",
    },
    {
        "context": _CNN,
        "answer": "CNNs are primarily designed for tabular time-series forecasting and are unable to process images.",
        "label": 0,
        "rationale": "Directly contradicts the context (images / computer vision).",
    },
    {
        "context": _OVERFITTING,
        "answer": "Overfitting is desirable because it improves accuracy on unseen test data and indicates the model has generalized well.",
        "label": 0,
        "rationale": "Contradicts the context's definition of overfitting.",
    },
    {
        "context": _DIMRED,
        "answer": "Dimensionality reduction works by generating brand-new synthetic features with GANs and autoencoders to increase the feature count.",
        "label": 0,
        "rationale": "Contradicts 'reduces the number of features'; methods not in context.",
    },
]


def counts() -> dict:
    """Return label distribution for a quick sanity check."""
    faithful = sum(1 for x in HUMAN_LABELS if x["label"] == 1)
    return {"total": len(HUMAN_LABELS), "faithful": faithful,
            "unfaithful": len(HUMAN_LABELS) - faithful}
