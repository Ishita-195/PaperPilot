# eval.py — Manual RAGAS-style evaluation (Free alternative, no RAGAS Collections)

from agent import build_app
from langchain_groq import ChatGroq
import json

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

print("[eval] Building agent...")
app, embedder, collection = build_app()

def ask(question, thread_id="eval-1"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"question": question, "messages": [], "eval_retries": 0},
        config
    )
    return result

# ================= EVALUATION DATA =================
eval_data = [
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

# ================= MANUAL EVALUATION FUNCTIONS =================

def score_faithfulness(answer, context):
    """Score 0-1: Is the answer grounded in the context?"""
    prompt = f"""Rate how faithful the answer is to the provided context (0.0-1.0).
0.0 = completely made up, not in context
1.0 = entirely grounded in context, no hallucination

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))  # Clamp to 0-1
    except:
        return 0.5  # Default if parsing fails

def score_answer_relevancy(question, answer):
    """Score 0-1: How relevant is the answer to the question?"""
    prompt = f"""Rate how relevant the answer is to the question (0.0-1.0).
0.0 = completely irrelevant, answers different question
1.0 = perfectly answers the question asked

QUESTION:
{question}

ANSWER:
{answer}

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5

def score_context_precision(question, contexts_str):
    """Score 0-1: How much of the retrieved context is relevant to the question?"""
    prompt = f"""Rate how precisely the retrieved context matches the question (0.0-1.0).
0.0 = context is completely irrelevant to question
1.0 = all retrieved context is highly relevant

QUESTION:
{question}

RETRIEVED CONTEXT:
{contexts_str}

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5

# ================= RUN EVALUATION =================
print("\n" + "="*60)
print("MANUAL RAGAS-STYLE EVALUATION (Free)")
print("="*60 + "\n")

faithfulness_scores = []
answer_relevancy_scores = []
context_precision_scores = []

for i, item in enumerate(eval_data, 1):
    question = item['question']
    ground_truth = item['ground_truth']
    
    print(f"[Q{i}] {question}")
    
    result = ask(question, thread_id=f"eval-{i}")
    
    answer = result.get("answer", "")
    contexts = result.get("retrieved", "")
    sources = result.get("sources", [])
    
    print(f"  Answer: {answer[:100]}...")
    print(f"  Sources: {sources}")
    
    # Score this Q&A pair
    print(f"  Scoring... (this takes ~10-15 seconds per question)")
    
    faithfulness = score_faithfulness(answer, contexts)
    answer_relevancy = score_answer_relevancy(question, answer)
    context_precision = score_context_precision(question, contexts)
    
    faithfulness_scores.append(faithfulness)
    answer_relevancy_scores.append(answer_relevancy)
    context_precision_scores.append(context_precision)
    
    print(f"    ✓ Faithfulness: {faithfulness:.3f}")
    print(f"    ✓ Answer Relevancy: {answer_relevancy:.3f}")
    print(f"    ✓ Context Precision: {context_precision:.3f}")
    print()

# ================= COMPUTE AVERAGES =================
avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
avg_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores)
avg_context_precision = sum(context_precision_scores) / len(context_precision_scores)

print("="*60)
print("RAGAS BASELINE SCORES (Manual Evaluation)")
print("="*60)
print(f"Faithfulness:        {avg_faithfulness:.3f}")
print(f"Answer Relevancy:    {avg_answer_relevancy:.3f}")
print(f"Context Precision:   {avg_context_precision:.3f}")
print("="*60 + "\n")

# Save results
baseline_data = {
    "faithfulness": round(avg_faithfulness, 3),
    "answer_relevancy": round(avg_answer_relevancy, 3),
    "context_precision": round(avg_context_precision, 3),
    "evaluation_method": "Manual LLM-based scoring (no RAGAS Collections)",
    "num_questions": len(eval_data),
    "individual_scores": {
        "faithfulness": [round(s, 3) for s in faithfulness_scores],
        "answer_relevancy": [round(s, 3) for s in answer_relevancy_scores],
        "context_precision": [round(s, 3) for s in context_precision_scores],
    }
}

with open("ragas_baseline.json", "w") as f:
    json.dump(baseline_data, f, indent=2)

print("✅ Baseline saved to ragas_baseline.json")
print("\n📊 Interpretation:")
print("   • Faithfulness ≥0.70 = Agent grounded in KB (no hallucination)")
print("   • Answer Relevancy ≥0.75 = Answers match questions")
print("   • Context Precision ≥0.70 = Retrieval quality is good")