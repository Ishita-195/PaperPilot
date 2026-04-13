from typing import TypedDict, List
from datetime import datetime

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

# ================= STATE =================
class AgentState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str

# ================= DOCUMENTS =================
DOCUMENTS = [
    {"id": "doc_001", "topic": "XGBoost", "text": "XGBoost is an optimized implementation of gradient boosting. It builds trees sequentially and uses regularization (L1 and L2) to reduce overfitting. It supports parallel processing, handles missing values natively, and is widely used in machine learning competitions due to its speed and accuracy."},
    {"id": "doc_002", "topic": "Gradient Boosting", "text": "Gradient Boosting builds models sequentially where each new model corrects the errors of the previous ones. It uses gradient descent to minimize a loss function. It can achieve high accuracy but is slower to train than XGBoost. Key hyperparameters include learning rate, number of estimators, and max depth."},
    {"id": "doc_003", "topic": "Random Forest", "text": "Random Forest is an ensemble method that builds multiple decision trees independently using random subsets of data and features, then combines their outputs by voting. It reduces overfitting compared to single trees and improves stability. It is robust to outliers and works well on high-dimensional data."},
    {"id": "doc_004", "topic": "KNN", "text": "K-Nearest Neighbors classifies a data point based on the majority class among its K closest neighbors in feature space. It uses distance metrics like Euclidean distance. It is easy to understand but computationally slow for large datasets since it stores all training data. Choosing the right K is important."},
    {"id": "doc_005", "topic": "SVM", "text": "Support Vector Machine finds the optimal hyperplane that maximizes the margin between classes. It works well for high-dimensional data and uses kernel functions (linear, RBF, polynomial) for non-linear boundaries. It is effective when classes are clearly separable and works well on small to medium datasets."},
    {"id": "doc_006", "topic": "SHAP", "text": "SHAP (SHapley Additive exPlanations) explains model predictions using Shapley values from cooperative game theory. Each feature is assigned an importance value for a specific prediction. SHAP provides both global explanations (overall feature importance) and local explanations (why a specific prediction was made). It is model-agnostic and works with any ML model."},
    {"id": "doc_007", "topic": "LIME", "text": "LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the complex model locally with a simpler interpretable model. It perturbs the input data, observes changes in output, and fits a linear model around the prediction. Unlike SHAP, LIME is faster but less consistent across runs."},
    {"id": "doc_008", "topic": "Evaluation Metrics", "text": "Classification models are evaluated using accuracy (overall correctness), precision (how many predicted positives are correct), recall (how many actual positives are caught), F1-score (harmonic mean of precision and recall), and ROC-AUC (ability to distinguish classes). For imbalanced datasets, F1-score and ROC-AUC are more reliable than accuracy."},
    {"id": "doc_009", "topic": "Feature Engineering", "text": "Feature engineering transforms raw data into useful features for ML models. It includes handling missing values (imputation), encoding categorical variables (one-hot or label encoding), scaling numerical features (normalization or standardization), creating interaction features, and selecting the most relevant features using methods like correlation analysis or feature importance scores."},
    {"id": "doc_010", "topic": "Model Comparison", "text": "Comparing ML models requires a consistent evaluation pipeline: same train/test split, same cross-validation strategy, and same metrics. In the student depression prediction benchmark, 14 models were compared. XGBoost and Gradient Boosting achieved the highest performance. Key metrics used were accuracy, F1-score, and ROC-AUC across 5-fold cross-validation."}
]

# ================= BUILD FUNCTION (called once by Streamlit cache) =================
def build_app():
    """
    All heavy imports and initialization happen here.
    Called once via @st.cache_resource — never at module import time.
    """
    print("[agent] Loading LLM...")
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    print("[agent] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    print("[agent] Setting up ChromaDB...")
    import chromadb
    chroma_client = chromadb.Client()  # in-memory, no persistence issues
    collection = chroma_client.get_or_create_collection("ml_papers")

    if collection.count() == 0:
        print("[agent] Indexing documents...")
        texts = [d["text"] for d in DOCUMENTS]
        embeddings = embedder.encode(texts).tolist()  # assign to variable first
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=[d["id"] for d in DOCUMENTS],
            metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
        )
        print(f"[agent] Indexed {len(DOCUMENTS)} documents.")

    print("[agent] Compiling LangGraph...")
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    # ---- NODE FUNCTIONS ----

    def memory_node(state: AgentState) -> dict:
        msgs = state.get("messages", [])
        question = state["question"]
        user_name = state.get("user_name", "")

        if "my name is" in question.lower():
            parts = question.lower().split("my name is")
            if len(parts) > 1:
                user_name = parts[1].strip().split()[0].capitalize()

        msgs.append({"role": "user", "content": question})
        return {"messages": msgs[-6:], "user_name": user_name}

    def router_node(state: AgentState) -> dict:
        question = state["question"]
        prompt = f"""You are a routing agent. Choose exactly one route.

- "retrieve" → questions about ML concepts, algorithms, models, metrics, or research
- "tool"     → questions about current date or time
- "skip"     → greetings, thanks, or purely conversational messages

Respond with ONE word only: retrieve, tool, or skip.

Question: {question}"""

        response = llm.invoke(prompt)
        route = response.content.strip().lower().split()[0]
        if route not in ["retrieve", "tool", "skip"]:
            route = "retrieve"
        return {"route": route}

    def retrieval_node(state: AgentState) -> dict:
        question = state["question"]
        q_emb = embedder.encode([question]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)

        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]

        context = ""
        for topic, chunk in zip(topics, chunks):
            context += f"[{topic}]\n{chunk}\n\n"

        return {"retrieved": context, "sources": topics}

    def skip_node(state: AgentState) -> dict:
        return {"retrieved": "", "sources": [], "tool_result": ""}

    def tool_node(state: AgentState) -> dict:
        question = state["question"].lower()
        try:
            if any(w in question for w in ["date", "time", "today", "day", "year"]):
                now = datetime.now()
                result = f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M')}"
            else:
                result = "I can provide the current date and time. For other real-time data, please check directly."
        except Exception as e:
            result = f"Tool error: {str(e)}"
        return {"tool_result": result}

    def answer_node(state: AgentState) -> dict:
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        user_name = state.get("user_name", "")

        name_part = f"The user's name is {user_name}. " if user_name else ""
        retry_note = "\nPrevious answer was flagged. Be more conservative — only use what is explicitly in the context." if eval_retries > 0 else ""
        history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages[-4:]])

        context_section = ""
        if retrieved:
            context_section += f"\nKNOWLEDGE BASE:\n{retrieved}"
        if tool_result:
            context_section += f"\nTOOL RESULT:\n{tool_result}"

        prompt = f"""{name_part}You are an ML Research Assistant.

STRICT RULE: Answer ONLY using the provided context below.
If the answer is not in the context, say "I don't have that in my knowledge base."
Do NOT use general knowledge. Do NOT fabricate results or citations.{retry_note}

CHAT HISTORY:
{history}
{context_section}

QUESTION: {question}

ANSWER:"""

        response = llm.invoke(prompt)
        return {"answer": response.content.strip()}

    def eval_node(state: AgentState) -> dict:
        retrieved = state.get("retrieved", "")
        answer = state.get("answer", "")
        eval_retries = state.get("eval_retries", 0)

        if not retrieved:
            return {"faithfulness": 1.0, "eval_retries": eval_retries + 1}

        prompt = f"""Rate how faithfully this answer sticks to the provided context.
1.0 = answer uses ONLY information from context.
0.0 = answer contains hallucinated information not in context.

CONTEXT: {retrieved[:800]}
ANSWER: {answer}

Reply with a single decimal number between 0.0 and 1.0. Nothing else."""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip().split()[0])
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.8

        return {"faithfulness": score, "eval_retries": eval_retries + 1}

    def save_node(state: AgentState) -> dict:
        msgs = state.get("messages", [])
        msgs.append({"role": "assistant", "content": state["answer"]})
        return {"messages": msgs}

    # ---- ROUTING FUNCTIONS ----

    def route_decision(state: AgentState) -> str:
        return state.get("route", "retrieve")

    def eval_decision(state: AgentState) -> str:
        if state.get("eval_retries", 0) >= MAX_EVAL_RETRIES:
            return "save"
        if state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD:
            return "save"
        return "answer"

    # ---- GRAPH ASSEMBLY ----

    g = StateGraph(AgentState)

    g.add_node("memory",   memory_node)
    g.add_node("router",   router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("tool",     tool_node)
    g.add_node("skip",     skip_node)
    g.add_node("answer",   answer_node)
    g.add_node("eval",     eval_node)
    g.add_node("save",     save_node)

    g.set_entry_point("memory")

    g.add_edge("memory",   "router")
    g.add_edge("retrieve", "answer")
    g.add_edge("tool",     "answer")
    g.add_edge("skip",     "answer")
    g.add_edge("answer",   "eval")
    g.add_edge("save",     END)

    g.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "tool":     "tool",
        "skip":     "skip"
    })
    g.add_conditional_edges("eval", eval_decision, {
        "answer": "answer",
        "save":   "save"
    })

    compiled_app = g.compile(checkpointer=MemorySaver())
    print("[agent] Ready.")
    return compiled_app, embedder, collection


# ================= STANDALONE ASK (for notebook testing) =================
def ask(question, messages=None, thread_id="1"):
    """For testing in a notebook or script — not used by Streamlit directly."""
    compiled_app, _, _ = build_app()
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_app.invoke({
        "question": question,
        "messages": messages or [],
        "eval_retries": 0
    }, config)
    return result