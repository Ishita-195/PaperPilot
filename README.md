# 🤖 ML Research Assistant

**ML Research Assistant** is an intelligent AI-powered chatbot that answers questions about machine learning concepts, algorithms, and research — grounded strictly in a curated knowledge base to eliminate hallucinations.

Built with **LangGraph**, **Retrieval-Augmented Generation (RAG)**, and a **faithfulness self-evaluation loop**, it combines structured vector search with dynamic reasoning to deliver accurate, context-aware responses.

---

## 🚀 Features

* 📚 **Knowledge Base (RAG)** — 26 curated ML documents indexed in ChromaDB covering algorithms, deep learning, explainability, evaluation, and more
* 🔀 **Intelligent Query Router** — automatically routes each question to the right pipeline: Knowledge Base retrieval, Tool use, or conversational memory
* 🧠 **Conversation Memory** — retains recent chat history via LangGraph's `MemorySaver` for multi-turn coherent dialogue
* ✅ **Faithfulness Evaluation Loop** — scores every answer against retrieved context (threshold: 0.7); retries generation if the answer drifts from the source
* 📄 **PDF Upload Support** — upload your own research papers and they are dynamically embedded and searchable alongside the built-in KB
* 🕐 **Tool Node** — answers real-time queries like current date/time
* 💬 **Interactive Chat UI** — clean Streamlit interface with live faithfulness score display and source citations per answer

---

## 🧠 How It Works

```
User Question
      │
      ▼
 Memory Node  ──► extracts user name, appends to history
      │
      ▼
 Router Node  ──► "retrieve" | "tool" | "skip"
      │
   ┌──┴──────────────┐
   ▼                  ▼
Retrieval Node     Tool Node
(ChromaDB vector   (date/time)
  search, top 3)
   │                  │
   └──────┬───────────┘
          ▼
     Answer Node  ──► LLM generates answer strictly from context
          │
          ▼
      Eval Node  ──► faithfulness score [0.0 – 1.0]
          │
     ┌────┴────┐
     │ < 0.7?  │──► retry Answer Node (max 2 retries)
     └────┬────┘
          ▼
      Save Node  ──► appends to message history → END
```

---

## 📖 Knowledge Base Topics

The built-in knowledge base covers **26 ML topics** across five categories:

| Category | Topics |
|---|---|
| **Core Algorithms** | XGBoost, Gradient Boosting, Random Forest, KNN, SVM, Decision Trees |
| **Deep Learning** | Neural Networks, CNN, RNN, Transformers |
| **Explainability** | SHAP, LIME |
| **Data Processing** | Feature Engineering, Data Preprocessing, Dimensionality Reduction |
| **Evaluation & Concepts** | Evaluation Metrics, Cross Validation, Overfitting, Underfitting, Bias-Variance Tradeoff, Regularization, Hyperparameter Tuning, Ensemble Learning, Clustering, Anomaly Detection, Model Comparison |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Groq (Llama 3.3 70B Versatile) |
| **Agent Framework** | LangGraph |
| **Vector Database** | ChromaDB (in-memory) |
| **Embeddings** | SentenceTransformers (`paraphrase-MiniLM-L3-v2`) |
| **Frontend** | Streamlit |
| **PDF Parsing** | pypdf |
| **Memory** | LangGraph `MemorySaver` |

---

## ⚙️ Installation

**1️⃣ Clone the repository**
```bash
git clone https://github.com/Ishita-195/ML-Research-Assistant.git
cd ML-Research-Assistant
```

**2️⃣ Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

**4️⃣ Set up environment variables**

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key at [https://console.groq.com](https://console.groq.com)

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open: **http://localhost:8501**

> ⚠️ First load takes ~30 seconds while the embedding model and ChromaDB collection initialise.

---

## 🎯 Example Questions

```
What is XGBoost and how does it differ from Gradient Boosting?
Explain the bias-variance tradeoff.
What is SHAP and how does it differ from LIME?
How do I evaluate a model on an imbalanced dataset?
What are the key hyperparameters in a Random Forest?
What is the difference between overfitting and underfitting?
How does cross-validation work?
Explain how Transformers use self-attention.
What is anomaly detection used for?
```

---

## 📂 Project Structure

```
ML-Research-Assistant/
├── app.py            # Streamlit frontend — chat UI, PDF upload, agent invocation
├── agent.py          # LangGraph agent — nodes, graph assembly, faithfulness eval
├── kb.py             # Knowledge base — 26 ML topic documents + ChromaDB setup
├── test.py           # Standalone testing scripts
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📄 PDF Upload

You can upload your own ML research papers directly from the sidebar. Uploaded PDFs are:
- Parsed with `pypdf` (first 2000 characters per paper)
- Embedded and added to the ChromaDB collection at runtime
- Searchable alongside the built-in knowledge base for that session

---

## 👩‍💻 Author

**Ishita** — BTech CSE | ML & AI Enthusiast

---

## ⭐ Support

If you found this project useful, give it a star ⭐ and feel free to open issues or contribute!

---

## 📜 License

This project is licensed under the **MIT License**.
