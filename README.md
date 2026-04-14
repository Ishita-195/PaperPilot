# 🤖 ML Research Assistant

An AI-powered chatbot that answers machine learning questions using retrieval-augmented generation (RAG).

## 🚀 Features

* Ask questions about ML algorithms, metrics, and explainability
* Uses vector search (ChromaDB) for context retrieval
* Faithfulness evaluation to reduce hallucination
* Chat-based UI using Streamlit
* Supports tools (date/time queries)

## 🧠 Tech Stack

* Python
* Streamlit
* LangGraph
* LangChain + Groq API
* Sentence Transformers
* ChromaDB

## ⚡ How It Works

1. User asks a question
2. System retrieves relevant context from knowledge base
3. LLM generates answer using ONLY retrieved data
4. Answer is evaluated for faithfulness

## 🛠️ Installation

```bash
git clone <your-repo-link>
cd ML-Research-Assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app.py
```

## 📌 Example Questions

* What is XGBoost?
* What are evaluation metrics?
* Explain SHAP vs LIME

## 📷 Demo

(Add screenshots here later)

## 🎯 Future Improvements

* PDF upload support
* Persistent chat memory
* Deployment on cloud

---
