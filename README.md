# PaperPilot

**A hallucination-aware ML research assistant — RAG with faithfulness gating.**

PaperPilot answers machine learning questions by retrieving context from a vector knowledge base, generating responses grounded *only* in that context, and then evaluating every answer for faithfulness before it reaches you. The name is the architecture: it pilots you through ML concepts, and nothing ships without passing the faithfulness gate.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-1C3C3C)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B6B)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203-F55036)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why PaperPilot?

Most RAG chatbots retrieve context and hope for the best. PaperPilot adds a **verification layer**: after generation, each answer is scored against the retrieved sources. If the answer isn't supported by the knowledge base, you'll know — hallucinations get flagged instead of delivered.

## Features

- **Ask anything ML** — algorithms, evaluation metrics, explainability (SHAP vs LIME, and more)
- **Vector retrieval** — semantic search over the knowledge base using ChromaDB + Sentence Transformers
- **Faithfulness gating** — every answer is evaluated against its retrieved context to reduce hallucination
- **Agentic workflow** — LangGraph routes between retriever, generator, evaluator, and tool nodes
- **Built-in tools** — handles utility queries (date/time) alongside knowledge questions
- **Chat UI** — clean conversational interface built with Streamlit

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph + LangChain |
| LLM | Groq API (LLaMA 3) |
| Embeddings | Sentence Transformers |
| Vector Store | ChromaDB |
| Language | Python |

## How It Works

```
User question
     |
     v
+--------------+    +---------------+    +--------------------+
|  Retriever   |--> |  Generator    |--> |  Faithfulness Gate  |--> Answer
|  (ChromaDB)  |    |  (Groq LLaMA) |    |    (Evaluator)      |
+--------------+    +---------------+    +--------------------+
```

1. **Retrieve** — the question is embedded and matched against the vector knowledge base
2. **Generate** — the LLM answers using *only* the retrieved context (no free-floating generation)
3. **Verify** — the answer is evaluated for faithfulness to its sources before being returned

## Project Structure

```
PaperPilot/
├── app.py            # Streamlit chat interface
├── agent.py          # LangGraph agent: retriever -> generator -> evaluator
├── kb.py             # Knowledge base ingestion & ChromaDB setup
├── test.py           # Tests
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Ishita-195/PaperPilot.git
cd PaperPilot
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set your Groq API key:

```bash
# Windows
set GROQ_API_KEY=your_key_here
# macOS / Linux
export GROQ_API_KEY=your_key_here
```

## Run the App

```bash
streamlit run app.py
```

## Example Questions

- *What is XGBoost?*
- *What are common evaluation metrics for classification?*
- *Explain SHAP vs LIME.*

## Demo

*(Screenshots coming soon)*

## Roadmap

- [ ] PDF upload support — bring your own papers into the knowledge base
- [ ] Persistent chat memory across sessions
- [ ] Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
- [ ] Faithfulness score displayed in the UI per answer

## Author

**Ishita** — [GitHub](https://github.com/Ishita-195)

---

If PaperPilot helped you, consider starring the repo.
