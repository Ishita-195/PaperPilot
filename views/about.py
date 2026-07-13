"""About page — how PaperPilot works, design decisions, and tech stack."""
import streamlit as st

from agent import FAITHFULNESS_THRESHOLD, MAX_EVAL_RETRIES
from backend.kb import DOCUMENTS

st.title("How PaperPilot works")
st.caption("Every answer passes a faithfulness gate before you see it.")

st.markdown(
    """
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:8px 0 4px;">
  <div style="background:#EDF5F0;border:1px solid #C5D8CE;border-radius:8px;padding:10px 14px;text-align:center;">
    <div style="font-size:14px;color:#0F6E56;font-weight:600;">retrieve</div>
    <div style="font-size:11px;color:#4A6357;">ChromaDB</div>
  </div>
  <div style="color:#7FB8A4;font-size:18px;">&rarr;</div>
  <div style="background:#EDF5F0;border:1px solid #C5D8CE;border-radius:8px;padding:10px 14px;text-align:center;">
    <div style="font-size:14px;color:#0F6E56;font-weight:600;">answer</div>
    <div style="font-size:11px;color:#4A6357;">Llama 3.3 70B</div>
  </div>
  <div style="color:#7FB8A4;font-size:18px;">&rarr;</div>
  <div style="background:#0F6E56;border-radius:8px;padding:10px 14px;text-align:center;">
    <div style="font-size:14px;color:#FFFFFF;font-weight:600;">judge</div>
    <div style="font-size:11px;color:#C5E5D8;">LLM-as-judge</div>
  </div>
  <div style="color:#7FB8A4;font-size:18px;">&rarr;</div>
  <div style="background:#EDF5F0;border:1px solid #C5D8CE;border-radius:8px;padding:10px 14px;text-align:center;">
    <div style="font-size:14px;color:#0F6E56;font-weight:600;">respond</div>
    <div style="font-size:11px;color:#4A6357;">with score</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    f"↺ If the judge scores below {FAITHFULNESS_THRESHOLD:.1f}, the answer is "
    f"regenerated with stricter grounding (up to {MAX_EVAL_RETRIES} retries)."
)

st.markdown("### The pipeline, step by step")
st.markdown(
    f"""
1. **Retrieve** — your question is embedded (`paraphrase-MiniLM-L3-v2`) and matched
   against {len(DOCUMENTS)} curated ML topics plus any papers you've uploaded or
   fetched from arXiv, stored in an in-memory ChromaDB index.
2. **Answer** — Llama 3.3 70B (via Groq) answers using *only* the retrieved
   context. If the context doesn't cover it, the assistant says so instead of guessing.
3. **Judge** — a second LLM pass scores how faithful the answer is to its
   sources, from 0.0 to 1.0.
4. **Retry or respond** — low scores trigger a regeneration that is told exactly
   why its previous attempt failed; passing answers are delivered with their
   score and sources.
"""
)

st.markdown("### Design decisions")
st.markdown(
    """
- **Groq for inference** — the faithfulness judge adds an extra LLM call per
  answer, so sub-second 70B inference keeps the app feeling instant.
- **MiniLM-L3 embeddings** — small enough for free-tier cold starts while
  staying accurate for short technical passages.
- **Ephemeral ChromaDB** — the index rebuilds in seconds on boot, which keeps
  deploys stateless and reproducible.
- **LLM-as-judge over string metrics** — n-gram overlap can't detect a fluent
  hallucination; a judge model can.
"""
)

st.markdown("### Evaluation targets")
st.markdown(
    """
| Metric | Meaning | Target |
|---|---|---|
| Faithfulness | Answer is grounded in retrieved context | ≥ 0.70 |
| Answer relevancy | Answer addresses the question | ≥ 0.75 |
| Context precision | Retrieved context is on-topic | ≥ 0.70 |
"""
)
st.caption("Run the suite yourself on the **Evaluation** page.")

st.markdown("### Tech stack")
st.markdown(
    """
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">
  <span style="background:#EDF5F0;color:#0F6E56;font-size:13px;padding:4px 12px;border-radius:12px;">LangGraph</span>
  <span style="background:#EDF5F0;color:#0F6E56;font-size:13px;padding:4px 12px;border-radius:12px;">LangChain</span>
  <span style="background:#EDF5F0;color:#0F6E56;font-size:13px;padding:4px 12px;border-radius:12px;">ChromaDB</span>
  <span style="background:#EDF5F0;color:#0F6E56;font-size:13px;padding:4px 12px;border-radius:12px;">Groq · Llama 3.3 70B</span>
  <span style="background:#EDF5F0;color:#0F6E56;font-size:13px;padding:4px 12px;border-radius:12px;">Sentence Transformers</span>
  <span style="background:#EDF5F0;color:#0F6E56;font-size:13px;padding:4px 12px;border-radius:12px;">Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")
st.link_button("View source on GitHub", "https://github.com/Ishita-195/PaperPilot")
