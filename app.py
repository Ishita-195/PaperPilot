import traceback
import uuid

import streamlit as st
from pypdf import PdfReader

APP_NAME = "PaperPilot"
CHUNK_WORDS = 200
CHUNK_OVERLAP = 40
MAX_CHUNKS_PER_PDF = 60

st.set_page_config(page_title=APP_NAME, page_icon="📄", layout="centered")


# ================= PDF HELPERS =================
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunk = " ".join(words[start : start + size])
        if chunk.strip():
            chunks.append(chunk)
        if len(chunks) >= MAX_CHUNKS_PER_PDF:
            break
    return chunks


# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ================= SIDEBAR =================
with st.sidebar:
    st.title(f"📄 {APP_NAME}")
    st.markdown(
        "An ML research assistant that answers only from its knowledge base "
        "and **verifies every answer for faithfulness** before showing it."
    )
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ================= EXTRACT & CHUNK PDFs =================
pdf_docs = []
if uploaded_files:
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        for i, chunk in enumerate(chunk_text(text)):
            pdf_docs.append(
                {
                    "id": f"{file.name}_{i}",
                    "topic": f"{file.name} (part {i + 1})",
                    "text": chunk,
                }
            )
    st.sidebar.success(f"{len(uploaded_files)} PDF(s) indexed ({len(pdf_docs)} chunks)")


# ================= LOAD AGENT =================
@st.cache_resource(show_spinner=False)
def load_agent(extra_docs):
    from agent import build_app

    return build_app(extra_docs)


with st.spinner("Loading agent... (first load takes ~30 seconds)"):
    app, embedder, collection = load_agent(pdf_docs)

# ================= MAIN UI =================
st.title(APP_NAME)
st.caption("Grounded answers about ML — with a faithfulness check on every response.")

EXAMPLE_QUESTIONS = [
    "How does XGBoost differ from Random Forest?",
    "When should I use SHAP instead of LIME?",
    "What is the bias-variance tradeoff?",
]

# empty state: suggested prompts
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

# chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# new input (typed, or from an example button)
prompt = st.chat_input("Ask your ML question...")
if not prompt:
    prompt = st.session_state.pop("pending_question", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, answering, and fact-checking..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                # NOTE: do not pass "messages" here — the checkpointer keeps
                # conversation history per thread_id, and passing a value
                # would overwrite it.
                result = app.invoke(
                    {"question": prompt, "eval_retries": 0},
                    config,
                )
                answer = result.get("answer", "Sorry, I could not generate a response.")
                sources = result.get("sources", [])
                faithfulness = result.get("faithfulness")
            except Exception:
                traceback.print_exc()  # full details go to the server logs only
                answer = (
                    "Sorry — something went wrong while generating the answer. "
                    "Please try again in a moment."
                )
                sources = []
                faithfulness = None

        st.markdown(answer)

        if sources:
            st.markdown("---")
            st.markdown("**Sources used:**")
            for s in dict.fromkeys(sources):  # dedupe, keep order
                st.markdown(f"- {s}")

        if faithfulness is not None:
            color = "green" if faithfulness >= 0.7 else "orange"
            st.caption(f":{color}[Faithfulness score: {faithfulness:.2f}]")

    st.session_state.messages.append({"role": "assistant", "content": answer})
