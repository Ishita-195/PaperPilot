import streamlit as st
import uuid
from pypdf import PdfReader

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

st.set_page_config(
    page_title="ML Research Assistant",
    page_icon="🤖",
    layout="centered"
)

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ================= SIDEBAR =================
# ================= SIDEBAR =================
with st.sidebar:
    st.title("ML Research Assistant")
    st.markdown("Ask questions about ML algorithms, explainability, and evaluation.")
    st.markdown("---")

    st.markdown("**Topics Covered:**")

    st.markdown("""
- XGBoost, Gradient Boosting  
- Random Forest, Decision Trees  
- KNN, SVM  
- Neural Networks, RNN, Transformers  
- SHAP, LIME (Explainability)  
- Evaluation (Accuracy, F1, ROC)  
- Feature Engineering, Dimensionality Reduction  
- Model Comparison  
""")

    st.markdown("---")
    # ================= PDF UPLOAD =================
    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
    st.markdown("---")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ================= EXTRACT PDFs EARLY =================
pdf_docs = []
if uploaded_files:
    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        pdf_docs.append({
            "id": f"pdf_{i}",
            "topic": file.name,
            "text": text[:2000]  # limit to first 2000 chars
        })
    st.sidebar.success(f"{len(pdf_docs)} PDF(s) loaded")

# ================= LOAD AGENT WITH PDFs =================
@st.cache_resource
def load_agent(extra_docs):
    from agent import build_app
    return build_app(extra_docs)

with st.spinner("Loading agent... (first load takes ~30 seconds)"):
    app, embedder, collection = load_agent(pdf_docs)  # ✅ PASS PDFs HERE

# ================= MAIN UI =================
st.title("ML Research Assistant")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask your ML question..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = app.invoke(
                    {
                        "question": prompt,
                        "messages": [],
                        "eval_retries": 0
                    },
                    config
                )
                answer = result.get("answer", "Sorry, I could not generate a response.")
                sources = result.get("sources", [])
                faithfulness = result.get("faithfulness", None)

            except Exception as e:
                answer = f"Something went wrong: {str(e)}"
                sources = []
                faithfulness = None

        st.markdown(answer)

        if sources:
            st.markdown("---")
            st.markdown("**Sources used:**")
            for s in sources:
                st.markdown(f"- {s}")

        if faithfulness is not None:
            color = "green" if faithfulness >= 0.7 else "orange"
            st.caption(f":{color}[Faithfulness score: {faithfulness:.2f}]")

    st.session_state.messages.append({"role": "assistant", "content": answer})