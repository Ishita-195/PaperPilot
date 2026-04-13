import streamlit as st
import uuid

st.set_page_config(
    page_title="ML Research Assistant",
    page_icon="🤖",
    layout="centered"
)

# ================= LOAD ONCE =================
# @st.cache_resource runs build_app() exactly once per session.
# All heavy loading (LLM, embedder, ChromaDB, graph compile) happens inside build_app().
# Streamlit will NOT re-run this on every message or every rerun.

@st.cache_resource
def load_agent():
    from agent import build_app
    return build_app()

with st.spinner("Loading agent... (first load takes ~30 seconds)"):
    app, embedder, collection = load_agent()

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ================= SIDEBAR =================
with st.sidebar:
    st.title("ML Research Assistant")
    st.markdown("Ask questions about ML algorithms, explainability, and model evaluation.")
    st.markdown("---")
    st.markdown("**Topics covered:**")
    st.markdown("""
- XGBoost & Gradient Boosting  
- Random Forest, KNN, SVM  
- SHAP & LIME explainability  
- Evaluation metrics  
- Feature engineering  
- Model comparison
""")
    st.markdown("---")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

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