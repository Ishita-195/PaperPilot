import streamlit as st

st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="centered")

pages = st.navigation(
    [
        st.Page("views/chat.py", title="Chat", icon="💬", default=True),
        st.Page("views/evaluation.py", title="Evaluation", icon="📊"),
        st.Page("views/about.py", title="About", icon="📖"),
    ]
)
pages.run()
