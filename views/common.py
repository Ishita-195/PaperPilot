"""Shared helpers for the Streamlit pages."""
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_agent(extra_docs):
    from agent import build_app

    return build_app(extra_docs)
