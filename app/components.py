import streamlit as st

def header():
    st.title("NER + RAG Dashboard")
    st.markdown("---")

def sidebar():
    st.sidebar.header("Configuration")
    return {
        "ner_threshold": st.sidebar.slider("NER Confidence Threshold", 0.5, 1.0, 0.85),
        "top_k": st.sidebar.slider("RAG Top-K Documents", 1, 10, 3)
    }

def footer():
    st.markdown("---")
    st.caption("© 2023 NER RAG Dashboard")