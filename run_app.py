import streamlit as st
from app import Home, Chatbot, Metrics

def main():
    st.set_page_config(page_title="NER+RAG Dashboard", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Chatbot", "Metrics"])

    if page == "Home":
        Home.show()
    elif page == "Chatbot":
        Chatbot.show()
    elif page == "Metrics":
        Metrics.show()

if __name__ == "__main__":
    main()