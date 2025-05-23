import streamlit as st
import google.generativeai as genai
from app.components import header, footer, sidebar
from rag.document_store import DocumentStore
from rag.retriever import retrieve_documents
from rag.ner_augmenter import augment_with_ner
from utils.ner import load_ner_model

def generate_gemini_response(prompt: str, docs: list) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    if not docs:
        return "I couldn't find any relevant documents to answer your question. Please try a different query."
    
    # Create a more structured context with document metadata
    context_parts = []
    for i, doc in enumerate(docs, 1):
        text_content = doc.get("text", doc.get("content", ""))
        doc_id = doc.get("id", f"doc_{i}")
        score = doc.get("score", 0.0)
        
        context_parts.append(f"Document {i} (ID: {doc_id}, Relevance: {score:.2f}):\n{text_content}")
    
    context = "\n\n" + "="*50 + "\n\n".join(context_parts)
    
    # Enhanced prompt for better document-based responses
    full_prompt = f"""You are an AI assistant that answers questions based ONLY on the provided documents. 

IMPORTANT INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the information in the provided documents
2. If the documents don't contain enough information to answer the question, say so clearly
3. Reference specific documents when possible (e.g., "According to Document 1...")
4. Be concise but comprehensive
5. If you find contradictory information across documents, mention it

DOCUMENTS:
{context}

QUESTION: {prompt}

ANSWER (based only on the provided documents):"""
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def show():
    header()
    config = sidebar()
    
    st.header("💬 NER-Augmented Chatbot")
    st.write("Ask questions about your documents with enhanced PII awareness")
    
    # Initialize components
    ner_model = load_ner_model()
    doc_store = DocumentStore()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Retrieve and augment with NER
        docs = retrieve_documents(prompt, doc_store, top_k=config['top_k'])
        print("DEBUG: Retrieved docs:", docs)
        augmented_response = augment_with_ner(prompt, docs, ner_model, config['ner_threshold'])
        
        # Pass both prompt and docs to Gemini for a data-aware answer
        gemini_response = generate_gemini_response(prompt, docs)
        
        final_response = f"{gemini_response}\n\nNER-Augmented Answer:\n{augmented_response}"
        
        with st.chat_message("assistant"):
            st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
    
    footer()