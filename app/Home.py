import streamlit as st
from app.components import header, footer, sidebar
from utils.mask import mask_pii
from utils.ner import load_ner_model

def show():
    header()
    config = sidebar()
    
    st.header("📝 PII Masking Tool")
    st.write("Identify and mask Personally Identifiable Information in text")
    
    ner_model = load_ner_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_text = st.text_area("Input text containing PII:", height=300,
                                placeholder="Enter text with names, emails, phone numbers...")
        
    with col2:
        if st.button("🚀 Mask PII", use_container_width=True):
            if input_text:
                masked_text = mask_pii(input_text, ner_model, config['ner_threshold'])
                st.text_area("Masked text:", value=masked_text, height=300)
            else:
                st.warning("Please enter some text to mask")
    
    footer()