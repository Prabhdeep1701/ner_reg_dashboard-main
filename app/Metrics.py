import streamlit as st
import json
import pandas as pd
import plotly.express as px
from app.components import header, footer

def load_metrics():
    try:
        with open("data/training_results.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def show():
    header()
    st.header("📊 Training Metrics")
    
    metrics = load_metrics()
    
    if metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            df = pd.DataFrame(metrics['accuracy'])
            fig = px.line(df, x="epoch", y="value", 
                         title="Accuracy Over Epochs", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            df = pd.DataFrame(metrics['loss'])
            fig = px.line(df, x="epoch", y="value", 
                         title="Loss Over Epochs", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        st.json(metrics, expanded=False)
    else:
        st.warning("No training metrics found. Please train the model first.")
    
    footer() 