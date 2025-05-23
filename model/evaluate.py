from transformers import pipeline
from model.model_utils import load_model
import pandas as pd

def evaluate_model(test_samples):
    tokenizer, model = load_model()
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    
    results = []
    for sample in test_samples:
        preds = ner_pipeline(sample['text'])
        results.append({
            'text': sample['text'],
            'true_entities': sample['entities'],
            'predicted_entities': preds
        })
    
    return pd.DataFrame(results)