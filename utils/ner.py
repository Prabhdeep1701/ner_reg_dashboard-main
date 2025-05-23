from transformers import pipeline
from model.model_utils import load_model

def load_ner_model():
    tokenizer, model = load_model()
    if model:
        return pipeline("ner", model=model, tokenizer=tokenizer)
    return pipeline("ner", model="dslim/bert-base-NER")