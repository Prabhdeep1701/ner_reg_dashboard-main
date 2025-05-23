import os
from transformers import AutoTokenizer, AutoModelForTokenClassification

def save_model(trainer, tokenizer):
    os.makedirs("./model/trained_model", exist_ok=True)
    trainer.save_model("./model/trained_model/ner_model.bin")
    tokenizer.save_pretrained("./model/trained_model/")

def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./model/trained_model/")
        model = AutoModelForTokenClassification.from_pretrained("./model/trained_model/ner_model.bin")
        return tokenizer, model
    except:
        return None, None