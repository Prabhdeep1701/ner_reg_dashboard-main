from typing import List, Dict

def mask_pii(text: str, ner_model, threshold: float = 0.85) -> str:
    entities = ner_model(text)
    masked_text = text
    
    # Process in reverse to avoid offset issues
    for entity in sorted(entities, key=lambda x: -x['start']):
        if entity['score'] >= threshold:
            start, end, label = entity['start'], entity['end'], entity['entity']
            masked_text = masked_text[:start] + f"[{label.upper()}]" + masked_text[end:]
    
    return masked_text