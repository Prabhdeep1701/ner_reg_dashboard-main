from typing import List

def augment_with_ner(query: str, documents: List[dict], ner_model, threshold: float = 0.85) -> str:
    # Simple augmentation - in practice you'd want more sophisticated logic
    response = f"Found {len(documents)} relevant documents:\n\n"
    
    for doc in documents:
        # Handle both 'text' and 'content' keys for compatibility
        text_content = doc.get('text', doc.get('content', ''))
        entities = ner_model(text_content)
        pii_counts = {}
        
        for ent in entities:
            if ent['score'] >= threshold:
                pii_counts[ent['entity']] = pii_counts.get(ent['entity'], 0) + 1
        
        response += f"- Document (score: {doc.get('score', 0.0):.2f}) contains: "
        if pii_counts:
            response += ", ".join([f"{count} {label}" for label, count in pii_counts.items()])
        else:
            response += "no detected PII"
        response += "\n"
    
    return response