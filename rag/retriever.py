from rag.document_store import DocumentStore
from typing import List, Dict, Any

def retrieve_documents(query: str, 
                      doc_store: DocumentStore, 
                      top_k: int = 3,
                      return_raw: bool = False) -> List[Dict[str, Any]]:
    """
    Retrieve documents from the document store based on a query.
    
    Args:
        query: The search query string
        doc_store: DocumentStore instance to search against
        top_k: Number of documents to return
        return_raw: If True, returns the raw document objects without processing
    
    Returns:
        List of document dictionaries. Format depends on return_raw parameter.
    """
    try:
        results = doc_store.search(query, top_k=top_k)
        
        if return_raw:
            # Return the raw documents as they are
            return [dict(hit) if hasattr(hit, '__dict__') else hit for hit in results]
        else:
            # Return a standardized format with common fields
            return [
                {
                    "content": hit.get("text", hit.get("content", str(hit))),
                    "score": float(hit.get("score", 0.0)),
                    "id": str(hit.get("id", "")),
                    "metadata": hit.get("metadata", {}),
                    "raw": hit  # Include the original document as well
                }
                for hit in results
            ]
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return []