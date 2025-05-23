import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class DocumentStore:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        self.db_path = "data/document_db.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT,
                vector BLOB
            )
        ''')
        self.conn.commit()

    def add_documents(self, documents: List[Dict]):
        cursor = self.conn.cursor()
        for doc in documents:
            # Generate embedding if vector doesn't exist or has wrong dimensions
            if 'vector' not in doc or len(doc['vector']) != self.embedding_dim:
                vector = self.embedder.encode(doc['text'], normalize_embeddings=True)
            else:
                vector = np.array(doc['vector'])
            
            # Convert vector to bytes before insertion
            vector_bytes = vector.tobytes()
            cursor.execute('''
                INSERT OR REPLACE INTO documents (id, text, vector)
                VALUES (?, ?, ?)
            ''', (doc['id'], doc['text'], vector_bytes))
        self.conn.commit()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, text, vector FROM documents')
        results = []
        for row in cursor.fetchall():
            try:
                vector_data = row[2]
                # Handle both string and bytes vector data
                if isinstance(vector_data, str):
                    # If it's a string representation of a list, evaluate it
                    import ast
                    vector_list = ast.literal_eval(vector_data)
                    vector = np.array(vector_list, dtype=np.float32)
                else:
                    # If it's bytes, convert normally
                    vector = np.frombuffer(vector_data, dtype=np.float32)
                
                # Ensure vector has correct dimensions
                if len(vector) == self.embedding_dim:
                    similarity = np.dot(query_embedding, vector)
                    results.append({'id': row[0], 'text': row[1], 'score': similarity})
            except Exception as e:
                print(f"Error processing vector for document {row[0]}: {e}")
                continue
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]