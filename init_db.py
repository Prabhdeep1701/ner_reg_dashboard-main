import sqlite3
import json
import os
import shutil
from sentence_transformers import SentenceTransformer

class DocumentStore:
    def __init__(self):
        self.db_path = "data/document_db.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._init_db()

    def _init_db(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            text TEXT,
            vector BLOB
        )
        """)
        self.conn.commit()

    def add_documents(self, documents):
        for doc in documents:
            if "vector" not in doc:
                doc["vector"] = self.embedder.encode(doc["text"]).tolist()
            self.cursor.execute("""
            INSERT INTO documents (id, text, vector)
            VALUES (?, ?, ?)
            """, (doc["id"], doc["text"], str(doc["vector"])))
        self.conn.commit()

    def search(self, query, k=5):
        # Implement search logic here
        pass

def main():
    # Remove existing database if it exists
    db_path = "data/document_db.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize document store
    doc_store = DocumentStore()

    # Load documents from JSON file
    with open("data/documents.json") as f:
        documents = json.load(f)

    # Add to database
    doc_store.add_documents(documents)
    print(f"Successfully added {len(documents)} documents to the database")

if __name__ == "__main__":
    main()