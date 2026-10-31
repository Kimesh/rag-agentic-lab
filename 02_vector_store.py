"""
STEP 2: Document Chunking & Persistent Vector Database
======================================================
This script demonstrates:
1. Slicing a real compliance policy document into overlapping Chunks.
2. Embedding those chunks ONCE and persisting them to disk (vector_store.json).
3. Querying the Vector Store to retrieve the EXACT relevant policy clause
   in milliseconds without re-embedding the whole document.

Zero external dependencies required (standard Python only).
"""

import os
import json
import math
import urllib.request
import urllib.error
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError(
        "GEMINI_API_KEY is not set. Please set the environment variable or add it to .env"
    )

GEMINI_API_KEY = get_api_key()
EMBEDDING_MODEL = "gemini-embedding-001"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key="

SCRIPT_DIR = Path(__file__).parent.resolve()
POLICY_FILE = SCRIPT_DIR / "data" / "compliance_policy.txt"
VECTOR_STORE_FILE = SCRIPT_DIR / "vector_store.json"

# ==============================================================================
# 1. EMBEDDINGS & MATH HELPERS
# ==============================================================================
def get_embedding(text: str) -> list:
    """Calls Gemini API to get a 3,072-dimensional vector."""
    url = API_URL + GEMINI_API_KEY
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]}
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["embedding"]["values"]

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Computes cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# ==============================================================================
# 2. CHUNKING STRATEGY (SLIDING WINDOW OVERLAP)
# ==============================================================================
def chunk_document(text: str, chunk_size_words: int = 60, overlap_words: int = 15) -> list:
    """
    Slices a long document into overlapping chunks of words.
    
    Why overlap?
    If an important sentence is cut right at word 60, the overlap ensures
    the sentence context is preserved in the next chunk!
    """
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size_words - overlap_words

    while start < len(words):
        end = start + chunk_size_words
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": chunk_text,
            "word_count": len(words[start:end]),
            "start_word_idx": start,
            "end_word_idx": min(end, len(words))
        })
        start += step
        if end >= len(words):
            break

    return chunks

# ==============================================================================
# 3. THE VECTOR STORE CLASS (IN-MEMORY + DISK PERSISTENCE)
# ==============================================================================
class SimpleVectorStore:
    def __init__(self):
        # List of records: [{"id": 1, "text": "...", "embedding": [...]}]
        self.records = []

    def add_document(self, text: str, chunk_size_words: int = 60, overlap_words: int = 15):
        """Chunks a document and generates embeddings for each chunk."""
        chunks = chunk_document(text, chunk_size_words, overlap_words)
        print(f"-> Sliced document into {len(chunks)} overlapping chunks.")

        for i, chunk in enumerate(chunks, 1):
            print(f"   [Embedding Chunk {i}/{len(chunks)}] ({chunk['word_count']} words)...")
            vector = get_embedding(chunk["text"])
            self.records.append({
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": vector,
                "metadata": {
                    "word_count": chunk["word_count"],
                    "start_word_idx": chunk["start_word_idx"]
                }
            })

    def save_to_disk(self, filepath: Path):
        """Persists the vectors and text to disk (JSON)."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.records, f)
        print(f"[OK] Saved {len(self.records)} vector records to: {filepath.name}")

    def load_from_disk(self, filepath: Path):
        """Loads pre-computed vectors from disk. Zero API calls needed!"""
        with open(filepath, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        print(f"[OK] Loaded {len(self.records)} pre-computed vector records from: {filepath.name}")

    def search(self, query: str, top_k: int = 2) -> list:
        """
        Embeds the query (1 API call) and finds the top-K closest document chunks.
        """
        query_vector = get_embedding(query)
        scored_results = []

        for record in self.records:
            score = cosine_similarity(query_vector, record["embedding"])
            scored_results.append({
                "id": record["id"],
                "text": record["text"],
                "similarity": score,
                "metadata": record.get("metadata", {})
            })

        # Sort descending by similarity
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_results[:top_k]

# ==============================================================================
# 4. RUNNING THE EXPERIMENT
# ==============================================================================
def run():
    print("=" * 70)
    print("  EXPERIMENT 2: DOCUMENT CHUNKING & VECTOR DATABASE STORAGE")
    print("=" * 70)

    store = SimpleVectorStore()

    # Step A: Check if we already have the vectors on disk
    if VECTOR_STORE_FILE.exists():
        print("\nFound existing vector database on disk!")
        store.load_from_disk(VECTOR_STORE_FILE)
    else:
        print("\nVector database not found. Reading and indexing document...")
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            document_content = f.read()

        store.add_document(document_content, chunk_size_words=65, overlap_words=15)
        store.save_to_disk(VECTOR_STORE_FILE)

    print("\n" + "=" * 70)
    print("  SIMULATING USER / AUDITOR QUERIES (SEMANTIC RETRIEVAL)")
    print("=" * 70)

    # Let's test with two completely different real-world questions
    test_queries = [
        "What is the reporting limit for international fund transfers?",
        "What happens if an account makes rapid transactions from a strange location?"
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n--- [SEARCH {q_idx}]: \"{query}\" ---")
        matches = store.search(query, top_k=2)

        for rank, match in enumerate(matches, 1):
            print(f"\n  Top Match #{rank} (Similarity Score: {match['similarity']:.4f})")
            print(f"  Chunk ID: {match['id']}")
            print(f"  Retrieved Text:")
            print(f"  \"{match['text']}\"")
        print("-" * 70)

if __name__ == "__main__":
    run()
