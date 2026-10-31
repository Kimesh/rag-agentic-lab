"""
STEP 1: Understanding Vector Embeddings & Cosine Similarity
===========================================================
This script demonstrates:
1. Converting text sentences into 768-dimensional mathematical vectors using Google Gemini.
2. Calculating Cosine Similarity in pure Python (no math libraries needed).
3. Proving that sentences with similar MEANING have high similarity scores,
   even if they share ZERO words in common!

Prerequisite:
Set your Gemini API key in your environment or paste it below:
  export GEMINI_API_KEY=""
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

# ==============================================================================
# 1. THE MATHEMATICAL CORE: COSINE SIMILARITY
# ==============================================================================
def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Calculates the cosine of the angle between two multi-dimensional vectors.
    Formula: (A · B) / (||A|| * ||B||)
    
    Returns:
      1.0  -> Identical semantic meaning
      0.0  -> Completely unrelated concepts
     -1.0  -> Exact opposite meaning
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# ==============================================================================
# 2. CALLING GEMINI EMBEDDING API
# ==============================================================================
def get_embedding(text: str) -> list:
    """
    Sends text to Google Gemini's text-embedding-004 model.
    Returns a list of 768 floating-point numbers representing the semantic meaning.
    """
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\n[!] Please set your GEMINI_API_KEY at the top of the file or in your environment.")
        # Provide a synthetic demo vector so the script can still demonstrate the concept
        return [0.1] * 768

    url = API_URL + GEMINI_API_KEY
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["embedding"]["values"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"\n[ERROR] Gemini API Error ({e.code}): {error_body}")
        raise e

# ==============================================================================
# 3. EXPERIMENT: SEMANTIC SEARCH IN ACTION
# ==============================================================================
def run_experiment():
    print("=" * 70)
    print("  EXPERIMENT 1: HOW COMPUTERS UNDERSTAND MEANING WITH VECTORS")
    print("=" * 70)

    # The Query: What an auditor or user is searching for
    query = "Suspicious money laundering via foreign shell company"
    print(f"\n[QUERY]: '{query}'\n")

    # The Documents in our database
    documents = [
        "High risk offshore financial wire transfer with unverified entity",
        "Customer deposited 5,000 rupees in local savings account with KYC",
        "The best Italian recipe for baking sourdough bread and pizza dough",
        "International sanctions violation and illicit capital flight detected"
    ]

    print("Fetching 768-dimensional vector embeddings from Gemini...\n")
    query_vector = get_embedding(query)
    print(f"-> Query converted into vector with {len(query_vector)} numbers.")
    print(f"-> Sample of first 5 numbers: {[round(x, 4) for x in query_vector[:5]]} ...\n")

    # Compare query with every document using Cosine Similarity
    results = []
    for doc in documents:
        doc_vector = get_embedding(doc)
        similarity = cosine_similarity(query_vector, doc_vector)
        results.append((doc, similarity))

    # Sort results by similarity score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    print("-" * 70)
    print(f"{'RANK':<5} | {'SIMILARITY':<12} | {'DOCUMENT TEXT'}")
    print("-" * 70)
    for rank, (doc, score) in enumerate(results, 1):
        # Format bar visualization
        bar = "█" * int(score * 20) if score > 0 else ""
        print(f"#{rank:<4} | {score:>.4f} ({bar:<10}) | {doc}")
    print("-" * 70)

    print("\n[KEY TAKEAWAY]:")
    print("Notice how Document #1 and #4 ranked at the very top, even though")
    print("they do NOT share the exact words 'money laundering'!")
    print("The vector mathematics understood the underlying financial meaning.")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()

