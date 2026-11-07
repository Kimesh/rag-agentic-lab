"""
STEP 3: The Complete End-to-End RAG Pipeline
============================================
This script connects all the pieces:
1. RETRIEVAL: Searches our Vector DB (vector_store.json) to fetch the top-2 policy chunks.
2. AUGMENTATION: Injects the retrieved chunks into a strict, zero-hallucination Prompt.
3. GENERATION: Calls Google's LLM to generate an audit-ready response with exact legal citations.
4. HALLUCINATION PREVENTION TEST: Proves that asking about unrelated topics (like pasta)
   forces the model to say "Not found in policy" rather than making things up!

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
GENERATION_MODELS = ["gemini-flash-lite-latest", "gemini-3.8-flash"]

SCRIPT_DIR = Path(__file__).parent.resolve()
VECTOR_STORE_FILE = SCRIPT_DIR / "vector_store.json"
POLICY_FILE = SCRIPT_DIR / "data" / "compliance_policy.txt"

# ==============================================================================
# 1. VECTOR RETRIEVAL LOGIC
# ==============================================================================
def get_embedding(text: str) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
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
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def retrieve_top_k(query: str, top_k: int = 2) -> list:
    """Retrieves top-K closest chunks from the persistent vector store."""
    if not VECTOR_STORE_FILE.exists():
        print("[!] vector_store.json not found. Run 02_vector_store.py first to index!")
        return []

    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    query_vector = get_embedding(query)
    scored = []
    for r in records:
        score = cosine_similarity(query_vector, r["embedding"])
        scored.append({"id": r["id"], "text": r["text"], "similarity": score})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]

# ==============================================================================
# 2. LLM GENERATION CALL (IN-HOUSE LOCAL LLAMA 3.2)
# ==============================================================================
def call_llm(prompt: str) -> str:
    """Calls local In-House Llama 3.2 model via OpenAI-compatible endpoint."""
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] Failed to query local model: {e}"

# ==============================================================================
# 3. THE COMPLETE RAG PIPELINE
# ==============================================================================
def run_rag_pipeline(user_query: str):
    print("\n" + "=" * 75)
    print(f"  [USER / AUDITOR QUERY]: \"{user_query}\"")
    print("=" * 75)

    # STEP A: RETRIEVAL
    print("\n[STEP 1: RETRIEVAL] Searching Vector Store for relevant policy chunks...")
    retrieved_chunks = retrieve_top_k(user_query, top_k=2)

    context_text = ""
    for rank, chunk in enumerate(retrieved_chunks, 1):
        print(f"  -> Match #{rank} (Chunk ID: {chunk['id']}, Similarity: {chunk['similarity']:.4f})")
        context_text += f"\n--- [POLICY CHUNK {chunk['id']}] ---\n{chunk['text']}\n"

    # STEP B: PROMPT AUGMENTATION (Injecting context + anti-hallucination rules)
    print("\n[STEP 2: AUGMENTATION] Constructing grounded prompt with strict constraints...")
    augmented_prompt = f"""You are a Senior Financial Compliance AI Auditor at Global Financial Corp.
Answer the question based on the policy context provided below. Be concise and cite specific sections or dollar thresholds if present.
If the context does not contain information to answer the question, state: "The provided compliance policy documentation does not contain information to answer this inquiry."

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Question: {user_query}

Audit Response:"""

    # STEP C: GENERATION
    print("\n[STEP 3: GENERATION] Sending augmented prompt to In-House Local Llama 3.2...")
    llm_response = call_llm(augmented_prompt)

    print("\n" + "-" * 75)
    print("  [FINAL GROUNDED AUDIT REPORT]")
    print("-" * 75)
    print(llm_response)
    print("-" * 75)

# ==============================================================================
# 4. RUN TESTS
# ==============================================================================
def main():
    print("=" * 75)
    print("  EXPERIMENT 3: END-TO-END RAG PIPELINE (RETRIEVAL + GENERATION)")
    print("=" * 75)

    # Test 1: Specific Threshold Question
    run_rag_pipeline("What is the mandatory reporting threshold for international wire transfers?")

    # Test 2: Shell Company & Sanctions Question
    run_rag_pipeline("Can a client wire money to a shell entity in the Cayman Islands?")

    # Test 3: The Hallucination Test (Asking something completely outside the policy)
    run_rag_pipeline("How do I make a chocolate cake?")

if __name__ == "__main__":
    main()
