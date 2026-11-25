"""
STEP 8: Data Ingestion & RAG Indexing with LlamaIndex
=====================================================
Demonstrates why teams use LlamaIndex alongside (or inside) agent frameworks:
  1. Connectors (`SimpleDirectoryReader`): Automatically loads files (TXT, PDF, DOCX, etc.)
     from a directory without writing custom file parsers.
  2. Global Settings & Chunking (`SentenceSplitter`): Configures the LLM, embedding model,
     and chunk size/overlap in one place.
  3. Indexing (`VectorStoreIndex`): Automatically chunks documents, computes embeddings,
     and builds a searchable vector index in 1 line of code (replacing 02_vector_store.py).
  4. Persistence (`storage_context.persist`): Saves the index to disk so documents do not
     need to be re-embedded on subsequent runs.
  5. Grounded Querying (`as_query_engine`) & Citations (`response.source_nodes`):
     Retrieves top-K chunks, injects them into a grounded prompt, and returns exact
     source chunks with similarity scores.

Dependencies:
  pip install llama-index-core llama-index-llms-google-genai llama-index-embeddings-google-genai
"""

import os
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ==============================================================================
# 1. API & DIRECTORY CONFIGURATION
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
    raise ValueError("GEMINI_API_KEY not found in environment or .env file.")

GEMINI_API_KEY = get_api_key()
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
PERSIST_DIR = SCRIPT_DIR / "llamaindex_storage"

# ==============================================================================
# 2. CONFIGURE LLAMAINDEX GLOBAL SETTINGS
# ==============================================================================
# Settings tells LlamaIndex which LLM, Embedding Model, and Chunker to use
# across every index and query engine automatically.
Settings.llm = GoogleGenAI(
    model="gemini-flash-lite-latest",
    api_key=GEMINI_API_KEY,
    temperature=0.0,
    max_retries=1,
)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GEMINI_API_KEY,
    retries=1,
)
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=32)

# ==============================================================================
# 3. INGESTION, INDEXING & PERSISTENCE
# ==============================================================================
def build_or_load_index() -> VectorStoreIndex:
    """Loads persisted vector index if available; otherwise ingests ./data and indexes it."""
    if PERSIST_DIR.exists():
        print(f"📂 Loading existing LlamaIndex store from: {PERSIST_DIR.name}/")
        storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        return load_index_from_storage(storage_context)

    print(f"📄 Loading raw documents from: {DATA_DIR.name}/")
    documents = SimpleDirectoryReader(input_dir=str(DATA_DIR)).load_data()
    print(f"   Loaded {len(documents)} document(s). Chunking & generating embeddings...")

    # Automatically splits documents into Nodes (chunks), calls Gemini embeddings,
    # and stores them in an in-memory VectorStoreIndex.
    index = VectorStoreIndex.from_documents(documents)

    # Persist to disk so we do not re-embed on the next run
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"💾 Index persisted to: {PERSIST_DIR.name}/")
    return index

# ==============================================================================
# 4. QUERY ENGINE EXECUTION WITH SOURCE CITATIONS
# ==============================================================================
def run_llamaindex_demo():
    print("=" * 75)
    print("🦙 LLAMAINDEX RAG PIPELINE: COMPLIANCE POLICY QUERY ENGINE")
    print("LLM: gemini-flash-lite-latest | Embeddings: gemini-embedding-001")
    print("=" * 75)

    index = build_or_load_index()

    # Turn the index into a RAG Query Engine (retrieves top-2 most relevant chunks)
    query_engine = index.as_query_engine(similarity_top_k=2)

    questions = [
        "What is the mandatory reporting threshold for international wire transfers, and what is structuring?",
        "Can we process a wire request for an unverified Cayman Islands shell company without UBO documents?",
    ]

    for i, question in enumerate(questions, start=1):
        print(f"\n--- [Query {i}] ---")
        print(f"❓ Question: {question}")

        response = query_engine.query(question)
        print(f"💡 Answer  : {str(response).strip()}")

        print("📚 Retrieved Source Chunks (Evidence):")
        for idx, node in enumerate(response.source_nodes, start=1):
            score = node.score if node.score is not None else 0.0
            snippet = node.get_text().replace("\n", " ")[:120]
            print(f"   [{idx}] Score={score:.4f} | File={node.metadata.get('file_name', 'unknown')} | \"{snippet}...\"")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    run_llamaindex_demo()
