"""
STEP 4: Autonomous AI Agent with Tools (The Foundation of MCP)
=============================================================
This script demonstrates an Autonomous Compliance Agent with a ReAct
(Reason + Act) loop.

Instead of a fixed RAG pipeline, the LLM has access to 3 tools:
  1. `lookup_account_sql`: Queries a Core Banking SQL database (sqlite3).
  2. `query_compliance_policy`: Queries our Step 2 Vector Store (vector_store.json).
  3. `raise_aml_alert_and_block`: Action tool that freezes an account and logs a regulatory incident.

The agent autonomously decides:
  - Which tool to call first
  - How to interpret the tool's result
  - Whether another tool is required
  - When to formulate the final audit report

Zero external pip dependencies (Standard Python 3 only).
"""

import os
import re
import json
import math
import sqlite3
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

# ==============================================================================
# TOOL 1: CORE BANKING RELATIONAL DATABASE (SQL)
# ==============================================================================
def init_mock_banking_db() -> sqlite3.Connection:
    """Sets up an in-memory SQL database with sample customer account records."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY,
            account_holder TEXT,
            account_type TEXT,
            balance_usd REAL,
            kyc_status TEXT,
            risk_tier TEXT,
            country TEXT
        )
    """)
    cursor.executemany("""
        INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        ("ACC-1002", "Alice Johnson", "INDIVIDUAL", 15400.0, "VERIFIED", "LOW_RISK", "USA"),
        ("ACC-9821", "Apex Offshore Holdings", "CORPORATE", 450000.0, "PENDING_VERIFICATION", "HIGH_RISK", "Cayman Islands"),
        ("ACC-4410", "Rapid Transit Logistics", "CORPORATE", 85000.0, "VERIFIED", "MEDIUM_RISK", "United Kingdom")
    ])
    conn.commit()
    return conn

DB_CONN = init_mock_banking_db()

def lookup_account_sql(account_id: str) -> str:
    """Tool: Queries core banking SQL database for an account's details."""
    cursor = DB_CONN.cursor()
    cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id.strip(),))
    row = cursor.fetchone()
    if not row:
        return json.dumps({"error": f"Account '{account_id}' not found in banking database."})
    return json.dumps({
        "account_id": row[0],
        "holder": row[1],
        "type": row[2],
        "balance_usd": row[3],
        "kyc_status": row[4],
        "risk_tier": row[5],
        "country": row[6]
    })

# ==============================================================================
# TOOL 2: COMPLIANCE POLICY VECTOR DATABASE (RAG)
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
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["embedding"]["values"]

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def query_compliance_policy(query: str) -> str:
    """Tool: Searches vector store for bank AML and compliance policies."""
    if not VECTOR_STORE_FILE.exists():
        return json.dumps({"error": "Vector store not found. Index documents first."})

    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    query_vec = get_embedding(query)
    scored = []
    for r in records:
        score = cosine_similarity(query_vec, r["embedding"])
        scored.append({"id": r["id"], "text": r["text"], "similarity": score})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_chunks = scored[:2]
    return json.dumps([{"chunk_id": c["id"], "text": c["text"]} for c in top_chunks])

# ==============================================================================
# TOOL 3: ACTION / AUDIT INCIDENT CREATOR
# ==============================================================================
def raise_aml_alert_and_block(account_id: str, reason: str) -> str:
    """Tool: Raises a high-severity AML incident ticket and freezes the transaction."""
    ticket_id = f"TICKET-AML-{abs(hash(account_id + reason)) % 100000}"
    return json.dumps({
        "status": "TRANSACTION_BLOCKED",
        "ticket_id": ticket_id,
        "account_id": account_id,
        "action_taken": "Account placed on temporary compliance hold",
        "reason": reason
    })

# ==============================================================================
# TOOL DISPATCHER (SIMULATING THE MCP CLIENT DISPATCHER)
# ==============================================================================
AVAILABLE_TOOLS = {
    "lookup_account_sql": lookup_account_sql,
    "query_compliance_policy": query_compliance_policy,
    "raise_aml_alert_and_block": raise_aml_alert_and_block
}

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Invokes the selected tool with supplied arguments."""
    func = AVAILABLE_TOOLS.get(tool_name)
    if not func:
        return json.dumps({"error": f"Tool '{tool_name}' is not recognized."})
    try:
        return func(**arguments)
    except Exception as e:
        return json.dumps({"error": f"Failed executing tool '{tool_name}': {str(e)}"})

# ==============================================================================
# LLM GENERATION WITH MULTI-MODEL FALLBACK
# ==============================================================================
def call_llm(prompt: str) -> str:
    import time
    for attempt in range(2):
        for model_name in GENERATION_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                time.sleep(1)
                continue
    return "[ERROR] Unable to reach LLM generation service."

# ==============================================================================
# THE AGENTIC REASONING LOOP (ReAct: Reason + Act)
# ==============================================================================
AGENT_SYSTEM_PROMPT = """You are an Autonomous Senior Financial Compliance Agent at Global Financial Corp.
You investigate transactions and ensure strict adherence to international AML and banking laws.

You have access to 3 tools:
1. `lookup_account_sql(account_id: str)`: Fetches account holder, balance, KYC verification status, risk tier, and country from the Core Banking SQL DB.
2. `query_compliance_policy(query: str)`: Searches bank compliance documentation for wire transfer rules, shell company rules, KYC thresholds, etc.
3. `raise_aml_alert_and_block(account_id: str, reason: str)`: Suspends the transaction and logs an official AML regulatory ticket when a violation is detected.

RULES:
- When you need information, call ONE tool at a time.
- Respond ONLY with a valid JSON object. No conversational preamble before or after the JSON.
- To call a tool, return:
{"thought": "reasoning on what information you need", "action": "call_tool", "tool_name": "<exact_tool_name>", "arguments": {"<param_name>": "<param_value>"}}
- When your investigation is complete and all necessary actions are taken, return:
{"thought": "reasoning explaining that all facts are collected", "action": "final_answer", "report": "<complete audit report with account findings, policy citations, and action taken>"}
"""

def extract_json(text: str) -> dict:
    """Robust extractor for JSON blocks returned by LLMs."""
    # Match markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Use JSONDecoder starting from the first '{' to parse valid JSON regardless of trailing characters
    idx = text.find("{")
    if idx != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[idx:])
            return obj
        except Exception:
            pass
    return {}

def run_autonomous_agent(investigation_prompt: str, max_iterations: int = 6):
    print("\n" + "=" * 80)
    print(f"  [AUDIT INVESTIGATION INITIATED]")
    print(f"  Query: \"{investigation_prompt}\"")
    print("=" * 80)

    prompt_history = f"{AGENT_SYSTEM_PROMPT}\n\nTask: {investigation_prompt}\n"

    for step in range(1, max_iterations + 1):
        print(f"\n>>> [STEP {step}: AGENT THINKING & DECISION]")

        response_text = call_llm(prompt_history)
        parsed = extract_json(response_text)

        if not parsed:
            print("  Raw Output (Unparseable):", response_text)
            break

        thought = parsed.get("thought", "")
        action = parsed.get("action", "")

        print(f"  Thought : {thought}")

        if action == "final_answer":
            print("\n" + "#" * 80)
            print("  [FINAL COMPLIANCE AUDIT REPORT - CASE CLOSED]")
            print("#" * 80)
            print(parsed.get("report", ""))
            print("#" * 80)
            return

        if action == "call_tool":
            tool_name = parsed.get("tool_name", "")
            args = parsed.get("arguments", {})
            print(f"  Action  : Execute Tool `{tool_name}` with {args}")

            # Execute tool in Python
            tool_result = execute_tool(tool_name, args)
            print(f"  Result  : {tool_result}")

            # Append the interaction to the agent's memory
            prompt_history += f"\nAgent Action: {json.dumps(parsed)}\nTool Result: {tool_result}\nWhat is your next step?"
        else:
            print(f"  [Unknown Action]: {action}")
            break

    print("[!] Agent reached max iterations.")

# ==============================================================================
# MAIN TESTS
# ==============================================================================
def main():
    print("=" * 80)
    print("  EXPERIMENT 4: AUTONOMOUS COMPLIANCE AGENT WITH TOOLS (MCP PATTERN)")
    print("=" * 80)

    # Test Scenario 1: High-Risk Violation
    # 1. SQL query ACC-9821 -> Cayman Islands, PENDING_VERIFICATION, HIGH_RISK
    # 2. Vector DB search -> High risk jurisdiction & unverified UBO rules
    # 3. Action tool -> Block transaction & raise alert
    # 4. Final audit report
    run_autonomous_agent(
        "Account ACC-9821 has requested an international wire transfer of $75,000 USD to a vendor in Zurich. "
        "Investigate this account and transaction against our compliance policy. "
        "If it violates any rules, block the transaction immediately and produce an audit report."
    )

    # Test Scenario 2: Fully Compliant Transaction
    # 1. SQL query ACC-1002 -> USA, VERIFIED, LOW_RISK, Balance $15,400
    # 2. Vector DB search -> Domestic wire transfer limits ($50,000 limit)
    # 3. Decision -> Approve without blocking
    run_autonomous_agent(
        "Account ACC-1002 has requested a domestic transfer of $4,000 USD to an electricity utility company. "
        "Investigate if this transaction requires blocking or special reporting."
    )

if __name__ == "__main__":
    main()
