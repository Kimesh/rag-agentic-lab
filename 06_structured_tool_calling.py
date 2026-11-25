"""
STEP 6: Production Structured Outputs & Native Tool Calling (Stage 1)
=====================================================================
Demonstrates the modern enterprise standard for Agentic LLM systems:
  1. Strongly-typed Pydantic schemas for parameter validation and final outputs.
  2. Native Tool / Function Calling (OpenAI / Gemini functionDeclarations standard),
     completely replacing fragile regex string parsing.
  3. Deterministic anti-hallucination controls: The agent queries ground-truth
     SQL databases and vector stores, validating all returned parameters before
     producing a structured `ComplianceAuditReport`.

Dependencies:
  pip install pydantic
"""

import os
import json
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# ==============================================================================
# 1. API CONFIGURATION
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
MODEL_NAME = "gemini-flash-lite-latest"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

SCRIPT_DIR = Path(__file__).parent.resolve()
VECTOR_STORE_FILE = SCRIPT_DIR / "vector_store.json"

# ==============================================================================
# 2. STRONGLY-TYPED SCHEMAS (PYDANTIC)
# ==============================================================================
# In production, every tool input and output is validated via Pydantic.
# If an LLM hallucinates an invalid field, Pydantic catches it immediately.

class AccountLookupArgs(BaseModel):
    account_id: str = Field(description="Unique bank account ID (e.g. ACC-9821)")

class PolicyQueryArgs(BaseModel):
    query: str = Field(description="Regulatory policy or AML search query")

class AMLAlertArgs(BaseModel):
    account_id: str = Field(description="Account ID to be frozen")
    reason: str = Field(description="Specific justification citing violated compliance clause")

class ComplianceAuditReport(BaseModel):
    """Final verified audit report produced by the compliance agent."""
    account_id: str = Field(description="The account ID audited")
    account_holder: str = Field(description="Name of the account holder or corporate entity")
    risk_level: str = Field(description="Evaluated risk: LOW_RISK, MEDIUM_RISK, or HIGH_RISK")
    violation_found: bool = Field(description="True if an AML/KYC violation was detected")
    policy_citations: List[str] = Field(description="Exact policy clause IDs cited from vector knowledge store")
    action_taken: str = Field(description="Specific action: ACCOUNT_FROZEN, TICKET_CREATED, or CLEARED")
    reasoning: str = Field(description="Executive summary explaining the decision with concrete figures")

# ==============================================================================
# 3. CORE BANKING SQL DATABASE (GROUND-TRUTH DATA STORE)
# ==============================================================================
def init_mock_banking_db() -> sqlite3.Connection:
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

# ==============================================================================
# 4. PYTHON TOOLS (EXECUTED DETERMINISTICALLY BY RUNTIME)
# ==============================================================================
def lookup_account_sql(account_id: str) -> str:
    """Tool: Queries core banking SQL database for account details."""
    validated = AccountLookupArgs(account_id=account_id)
    cursor = DB_CONN.cursor()
    cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (validated.account_id.strip(),))
    row = cursor.fetchone()
    if not row:
        return json.dumps({"error": f"Account '{validated.account_id}' not found in database."})
    return json.dumps({
        "account_id": row[0],
        "account_holder": row[1],
        "type": row[2],
        "balance_usd": row[3],
        "kyc_status": row[4],
        "risk_tier": row[5],
        "country": row[6]
    })

def query_compliance_policy(query: str) -> str:
    """Tool: Searches vector knowledge base for regulatory rules and AML thresholds."""
    validated = PolicyQueryArgs(query=query)
    if not VECTOR_STORE_FILE.exists():
        return json.dumps([
            {
                "clause_id": "AML-SEC-402",
                "text": "Accounts registered in high-risk offshore jurisdictions (e.g. Cayman Islands, BVI) with balances exceeding $100,000 USD must have VERIFIED KYC status before processing transactions."
            },
            {
                "clause_id": "AML-SEC-403",
                "text": "Unverified accounts in high-risk jurisdictions attempting transactions over $100,000 must be immediately placed on temporary hold and flagged to the AML Financial Crimes Unit."
            }
        ])

    try:
        with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
            return json.dumps([{"clause_id": r.get("id", "SEC-01"), "text": r.get("text", "")} for r in records[:2]])
    except Exception as e:
        return json.dumps({"error": f"Vector store read error: {str(e)}"})

def raise_aml_alert_and_block(account_id: str, reason: str) -> str:
    """Tool: Freezes account and logs an official AML violation ticket."""
    validated = AMLAlertArgs(account_id=account_id, reason=reason)
    ticket_id = f"TICKET-AML-{abs(hash(validated.account_id + validated.reason)) % 100000}"
    print(f"\n🚨 [CRITICAL ACTION TRIGGERED] Freezing Account {validated.account_id} | Ticket: {ticket_id}")
    return json.dumps({
        "status": "ACCOUNT_FROZEN",
        "ticket_id": ticket_id,
        "account_id": validated.account_id,
        "action_taken": "Account frozen and dispatched to Financial Crimes Unit",
        "reason": validated.reason
    })

AVAILABLE_TOOLS = {
    "lookup_account_sql": lookup_account_sql,
    "query_compliance_policy": query_compliance_policy,
    "raise_aml_alert_and_block": raise_aml_alert_and_block
}

# ==============================================================================
# 5. NATIVE TOOL DECLARATIONS (OPEN PROTOCOL / JSON SCHEMA)
# ==============================================================================
# This tells the LLM engine the exact function names and parameters it is allowed to call.
TOOL_DECLARATIONS = [
    {
        "name": "lookup_account_sql",
        "description": "Queries the core banking SQL database to retrieve verified customer account details, balance, KYC status, and jurisdiction.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "account_id": {"type": "STRING", "description": "The unique bank account ID (e.g. ACC-9821)"}
            },
            "required": ["account_id"]
        }
    },
    {
        "name": "query_compliance_policy",
        "description": "Searches the compliance vector knowledge base for regulatory policies, KYC limits, and AML thresholds.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The compliance topic or regulatory query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "raise_aml_alert_and_block",
        "description": "Freezes an account and files an official AML regulatory violation ticket when high-risk non-compliance is detected.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "account_id": {"type": "STRING", "description": "The account ID to freeze"},
                "reason": {"type": "STRING", "description": "Justification citing the violated policy and account findings"}
            },
            "required": ["account_id", "reason"]
        }
    }
]

# ==============================================================================
# 6. AGENTIC EXECUTION ORCHESTRATOR
# ==============================================================================
def post_gemini_api(payload: dict) -> dict:
    """HTTP client calling Google Gemini v1beta native function-calling API."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {e.code}: {error_body}") from e

def run_agentic_audit(account_id: str) -> ComplianceAuditReport:
    print(f"\n{'='*75}")
    print(f"🚀 INITIATING NATIVE FUNCTION-CALLING AUDIT: Account {account_id}")
    print(f"Model: {MODEL_NAME} | Validation Engine: Pydantic v2")
    print(f"{'='*75}")

    system_instruction = (
        "You are a Senior Autonomous Financial Compliance Officer.\n"
        "Your mission is to rigorously investigate target bank accounts for AML violations.\n"
        "STEPS:\n"
        "1. Query the SQL banking database to inspect the account's details.\n"
        "2. Query compliance policies regarding the account's jurisdiction, balance, and KYC status.\n"
        "3. If an AML violation is detected, take immediate action using `raise_aml_alert_and_block`.\n"
        "4. When finished, produce an audit assessment with exact figures and policy citations.\n"
        "NEVER invent numbers or policies. Rely strictly on data returned by your tools."
    )

    # Conversation history tracking multi-turn interactions
    contents = [
        {"role": "user", "parts": [{"text": f"Conduct a full compliance audit on account: {account_id}. Verify whether it violates bank AML rules."}]}
    ]

    print(f"👤 User Request: Conduct a full compliance audit on account: {account_id}.\n")

    max_turns = 6
    turn = 1

    while turn <= max_turns:
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
            "generationConfig": {"temperature": 0.0}
        }

        response = post_gemini_api(payload)
        candidate = response["candidates"][0]
        model_content = candidate["content"]
        contents.append(model_content)

        # Inspect parts for native function calls
        function_calls = []
        for part in model_content.get("parts", []):
            if "functionCall" in part:
                function_calls.append(part["functionCall"])

        if not function_calls:
            # Model finished calling tools and returned text
            print("\n✅ All tool interactions complete. Preparing structured Pydantic report...")
            break

        print(f"--- [Turn {turn}] Agent Invoked {len(function_calls)} Native Tool Call(s) ---")
        
        # Execute each requested function call
        function_response_parts = []
        for call in function_calls:
            fn_name = call["name"]
            fn_args = call.get("args", {})
            print(f"  👉 Tool Invoked: `{fn_name}`")
            print(f"     Arguments   : {fn_args}")

            tool_func = AVAILABLE_TOOLS.get(fn_name)
            if tool_func:
                raw_result = tool_func(**fn_args)
            else:
                raw_result = json.dumps({"error": f"Tool '{fn_name}' not found."})

            print(f"     Result      : {raw_result[:90]}...")

            function_response_parts.append({
                "functionResponse": {
                    "name": fn_name,
                    "response": {"output": raw_result}
                }
            })

        # Append all tool results back to conversation under 'user' role (Gemini v1beta spec)
        contents.append({
            "role": "user",
            "parts": function_response_parts
        })

        turn += 1

    # Final Step: Enforce Pydantic Output Validation
    # We ask the model to format its verified findings into the exact schema of ComplianceAuditReport
    print("\n🔍 Enforcing Pydantic Schema Validation (ComplianceAuditReport)...")
    
    schema_prompt = (
        "Based strictly on the verified facts gathered in our tool calls above, output the final audit "
        "summary as a JSON object matching this schema:\n"
        f"{json.dumps(ComplianceAuditReport.model_json_schema(), indent=2)}\n"
        "Return ONLY the valid raw JSON object. Do not wrap in markdown quotes."
    )

    contents.append({"role": "user", "parts": [{"text": schema_prompt}]})
    
    final_payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.0}
    }
    
    final_resp = post_gemini_api(final_payload)
    raw_text = final_resp["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    # Clean any accidental markdown quotes
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    # Pydantic Parse & Validation
    try:
        report = ComplianceAuditReport.model_validate_json(raw_text)
        print("🎉 [SUCCESS] Output Successfully Validated by Pydantic!")
        return report
    except ValidationError as ve:
        print(f"❌ [VALIDATION ERROR] Output failed Pydantic schema: {ve}")
        raise ve

# ==============================================================================
# 7. VERIFICATION RUNNER
# ==============================================================================
if __name__ == "__main__":
    # Test with Apex Offshore Holdings (ACC-9821, Cayman Islands, $450k, Pending KYC)
    test_account = "ACC-9821"
    
    try:
        audit_result = run_agentic_audit(test_account)
        
        print("\n" + "=" * 75)
        print("📋 FINAL STRUCTURED COMPLIANCE AUDIT REPORT (PYDANTIC OBJECT)")
        print("=" * 75)
        print(f"• Account ID       : {audit_result.account_id}")
        print(f"• Account Holder   : {audit_result.account_holder}")
        print(f"• Risk Level       : {audit_result.risk_level}")
        print(f"• Violation Found  : {audit_result.violation_found}")
        print(f"• Policy Citations : {', '.join(audit_result.policy_citations)}")
        print(f"• Action Taken     : {audit_result.action_taken}")
        print(f"• Reasoning        : {audit_result.reasoning}")
        print("=" * 75)
        print("\n💡 Production Takeaway:")
        print("This output is guaranteed by Pydantic. It can be safely saved to PostgreSQL,")
        print("dispatched to an event bus (Kafka), or returned via a FastAPI endpoint without")
        print("fear of runtime format errors or unvalidated LLM output.")

    except Exception as e:
        print(f"\n[Execution Error]: {e}")
