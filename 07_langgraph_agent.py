"""
STEP 7: Production Agentic Workflows with LangGraph & LangChain
===============================================================
Demonstrates the modern industry-standard way to build autonomous agents:
  1. LangGraph StateGraph (Nodes, Edges, State, and Cycles).
  2. LangChain `@tool` decorators for strongly-typed tool definitions.
  3. Prebuilt `ToolNode` and conditional routing (`should_continue`).
  4. MemorySaver Checkpointer for session persistence (State Management).
  5. Native Pydantic Structured Outputs via `with_structured_output()`.

This implements the EXACT same banking compliance audit scenario as:
  - 04_agentic_tools.py (Manual string ReAct with regex)
  - 06_structured_tool_calling.py (Native API functionDeclarations)

Dependencies:
  pip install langgraph langchain-core langchain-google-genai pydantic
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field

# LangChain & LangGraph Core Components
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

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
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# ==============================================================================
# 2. OUTPUT CONTRACT (PYDANTIC DTO)
# ==============================================================================
class ComplianceAuditReport(BaseModel):
    """Final verified audit report produced by the LangGraph agent."""
    account_id: str = Field(description="The account ID audited")
    account_holder: str = Field(description="Name of the account holder or corporate entity")
    risk_level: str = Field(description="Evaluated risk: LOW_RISK, MEDIUM_RISK, or HIGH_RISK")
    violation_found: bool = Field(description="True if an AML/KYC violation was detected")
    policy_citations: List[str] = Field(description="Exact policy clause IDs cited from vector knowledge store")
    action_taken: str = Field(description="Specific action: ACCOUNT_FROZEN, TICKET_CREATED, or CLEARED")
    reasoning: str = Field(description="Executive summary explaining the decision with concrete figures")

# ==============================================================================
# 3. MOCK BANKING DATABASE
# ==============================================================================
def init_mock_banking_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
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
# 4. LANGCHAIN TOOLS (DECORATED WITH @tool)
# ==============================================================================
# In LangChain, the @tool decorator automatically converts the function's
# docstring and type hints into the exact JSON Schema required by the LLM.

@tool
def lookup_account_sql(account_id: str) -> str:
    """Queries the core banking SQL database to retrieve verified customer account details, balance, KYC status, and jurisdiction."""
    cursor = DB_CONN.cursor()
    cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id.strip(),))
    row = cursor.fetchone()
    if not row:
        return json.dumps({"error": f"Account '{account_id}' not found."})
    return json.dumps({
        "account_id": row[0],
        "account_holder": row[1],
        "type": row[2],
        "balance_usd": row[3],
        "kyc_status": row[4],
        "risk_tier": row[5],
        "country": row[6]
    })

@tool
def query_compliance_policy(query: str) -> str:
    """Searches the compliance vector knowledge base for regulatory policies, KYC limits, and AML thresholds."""
    # Simulated vector knowledge base lookup
    return json.dumps([
        {
            "clause_id": "AML-SEC-402",
            "text": "Accounts registered in high-risk offshore jurisdictions (e.g. Cayman Islands) with balances exceeding $100,000 USD must have VERIFIED KYC status before processing transactions."
        },
        {
            "clause_id": "AML-SEC-403",
            "text": "Unverified accounts in high-risk jurisdictions attempting transactions over $100,000 must be immediately placed on temporary hold and flagged to the AML Financial Crimes Unit."
        }
    ])

@tool
def raise_aml_alert_and_block(account_id: str, reason: str) -> str:
    """Freezes an account and files an official AML regulatory violation ticket when high-risk non-compliance is detected."""
    ticket_id = f"TICKET-AML-{abs(hash(account_id + reason)) % 100000}"
    print(f"\n🚨 [CRITICAL ACTION] Freezing Account {account_id} | Ticket: {ticket_id}")
    return json.dumps({
        "status": "ACCOUNT_FROZEN",
        "ticket_id": ticket_id,
        "account_id": account_id,
        "action_taken": "Account frozen and dispatched to Financial Crimes Unit",
        "reason": reason
    })

# The list of tools provided to LangGraph
TOOLS = [lookup_account_sql, query_compliance_policy, raise_aml_alert_and_block]

# ==============================================================================
# 5. LANGGRAPH STATE DEFINITION
# ==============================================================================
# The state is the data contract passed between nodes.
# 'add_messages' ensures that new messages from LLM or tools are automatically appended.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ==============================================================================
# 6. INITIALIZING THE LLM WITH TOOLS
# ==============================================================================
# Using ChatGoogleGenerativeAI with temperature=0.0 for deterministic execution
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.0,
    max_retries=1,
    timeout=20
)

# Bind tools directly to the model
llm_with_tools = llm.bind_tools(TOOLS)

# ==============================================================================
# 7. GRAPH NODES & CONDITIONAL ROUTING
# ==============================================================================
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a Senior Autonomous Financial Compliance Officer.\n"
    "Your mission is to rigorously investigate target bank accounts for AML violations.\n"
    "STEPS:\n"
    "1. Query the SQL banking database to inspect the account's details.\n"
    "2. Query compliance policies regarding the account's jurisdiction, balance, and KYC status.\n"
    "3. If an AML violation is detected, take immediate action using `raise_aml_alert_and_block`.\n"
    "4. When finished, summarize your findings.\n"
    "NEVER invent numbers or policies. Rely strictly on facts returned by your tools."
))

def agent_node(state: AgentState) -> dict:
    """The Agent Reasoning Node: Calls the LLM to decide the next step or tool."""
    # Ensure system prompt is at the start of conversation
    messages = [SYSTEM_PROMPT] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Conditional Edge: Determines if the agent called a tool or is finished."""
    last_message = state["messages"][-1]
    # Check if the LLM emitted any tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ==============================================================================
# 8. COMPILING THE LANGGRAPH STATE MACHINE
# ==============================================================================
def build_compliance_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))  # Prebuilt node that executes tools

    # 2. Add Edges
    workflow.add_edge(START, "agent")
    
    # Conditional edge: From agent -> either call tools OR terminate
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Cycle Edge: From tools -> loop back to agent
    workflow.add_edge("tools", "agent")

    # 3. Add Memory Checkpointer (Session Persistence)
    memory = MemorySaver()
    
    # Compile graph into an executable runnable
    app = workflow.compile(checkpointer=memory)
    return app

# ==============================================================================
# 9. RUNNER & STRUCTURED PYDANTIC OUTPUT
# ==============================================================================
def run_langgraph_audit(account_id: str):
    print("=" * 75)
    print(f"🚀 INITIATING LANGGRAPH AGENT AUDIT: Account {account_id}")
    print("Engine: LangGraph StateGraph | LLM: Gemini Flash Lite | Checkpointer: MemorySaver")
    print("=" * 75)

    app = build_compliance_graph()
    
    # Every session has a thread_id for checkpointing and memory persistence
    config = {"configurable": {"thread_id": f"audit-session-{account_id}"}}
    
    initial_input = {
        "messages": [
            HumanMessage(content=f"Conduct a full compliance audit on account: {account_id}. Verify whether it violates bank AML rules.")
        ]
    }

    # Stream the graph step by step
    for event in app.stream(initial_input, config=config, stream_mode="values"):
        last_msg = event["messages"][-1]
        
        # Display Agent Thoughts / Tool Calls
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"  👉 Agent Requested Tool : `{tc['name']}` with {tc['args']}")
        elif last_msg.type == "tool":
            print(f"  ✅ Tool Output Received : {last_msg.content[:85]}...")

    print("\n✅ LangGraph execution complete. Generating Pydantic report...")
    
    # Convert conversation history into a strongly-typed Pydantic model
    structured_llm = llm.with_structured_output(ComplianceAuditReport)
    
    # Fetch final state from checkpointer and ask the structured LLM to populate the DTO
    final_state = app.get_state(config)
    report_messages = list(final_state.values["messages"]) + [
        HumanMessage(content="Format the verified compliance audit findings above into the ComplianceAuditReport schema.")
    ]
    report = structured_llm.invoke(report_messages)
    return report

if __name__ == "__main__":
    test_account = "ACC-9821"
    try:
        audit_result = run_langgraph_audit(test_account)
        print("\n" + "=" * 75)
        print("📋 FINAL LANGGRAPH COMPLIANCE AUDIT REPORT (PYDANTIC OBJECT)")
        print("=" * 75)
        print(f"• Account ID       : {audit_result.account_id}")
        print(f"• Account Holder   : {audit_result.account_holder}")
        print(f"• Risk Level       : {audit_result.risk_level}")
        print(f"• Violation Found  : {audit_result.violation_found}")
        print(f"• Policy Citations : {', '.join(audit_result.policy_citations)}")
        print(f"• Action Taken     : {audit_result.action_taken}")
        print(f"• Reasoning        : {audit_result.reasoning}")
        print("=" * 75)
    except Exception as e:
        print(f"\n[Execution Error]: {e}")
