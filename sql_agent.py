"""
SQL Agent (數據 AI)
Connects to SQLite via MCP protocol for structured data queries.
Supports two scenarios:
  1. Enterprise operations: customer sales data
  2. MediaTek: CR (Change Request) analysis data
"""

import os
import sqlite3
import json
from openai import OpenAI
from mcp_protocol import MCPServer, MCPTool

# ============================================================
# Database Initialization — Demo Data
# ============================================================

DB_PATH = os.path.join(os.path.dirname(__file__), "demo_data.db")


def init_database():
    """Create demo tables with sample data for both scenarios"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # --- Scenario 1: Enterprise Operations (客戶業績) ---
    c.execute("DROP TABLE IF EXISTS customers")
    c.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            region TEXT,
            sales_amount REAL,
            return_count INTEGER,
            last_order_date TEXT,
            account_manager TEXT
        )
    """)
    customers = [
        (1, "台北精密科技", "北區", 1250000, 2, "2025-12-15", "王小明"),
        (2, "高雄光電股份", "南區", 890000, 5, "2025-11-20", "李大華"),
        (3, "新竹半導體", "北區", 3200000, 0, "2026-01-10", "王小明"),
        (4, "台中機械工業", "中區", 320000, 8, "2025-10-05", "陳美玲"),
        (5, "桃園電子", "北區", 750000, 3, "2025-12-28", "張志豪"),
        (6, "台南化工", "南區", 180000, 12, "2025-09-15", "李大華"),
        (7, "花蓮農產加工", "東區", 95000, 1, "2025-11-30", "陳美玲"),
        (8, "嘉義食品", "南區", 210000, 6, "2025-10-22", "張志豪"),
        (9, "宜蘭生技", "東區", 150000, 4, "2026-01-05", "王小明"),
        (10, "彰化精工", "中區", 420000, 2, "2025-12-01", "陳美玲"),
    ]
    c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", customers)

    # --- Scenario 2: MediaTek CR Analysis (Change Requests) ---
    c.execute("DROP TABLE IF EXISTS change_requests")
    c.execute("""
        CREATE TABLE change_requests (
            cr_id TEXT PRIMARY KEY,
            title TEXT,
            module TEXT,
            severity TEXT,
            status TEXT,
            created_date TEXT,
            resolved_date TEXT,
            assignee TEXT,
            root_cause TEXT,
            resolution_time_days INTEGER
        )
    """)
    crs = [
        ("CR-2026-001", "Modem FW crash on band switching", "Modem", "Critical", "Resolved", "2026-01-05", "2026-01-12", "Engineer_A", "Race condition in RF handler", 7),
        ("CR-2026-002", "Audio latency in BT codec", "Audio", "Major", "Resolved", "2026-01-08", "2026-01-20", "Engineer_B", "Buffer underrun in DSP pipeline", 12),
        ("CR-2026-003", "Power consumption spike in idle", "Power", "Critical", "Open", "2026-01-15", None, "Engineer_C", None, None),
        ("CR-2026-004", "Camera ISP color shift under low light", "Camera", "Minor", "Resolved", "2026-01-18", "2026-01-22", "Engineer_D", "White balance LUT outdated", 4),
        ("CR-2026-005", "5G NR handover failure", "Modem", "Critical", "Resolved", "2026-01-20", "2026-02-05", "Engineer_A", "Timing misalignment in RRC reconfig", 16),
        ("CR-2026-006", "GPS cold start timeout", "Connectivity", "Major", "Open", "2026-02-01", None, "Engineer_E", None, None),
        ("CR-2026-007", "Display flicker on OLED panel", "Display", "Major", "Resolved", "2026-02-03", "2026-02-10", "Engineer_F", "PWM freq below flicker threshold", 7),
        ("CR-2026-008", "Thermal throttling too aggressive", "Power", "Minor", "Resolved", "2026-02-05", "2026-02-08", "Engineer_C", "Threshold set 5C too low", 3),
        ("CR-2026-009", "Wi-Fi 7 MLO link drop", "Connectivity", "Critical", "Open", "2026-02-10", None, "Engineer_E", None, None),
        ("CR-2026-010", "Modem assert on VoNR call setup", "Modem", "Critical", "Resolved", "2026-02-12", "2026-02-28", "Engineer_A", "SIP timer misconfiguration", 16),
        ("CR-2026-011", "Memory leak in media codec", "Media", "Major", "Resolved", "2026-02-15", "2026-02-22", "Engineer_B", "Missing buffer release in error path", 7),
        ("CR-2026-012", "Sensor hub I2C timeout", "Sensor", "Minor", "Resolved", "2026-02-18", "2026-02-20", "Engineer_D", "Pull-up resistor value incorrect", 2),
    ]
    c.executemany("INSERT INTO change_requests VALUES (?,?,?,?,?,?,?,?,?,?)", crs)

    conn.commit()
    conn.close()
    print("✅ Demo database initialized")


# ============================================================
# MCP Server Setup
# ============================================================

def _execute_sql(query: str) -> dict:
    """Execute a read-only SQL query against the demo database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def _get_schema() -> dict:
    """Return database schema information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table'"
    )
    tables = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return {"tables": tables}


def create_sql_mcp_server() -> MCPServer:
    """Create and configure the SQL Agent's MCP server"""
    server = MCPServer(name="sql-data-server")

    server.register_tool(MCPTool(
        name="execute_query",
        description="Execute a read-only SQL SELECT query against the database",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=_execute_sql,
    ))

    server.register_tool(MCPTool(
        name="get_schema",
        description="Get the database schema (table names and column definitions)",
        input_schema={"type": "object", "properties": {}},
        handler=_get_schema,
    ))

    return server


# ============================================================
# SQL Agent Logic
# ============================================================

client = None


def get_openai_client():
    global client
    if client is None:
        client = OpenAI()
    return client


def sql_agent_execute(task_description: str) -> dict:
    """
    SQL Agent: receives a natural language task via A2A,
    generates SQL via LLM, executes via MCP, returns results.
    """
    mcp = create_sql_mcp_server()

    # Step 1: Get schema via MCP
    schema_resp = mcp.call_tool("get_schema", {})
    schema_info = schema_resp.result

    # Step 2: Use LLM to generate SQL from natural language
    llm = get_openai_client()
    prompt = f"""You are a SQL expert. Given the database schema below, generate a SQLite SELECT query to answer the user's question.

DATABASE SCHEMA:
{json.dumps(schema_info, indent=2, ensure_ascii=False)}

USER QUESTION: {task_description}

RULES:
- Output ONLY the SQL query, nothing else.
- Use only SELECT statements (read-only).
- Return useful columns for the answer.
- If asking for "worst performing" or "lowest", ORDER BY the relevant metric ASC and LIMIT.
- Use Chinese-friendly column aliases when appropriate.
"""

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    sql_query = response.choices[0].message.content.strip()
    # Clean markdown code fences if present
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    # Step 3: Execute SQL via MCP
    query_resp = mcp.call_tool("execute_query", {"query": sql_query})

    return {
        "agent": "sql_agent",
        "generated_sql": sql_query,
        "data": query_resp.result,
        "mcp_log": mcp.call_log,
    }


if __name__ == "__main__":
    init_database()
    # Quick test
    result = sql_agent_execute("找出業績最差的三個客戶")
    print(json.dumps(result, indent=2, ensure_ascii=False))
