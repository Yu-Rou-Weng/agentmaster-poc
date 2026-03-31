"""
IR Agent (文件/檢索 AI)
Connects to ChromaDB vector database via MCP protocol.
Supports two scenarios:
  1. Enterprise operations: company policy documents
  2. MediaTek: release notes and engineering specifications
"""

import os
import json
import chromadb
from openai import OpenAI
from mcp_protocol import MCPServer, MCPTool

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# ============================================================
# Document Corpus — Demo Data
# ============================================================

POLICY_DOCS = [
    {
        "id": "policy-001",
        "title": "退換貨管理辦法",
        "category": "enterprise",
        "content": """第一條：本公司產品於購買日起 30 天內，如有品質瑕疵，客戶可申請免費換貨。
第二條：超過 30 天但未滿 90 天者，需經品管部門鑑定確認為非人為損壞後，方可辦理退換貨，但需扣除產品原價 15% 作為折舊費用。
第三條：超過 90 天之退換貨申請，除非屬於產品召回範圍，否則一律不予受理。
第四條：違約金規定——若客戶於合約期間內無故退貨達 3 次以上，將依合約第 12 條收取訂單總金額 5% 之違約金。
第五條：大量訂單（單筆超過 50 萬元）之退換貨，需由業務副總裁簽核後方可執行。""",
    },
    {
        "id": "policy-002",
        "title": "客戶信用評等管理辦法",
        "category": "enterprise",
        "content": """第一條：客戶信用評等分為 A、B、C、D 四級。A 級客戶享有 60 天帳期，B 級 45 天，C 級 30 天，D 級須預付款。
第二條：連續兩季業績低於合約保證量 80% 之客戶，信用評等自動降一級。
第三條：退貨率超過 10% 之客戶，信用評等自動降一級，且需重新簽訂合約。
第四條：信用評等為 D 級之客戶，業務部門需每月提交客戶維護報告。""",
    },
    {
        "id": "policy-003",
        "title": "Modem Firmware Release Note v3.2.1",
        "category": "mediatek",
        "content": """Release Date: 2026-02-28
Module: Modem Firmware
Version: v3.2.1

Key Fixes:
1. [CR-2026-001] Fixed race condition in RF band switching handler that caused firmware crash during inter-band handover. Root cause: concurrent access to shared RF state machine without proper mutex locking.
2. [CR-2026-005] Resolved 5G NR handover failure due to timing misalignment in RRC reconfiguration. Applied 2ms guard interval to measurement gap scheduling.
3. [CR-2026-010] Fixed VoNR call setup assertion caused by SIP timer misconfiguration. Corrected T1/T2 timer values per 3GPP TS 24.229 spec.

Known Issues:
- Intermittent RSRP measurement fluctuation under high-speed mobility (>300km/h). Under investigation.

Compatibility: Dimensity 9400 / 8300 platform, Android 16 BSP v2.1+""",
    },
    {
        "id": "policy-004",
        "title": "Power Management Design Guideline v2.0",
        "category": "mediatek",
        "content": """1. Thermal Throttling Policy:
   - CPU frequency reduction triggered at junction temperature Tj >= 85°C (previously 80°C, updated per CR-2026-008).
   - GPU throttle at Tj >= 90°C.
   - Emergency shutdown at Tj >= 105°C.

2. Idle Power Budget:
   - Target: < 15mW in deep sleep (Screen Off, Modem Idle).
   - CR-2026-003 identified abnormal 45mW consumption traced to PMIC regulator not entering low-power mode. Fix in progress.

3. Battery Charging Specification:
   - Support USB PD 3.1 (up to 240W) and proprietary Pump Express+ 5.0.
   - Charging IC thermal protection at 45°C battery surface temperature.""",
    },
    {
        "id": "policy-005",
        "title": "Software Quality Assurance Process",
        "category": "mediatek",
        "content": """CR (Change Request) Management SOP:
1. All CRs must be logged in JIRA within 24 hours of discovery.
2. Severity classification: Critical (service down/crash), Major (feature impaired), Minor (cosmetic/workaround available).
3. Critical CRs must have root cause analysis within 3 business days.
4. Resolution SLA: Critical = 14 days, Major = 30 days, Minor = 60 days.
5. All CR fixes require peer code review + regression test pass before merge.
6. Monthly CR metrics review: open/close ratio, average resolution time, recurrence rate.
7. Modules exceeding 5 open Critical CRs are escalated to VP Engineering for resource reallocation.""",
    },
]


# ============================================================
# Vector Database Initialization
# ============================================================

def init_vector_db():
    """Initialize ChromaDB with policy documents"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if any
    try:
        chroma_client.delete_collection("policy_docs")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="policy_docs",
        metadata={"hnsw:space": "cosine"},
    )

    for doc in POLICY_DOCS:
        collection.add(
            ids=[doc["id"]],
            documents=[doc["content"]],
            metadatas=[{"title": doc["title"], "category": doc["category"]}],
        )

    print(f"✅ Vector DB initialized with {len(POLICY_DOCS)} documents")
    return collection


# ============================================================
# MCP Server Setup
# ============================================================

def _search_documents(query: str, n_results: int = 3) -> dict:
    """Search the vector database for relevant documents"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma_client.get_collection("policy_docs")
    except Exception:
        return {"error": "Vector DB not initialized. Run init_vector_db() first."}

    results = collection.query(query_texts=[query], n_results=n_results)

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "category": results["metadatas"][0][i]["category"],
            "content": results["documents"][0][i],
            "relevance_distance": results["distances"][0][i],
        })

    return {"query": query, "results": docs, "count": len(docs)}


def _list_documents() -> dict:
    """List all available documents in the vector DB"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma_client.get_collection("policy_docs")
        all_docs = collection.get()
        docs = []
        for i in range(len(all_docs["ids"])):
            docs.append({
                "id": all_docs["ids"][i],
                "title": all_docs["metadatas"][i]["title"],
                "category": all_docs["metadatas"][i]["category"],
            })
        return {"documents": docs, "total": len(docs)}
    except Exception:
        return {"error": "Vector DB not initialized."}


def create_ir_mcp_server() -> MCPServer:
    """Create and configure the IR Agent's MCP server"""
    server = MCPServer(name="ir-document-server")

    server.register_tool(MCPTool(
        name="search_documents",
        description="Search policy documents and release notes by semantic similarity",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n_results": {"type": "integer", "default": 3},
            },
        },
        handler=_search_documents,
    ))

    server.register_tool(MCPTool(
        name="list_documents",
        description="List all available documents in the knowledge base",
        input_schema={"type": "object", "properties": {}},
        handler=_list_documents,
    ))

    return server


# ============================================================
# IR Agent Logic
# ============================================================

client = None


def get_openai_client():
    global client
    if client is None:
        client = OpenAI()
    return client


def ir_agent_execute(task_description: str) -> dict:
    """
    IR Agent: receives a task via A2A,
    searches vector DB via MCP, summarizes results with LLM.
    """
    mcp = create_ir_mcp_server()

    # Step 1: Search documents via MCP
    search_resp = mcp.call_tool("search_documents", {
        "query": task_description,
        "n_results": 3,
    })
    search_results = search_resp.result

    if "error" in search_results:
        return {"agent": "ir_agent", "error": search_results["error"]}

    # Step 2: Use LLM to summarize relevant documents
    docs_text = ""
    for doc in search_results["results"]:
        docs_text += f"\n--- {doc['title']} ---\n{doc['content']}\n"

    llm = get_openai_client()
    prompt = f"""你是一位企業政策與技術文件分析專家。根據以下檢索到的文件內容，針對使用者的問題提供精準的摘要與分析。

使用者問題：{task_description}

檢索到的相關文件：
{docs_text}

請用繁體中文回答，重點摘要相關規定或技術資訊，並標注來源文件名稱。如果文件中沒有直接相關的資訊，請誠實說明。"""

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )
    summary = response.choices[0].message.content.strip()

    return {
        "agent": "ir_agent",
        "retrieved_docs": [
            {"title": d["title"], "category": d["category"]}
            for d in search_results["results"]
        ],
        "summary": summary,
        "mcp_log": mcp.call_log,
    }


if __name__ == "__main__":
    init_vector_db()
    # Quick test
    result = ir_agent_execute("退換貨的違約金規定是什麼？")
    print(json.dumps(result, indent=2, ensure_ascii=False))
