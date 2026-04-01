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

    {"id": "knowhow-modem-001", "title": "[Modem Know-How] 5G NR Handover 常見失敗原因與排查步驟",
     "category": "knowhow", "domain": "modem",
     "content": """作者：Engineer_A (Modem Team)\nQ: 5G NR handover 失敗要怎麼排查？\n排查步驟：\n1. 確認是 intra-freq 還是 inter-freq handover，看 RRC Reconfiguration 的 measConfig\n2. 抓 modem log 搜尋 HO_FAIL 或 RLF\n3. 檢查 measurement gap 設定，常見問題是 gap pattern 跟 SSB periodicity 衝突\n4. 確認 target cell PCI 有沒有 confusion（PCI mod 3 相同導致偵測錯誤）\n5. CHO threshold 是否設太低\n常見地雷：Timing Advance 沒帶入→RACH失敗；measObjectNR ssbFrequency 不一致；NSA模式 A2 event threshold 太敏感\n相關 CR：CR-2026-005（已修復）"""},
    {"id": "knowhow-modem-002", "title": "[Modem Know-How] VoNR 通話建立失敗的 Debug 流程",
     "category": "knowhow", "domain": "modem",
     "content": """作者：Engineer_A\nQ: VoNR call setup 失敗怎麼查？\n1. 確認 IMS registration (AT+CIREG?)\n2. 抓 SIP trace 看 INVITE 和 100 Trying\n3. 常見：SIP timer T1 太短(預設500ms)；QoS flow 沒建立(查PDU Session Modification QFI=1)；Codec negotiation 失敗(確認SDP有AMR-WB和EVS)\n踩坑：某客戶不支援EVS只帶EVS→488 Not Acceptable，確保AMR-WB作fallback\nCR-2026-010: SIP timer VoNR用了VoLTE設定值"""},
    {"id": "knowhow-power-001", "title": "[Power Know-How] Idle 功耗異常的排查 SOP",
     "category": "knowhow", "domain": "power",
     "content": """作者：Engineer_C (Power Team)\nQ: 手機待機功耗偏高怎麼查？\n1. Power monitor量suspend電流，正常<5mA\n2. 查wakelock: adb shell dumpsys power | grep Wake Locks\n3. 查PMIC: cat /sys/kernel/debug/regulator/regulator_summary，常見某regulator沒進LPM\n4. Modem: 檢查eDRX/PSM、paging cycle\n5. Connectivity: Wi-Fi beacon interval(DTIM x3+)、BT LE scan背景\n地雷：Sensor hub I2C沒clock gating→2-3mA(CR-2026-012)；GPS TCXO沒關→8mA；PMIC S2R slew rate錯→延遲deep sleep"""},
    {"id": "knowhow-connectivity-001", "title": "[Connectivity Know-How] Wi-Fi 7 MLO 斷線問題排查",
     "category": "knowhow", "domain": "connectivity",
     "content": """作者：Engineer_E (Connectivity Team)\nQ: Wi-Fi 7 MLO連線不穩怎麼查？\n1. 確認AP真的支援MLO(看beacon Multi-Link Element)\n2. 抓driver log搜MLO/link switch/EMLSR\n3. 常見：EMLSR切換太慢(預期<50us有些AP要100us+)；TID-to-link mapping不一致；CSA在一個link發生時MLO行為不一致\n踩坑：某品牌AP 6GHz DFS radar偵測後CSA切channel但MLE link info沒同步→STA認為link斷了\nCR-2026-009正在查，workaround: disable 6GHz只用2.4G+5G"""},
    {"id": "knowhow-build-001", "title": "[Build/Release Know-How] 自動化打包軟體 Patch 流程",
     "category": "knowhow", "domain": "build",
     "content": """作者：Build Team\nQ: 怎麼打包patch給客戶？\n1. 從Gerrit找cherry-pick commits，確認有CR number+code review\n2. 切branch: git checkout -b patch_vX.X.X_hotfix customer/vX.X.X\n3. Cherry-pick(按dependency順序)\n4. 觸發Jenkins CI build，跑BVT+smoke test\n5. Artifacts上傳Artifactory\n6. release_tool.py --mode=patch 生成package\n7. 上傳客戶FTP通知FAE+PM\n注意：不同客戶baseline不同！conflict找原author不要自己解；每個patch建JIRA Release ticket"""},
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



# ============================================================
# Knowledge Upload Feature (Added for Scene 3)
# ============================================================

def upload_knowledge(title: str, content: str, domain: str, author: str = "Anonymous") -> dict:
    """Allow engineers to upload their domain know-how"""
    import time as _t
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma_client.get_collection("policy_docs")
    except Exception:
        return {"error": "Vector DB not initialized."}
    doc_id = f"knowhow-upload-{int(_t.time())}"
    formatted_content = f"作者：{author}\n\n{content}"
    collection.add(
        ids=[doc_id], documents=[formatted_content],
        metadatas=[{"title": f"[{domain.upper()} Know-How] {title}", "category": "knowhow", "domain": domain.lower()}],
    )
    return {"status": "success", "doc_id": doc_id, "title": title, "domain": domain, "message": f"知識已上傳到 {domain} 知識庫！"}


def list_knowledge_by_domain() -> dict:
    """List all know-how documents grouped by domain"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma_client.get_collection("policy_docs")
        all_docs = collection.get()
        domains = {}
        for i in range(len(all_docs["ids"])):
            meta = all_docs["metadatas"][i]
            domain = meta.get("domain", "general")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append({"id": all_docs["ids"][i], "title": meta["title"], "category": meta["category"]})
        return {"domains": domains, "total": len(all_docs["ids"])}
    except Exception:
        return {"error": "Vector DB not initialized."}

if __name__ == "__main__":
    init_vector_db()
    # Quick test
    result = ir_agent_execute("退換貨的違約金規定是什麼？")
    print(json.dumps(result, indent=2, ensure_ascii=False))
