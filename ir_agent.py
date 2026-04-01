"""
IR Agent v3 — Lightweight Vector DB (no ChromaDB dependency)
"""
import os, json, time as _time
import lite_vectordb
from openai import OpenAI
from mcp_protocol import MCPServer, MCPTool

POLICY_DOCS = [
    {"id": "policy-001", "title": "退換貨管理辦法", "category": "enterprise", "domain": "general",
     "content": "第一條：本公司產品於購買日起30天內，如有品質瑕疵，客戶可申請免費換貨。\n第二條：超過30天但未滿90天者，需經品管部門鑑定確認為非人為損壞後，方可辦理退換貨，但需扣除產品原價15%作為折舊費用。\n第三條：超過90天之退換貨申請，除非屬於產品召回範圍，否則一律不予受理。\n第四條：違約金規定——若客戶於合約期間內無故退貨達3次以上，將依合約第12條收取訂單總金額5%之違約金。\n第五條：大量訂單（單筆超過50萬元）之退換貨，需由業務副總裁簽核後方可執行。"},
    {"id": "policy-002", "title": "客戶信用評等管理辦法", "category": "enterprise", "domain": "general",
     "content": "第一條：客戶信用評等分為A、B、C、D四級。A級客戶享有60天帳期，B級45天，C級30天，D級須預付款。\n第二條：連續兩季業績低於合約保證量80%之客戶，信用評等自動降一級。\n第三條：退貨率超過10%之客戶，信用評等自動降一級，且需重新簽訂合約。\n第四條：信用評等為D級之客戶，業務部門需每月提交客戶維護報告。"},
    {"id": "policy-003", "title": "Modem Firmware Release Note v3.2.1", "category": "mediatek", "domain": "modem",
     "content": "Release Date: 2026-02-28 | Module: Modem Firmware | Version: v3.2.1\nKey Fixes:\n1. [CR-2026-001] Fixed race condition in RF band switching handler. Root cause: concurrent access to shared RF state machine without proper mutex locking.\n2. [CR-2026-005] Resolved 5G NR handover failure due to timing misalignment in RRC reconfiguration. Applied 2ms guard interval.\n3. [CR-2026-010] Fixed VoNR call setup assertion caused by SIP timer misconfiguration.\nKnown Issues: Intermittent RSRP measurement fluctuation under high-speed mobility (>300km/h).\nCompatibility: Dimensity 9400 / 8300, Android 16 BSP v2.1+"},
    {"id": "policy-004", "title": "Power Management Design Guideline v2.0", "category": "mediatek", "domain": "power",
     "content": "Thermal Throttling: CPU freq reduction at Tj >= 85C (updated per CR-2026-008). GPU throttle at 90C. Emergency shutdown at 105C.\nIdle Power Budget: Target < 15mW deep sleep. CR-2026-003 found 45mW abnormal consumption from PMIC regulator not entering LPM.\nBattery Charging: USB PD 3.1 up to 240W, Pump Express+ 5.0. Thermal protection at 45C battery surface."},
    {"id": "policy-005", "title": "Software Quality Assurance Process", "category": "mediatek", "domain": "general",
     "content": "CR Management SOP:\n1. Log in JIRA within 24h. 2. Severity: Critical/Major/Minor. 3. Critical RCA within 3 days.\n4. SLA: Critical=14d, Major=30d, Minor=60d. 5. Peer review + regression before merge.\n6. Monthly metrics review. 7. Modules with 5+ open Critical CRs escalated to VP."},
    {"id": "knowhow-modem-001", "title": "[Modem Know-How] 5G NR Handover 常見失敗原因與排查步驟", "category": "knowhow", "domain": "modem",
     "content": "作者：Engineer_A (Modem Team)\nQ: 5G NR handover 失敗要怎麼排查？\n排查步驟：\n1. 確認是 intra-freq 還是 inter-freq handover，看 RRC Reconfiguration 的 measConfig\n2. 抓 modem log 搜尋 HO_FAIL 或 RLF\n3. 檢查 measurement gap 設定，常見問題是 gap pattern 跟 SSB periodicity 衝突\n4. 確認 target cell PCI 有沒有 confusion（PCI mod 3 相同導致偵測錯誤）\n5. CHO threshold 是否設太低\n常見地雷：Timing Advance 沒帶入導致RACH失敗；measObjectNR ssbFrequency 不一致；NSA模式 A2 event threshold 太敏感\n相關 CR：CR-2026-005（已修復）"},
    {"id": "knowhow-modem-002", "title": "[Modem Know-How] VoNR 通話建立失敗的 Debug 流程", "category": "knowhow", "domain": "modem",
     "content": "作者：Engineer_A\nQ: VoNR call setup 失敗怎麼查？\n1. 確認 IMS registration (AT+CIREG?)\n2. 抓 SIP trace 看 INVITE 和 100 Trying\n3. 常見：SIP timer T1 太短(預設500ms)；QoS flow 沒建立(查PDU Session Modification QFI=1)；Codec negotiation 失敗(確認SDP有AMR-WB和EVS)\n踩坑：某客戶不支援EVS只帶EVS導致488 Not Acceptable，確保AMR-WB作fallback\nCR-2026-010: SIP timer VoNR用了VoLTE設定值"},
    {"id": "knowhow-power-001", "title": "[Power Know-How] Idle 功耗異常的排查 SOP", "category": "knowhow", "domain": "power",
     "content": "作者：Engineer_C (Power Team)\nQ: 手機待機功耗偏高怎麼查？\n1. Power monitor量suspend電流，正常<5mA\n2. 查wakelock: adb shell dumpsys power | grep Wake Locks\n3. 查PMIC: cat /sys/kernel/debug/regulator/regulator_summary，常見某regulator沒進LPM\n4. Modem: 檢查eDRX/PSM、paging cycle\n5. Connectivity: Wi-Fi beacon interval(DTIM x3+)、BT LE scan背景\n地雷：Sensor hub I2C沒clock gating導致2-3mA(CR-2026-012)；GPS TCXO沒關導致8mA；PMIC S2R slew rate錯導致延遲deep sleep"},
    {"id": "knowhow-connectivity-001", "title": "[Connectivity Know-How] Wi-Fi 7 MLO 斷線問題排查", "category": "knowhow", "domain": "connectivity",
     "content": "作者：Engineer_E (Connectivity Team)\nQ: Wi-Fi 7 MLO連線不穩怎麼查？\n1. 確認AP真的支援MLO(看beacon Multi-Link Element)\n2. 抓driver log搜MLO/link switch/EMLSR\n3. 常見：EMLSR切換太慢(預期<50us有些AP要100us+)；TID-to-link mapping不一致；CSA在一個link發生時MLO行為不一致\n踩坑：某品牌AP 6GHz DFS radar偵測後CSA切channel但MLE link info沒同步導致STA認為link斷了\nCR-2026-009正在查，workaround: disable 6GHz只用2.4G+5G"},
    {"id": "knowhow-build-001", "title": "[Build/Release Know-How] 自動化打包軟體 Patch 流程", "category": "knowhow", "domain": "build",
     "content": "作者：Build Team\nQ: 怎麼打包patch給客戶？\n1. 從Gerrit找cherry-pick commits，確認有CR number+code review\n2. 切branch: git checkout -b patch_vX.X.X_hotfix customer/vX.X.X\n3. Cherry-pick(按dependency順序)\n4. 觸發Jenkins CI build，跑BVT+smoke test\n5. Artifacts上傳Artifactory\n6. release_tool.py --mode=patch 生成package\n7. 上傳客戶FTP通知FAE+PM\n注意：不同客戶baseline不同！conflict找原author不要自己解；每個patch建JIRA Release ticket"},
]


def init_vector_db():
    if lite_vectordb.is_initialized():
        print("✅ Vector DB already initialized, skipping")
        return
    count = lite_vectordb.init_collection(POLICY_DOCS)
    print(f"✅ Vector DB initialized with {count} documents (including domain know-how)")


def upload_knowledge(title, content, domain, author="Anonymous"):
    if not lite_vectordb.is_initialized():
        return {"error": "Vector DB not initialized."}
    doc_id = f"knowhow-upload-{int(_time.time())}"
    full_title = f"[{domain.upper()} Know-How] {title}"
    lite_vectordb.add_document(doc_id, full_title, f"作者：{author}\n\n{content}", "knowhow", domain.lower())
    return {"status": "success", "doc_id": doc_id, "title": title, "domain": domain, "message": f"知識已上傳到 {domain} 知識庫！"}


def list_knowledge_by_domain():
    return lite_vectordb.list_all()


def _search_documents(query, n_results=3, domain=None):
    results = lite_vectordb.search(query, n_results=n_results, domain=domain)
    docs = [{"id": r["id"], "title": r["title"], "category": r["category"],
             "domain": r["domain"], "content": r["content"], "relevance_distance": 1-r["score"]} for r in results]
    return {"query": query, "domain_filter": domain, "results": docs, "count": len(docs)}


def _list_documents():
    return lite_vectordb.list_all()


def create_ir_mcp_server():
    server = MCPServer(name="ir-document-server")
    server.register_tool(MCPTool(
        name="search_documents", description="Search documents and domain know-how by semantic similarity",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}, "n_results": {"type": "integer"}, "domain": {"type": "string"}}},
        handler=_search_documents,
    ))
    server.register_tool(MCPTool(
        name="list_documents", description="List all documents grouped by domain",
        input_schema={"type": "object", "properties": {}}, handler=_list_documents,
    ))
    return server


_client = None
def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def ir_agent_execute(task_description):
    mcp = create_ir_mcp_server()
    domain = None
    domain_keywords = {
        "modem": ["modem", "5g", "nr", "lte", "vonr", "handover", "rrc", "sip", "通話", "信號"],
        "power": ["power", "thermal", "功耗", "耗電", "電池", "充電", "pmic", "throttl", "idle", "待機"],
        "connectivity": ["wifi", "wi-fi", "bluetooth", "bt", "gps", "mlo", "連線", "斷線"],
        "build": ["build", "patch", "打包", "release", "gerrit", "jenkins", "ci", "交付"],
    }
    task_lower = task_description.lower()
    for d, keywords in domain_keywords.items():
        if any(kw in task_lower for kw in keywords):
            domain = d
            break
    search_params = {"query": task_description, "n_results": 4}
    if domain:
        search_params["domain"] = domain
    search_resp = mcp.call_tool("search_documents", search_params)
    search_results = search_resp.result
    if not search_results.get("results"):
        return {"agent": "ir_agent", "error": "No results found", "detected_domain": domain or "general"}
    docs_text = ""
    for doc in search_results["results"]:
        docs_text += f"\n--- {doc['title']} (Domain: {doc['domain']}) ---\n{doc['content']}\n"
    llm = get_openai_client()
    prompt = f"""你是一位資深工程師兼知識管理專家。根據以下從部門知識庫檢索到的文件，針對同仁的問題提供精準的回答。

同仁的問題：{task_description}

檢索到的相關知識文件：
{docs_text}

回答要求：
1. 用繁體中文回答，語氣像資深同事在教你
2. 如果有具體排查步驟，按順序列出
3. 如果有相關 CR 或 Release Note，標注出來
4. 如果有踩坑紀錄，特別提醒
5. 標註知識來源（哪份文件、哪位作者）
6. 如果知識庫沒有直接答案，誠實說明並建議找誰問"""
    response = llm.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1500)
    return {
        "agent": "ir_agent", "detected_domain": domain or "general",
        "retrieved_docs": [{"title": d["title"], "category": d["category"], "domain": d["domain"]} for d in search_results["results"]],
        "summary": response.choices[0].message.content.strip(), "mcp_log": mcp.call_log,
    }


if __name__ == "__main__":
    init_vector_db()
    result = ir_agent_execute("5G NR handover 失敗要怎麼排查？")
    print(json.dumps(result, indent=2, ensure_ascii=False))
