"""
AgentMaster PoC — Main Application
Flask API + Orchestrator Agent + Web UI

Architecture Reference:
  - AgentMaster (Stanford / George Mason University, 2025)
  - A2A Protocol (Google) + MCP Protocol (Anthropic)

No OpenClaw dependency — pure Python/Flask implementation.
"""

import os
import json
import time
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

from a2a_protocol import A2ARouter, A2ATask, A2AMessage, MessageRole, TaskState
from sql_agent import sql_agent_execute, init_database
from ir_agent import ir_agent_execute, init_vector_db, upload_knowledge, list_knowledge_by_domain

# ============================================================
# Flask App Setup
# ============================================================

app = Flask(__name__)
client = OpenAI()

# ============================================================
# A2A Router — Register Domain Agents
# ============================================================

router = A2ARouter()
router.register_agent("sql_agent", sql_agent_execute)
router.register_agent("ir_agent", ir_agent_execute)


# ============================================================
# Orchestrator Agent (主管 AI)
# ============================================================

def assess_complexity(user_query: str) -> dict:
    """
    Orchestrator Step 1: Assess query complexity and decompose tasks.
    Returns a plan with sub-tasks assigned to domain agents.
    """
    prompt = f"""你是一個多代理系統的協調者（Orchestrator）。你的工作是分析使用者的問題，判斷需要哪些專家代理來處理。

可用的專家代理：
1. sql_agent — 擅長查詢結構化數據（客戶業績、CR 分析、統計數據）。它可以存取 SQLite 資料庫，內含：
   - customers 表：id, name, region, sales_amount, return_count, last_order_date, account_manager
   - change_requests 表：cr_id, title, module, severity, status, created_date, resolved_date, assignee, root_cause, resolution_time_days
2. ir_agent — 擅長搜尋非結構化文件（公司政策、Release Note、技術規範、設計指南）。它可以存取向量資料庫中的政策文件和技術文件。

使用者問題：「{user_query}」

請分析此問題並以下列 JSON 格式回覆（只輸出 JSON，不要其他文字）：
{{
  "complexity": "simple|complex",
  "reasoning": "你的分析理由（繁體中文）",
  "tasks": [
    {{
      "agent": "sql_agent 或 ir_agent",
      "description": "這個代理需要做什麼（具體描述，讓代理能理解並執行）"
    }}
  ]
}}

判斷規則：
- 如果問題只需要一個代理就能回答，complexity = "simple"
- 如果問題需要兩個代理協作（例如先查數據再查政策），complexity = "complex"
- 每個 task 的 description 要具體，讓代理知道要查什麼"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800,
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        plan = json.loads(content)
    except json.JSONDecodeError:
        plan = {
            "complexity": "simple",
            "reasoning": "無法解析 LLM 回覆，使用預設單代理模式",
            "tasks": [{"agent": "ir_agent", "description": user_query}],
        }

    return plan


def synthesize_results(user_query: str, task_results: list) -> str:
    """
    Orchestrator Step 3: Synthesize results from all agents into a final report.
    """
    results_text = ""
    for tr in task_results:
        agent_name = tr.get("assigned_agent", "unknown")
        result = tr.get("result", {})
        results_text += f"\n=== {agent_name} 的回報 ===\n{json.dumps(result, indent=2, ensure_ascii=False)}\n"

    prompt = f"""你是一個多代理系統的協調者。以下是各專家代理針對使用者問題的回報結果。
請將這些碎片化的資訊整合成一份清晰、完整、專業的繁體中文報告。

使用者原始問題：「{user_query}」

各代理回報結果：
{results_text}

整合要求：
1. 先回答使用者的核心問題
2. 如果有數據查詢結果，用表格或列表呈現
3. 如果有政策/文件摘要，標註來源文件
4. 結尾提供簡短的行動建議
5. 語氣專業但易懂，適合給主管看"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )

    return response.choices[0].message.content.strip()


def orchestrate(user_query: str) -> dict:
    """
    Full Orchestrator workflow:
    1. Assess complexity & decompose tasks
    2. Dispatch tasks to agents via A2A
    3. Synthesize final report
    """
    start_time = time.time()
    workflow_log = {"query": user_query, "steps": []}

    # Step 1: Task decomposition
    plan = assess_complexity(user_query)
    workflow_log["steps"].append({
        "step": "1_task_decomposition",
        "complexity": plan["complexity"],
        "reasoning": plan["reasoning"],
        "task_count": len(plan["tasks"]),
    })

    # Step 2: Dispatch via A2A
    completed_tasks = []
    for task_spec in plan["tasks"]:
        a2a_task = A2ATask(
            assigned_agent=task_spec["agent"],
            description=task_spec["description"],
        )
        result_task = router.dispatch(a2a_task)
        completed_tasks.append(result_task.to_dict())

        workflow_log["steps"].append({
            "step": f"2_a2a_dispatch_{task_spec['agent']}",
            "task_id": result_task.task_id,
            "state": result_task.state.value if isinstance(result_task.state, TaskState) else result_task.state,
            "a2a_messages": len(result_task.messages),
        })

    # Step 3: Synthesize
    final_report = synthesize_results(user_query, completed_tasks)
    elapsed = round(time.time() - start_time, 2)

    workflow_log["steps"].append({
        "step": "3_synthesis",
        "elapsed_seconds": elapsed,
    })

    return {
        "query": user_query,
        "report": final_report,
        "plan": plan,
        "agent_results": completed_tasks,
        "workflow_log": workflow_log,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# Flask Routes
# ============================================================

@app.route("/")
def index():
    return render_template_string(WEB_UI_HTML)


@app.route("/api/query", methods=["POST"])
def api_query():
    """Main API endpoint — accepts natural language query"""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    result = orchestrate(data["query"])
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "agents": list(router.agents.keys()),
        "model": "gpt-4o-mini",
        "protocols": ["A2A", "MCP"],
    })


@app.route("/api/knowledge/upload", methods=["POST"])
def api_upload_knowledge():
    data = request.get_json()
    required = ["title", "content", "domain"]
    if not data or not all(k in data for k in required):
        return jsonify({"error": "Missing required fields: title, content, domain"}), 400
    result = upload_knowledge(
        title=data["title"], content=data["content"],
        domain=data["domain"], author=data.get("author", "Anonymous"),
    )
    return jsonify(result)


@app.route("/api/knowledge/list")
def api_list_knowledge():
    return jsonify(list_knowledge_by_domain())


@app.route("/api/knowledge/preview")
def api_preview_knowledge():
    doc_id = request.args.get("id", "")
    if not doc_id:
        return jsonify({"error": "Missing id parameter"}), 400
    import chromadb
    from ir_agent import CHROMA_DIR
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("policy_docs")
        result = collection.get(ids=[doc_id])
        if result["ids"]:
            return jsonify({
                "id": result["ids"][0],
                "title": result["metadatas"][0].get("title", ""),
                "content": result["documents"][0],
                "domain": result["metadatas"][0].get("domain", ""),
            })
        return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Web UI (Single-file HTML)
# ============================================================

WEB_UI_HTML = r"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentMaster PoC — Multi-Agent System</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2d3148; --text: #e4e6f0; --text2: #9498b3;
    --accent: #6c8cff; --accent2: #4a6cf7; --green: #4ade80;
    --amber: #fbbf24; --coral: #f87171; --teal: #2dd4bf;
    --purple: #a78bfa; --radius: 12px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

  .container { max-width: 960px; margin: 0 auto; padding: 24px 16px; }
  h1 { font-size: 24px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--text2); font-size: 14px; margin-bottom: 24px; }

  /* Protocol badges */
  .badges { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  .badge { padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 500; }
  .badge-a2a { background: rgba(108,140,255,0.15); color: var(--accent); border: 1px solid rgba(108,140,255,0.3); }
  .badge-mcp { background: rgba(167,139,250,0.15); color: var(--purple); border: 1px solid rgba(167,139,250,0.3); }
  .badge-agent { background: rgba(45,212,191,0.15); color: var(--teal); border: 1px solid rgba(45,212,191,0.3); }

  /* Input area */
  .input-section { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; margin-bottom: 16px; }
  .input-row { display: flex; gap: 8px; }
  textarea { flex: 1; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 12px; font-size: 14px; resize: vertical; min-height: 60px; font-family: inherit; }
  textarea:focus { outline: none; border-color: var(--accent); }
  button { padding: 12px 24px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: var(--accent2); color: #fff; }
  .btn-primary:hover { background: var(--accent); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Quick examples */
  .examples { display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap; }
  .example-btn { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text2); padding: 6px 12px; font-size: 12px; cursor: pointer; transition: all 0.15s; }
  .example-btn:hover { border-color: var(--accent); color: var(--text); }

  /* Results */
  .result-section { display: none; }
  .result-section.show { display: block; }

  .report-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 16px; }
  .report-card h3 { font-size: 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .report-content { line-height: 1.7; white-space: pre-wrap; font-size: 14px; }

  /* Workflow steps */
  .workflow { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
  .step { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; }
  .step-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .step-name { font-weight: 600; font-size: 13px; }
  .step-badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; }
  .step-detail { color: var(--text2); font-size: 12px; }

  /* Agent detail panels */
  .agent-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  @media (max-width: 640px) { .agent-panels { grid-template-columns: 1fr; } }
  .agent-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; }
  .agent-panel h4 { font-size: 14px; margin-bottom: 8px; }
  .agent-panel pre { background: var(--surface2); border-radius: 6px; padding: 10px; font-size: 11px; overflow-x: auto; color: var(--text2); line-height: 1.5; max-height: 300px; overflow-y: auto; }

  /* Loading */
  .loading { text-align: center; padding: 40px; color: var(--text2); }
  .spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; margin-bottom: 12px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .elapsed { color: var(--green); font-size: 13px; }
</style>
</head>
<body>
<div class="container">
  <h1>🤖 AgentMaster PoC</h1>
  <p class="subtitle">Multi-Agent System with A2A + MCP Protocols — Pure Python/Flask</p>

  <div class="badges">
    <span class="badge badge-a2a">A2A Protocol</span>
    <span class="badge badge-mcp">MCP Protocol</span>
    <span class="badge badge-agent">Orchestrator + SQL Agent + IR Agent</span>
  </div>

  <div class="input-section">
    <div class="input-row">
      <textarea id="queryInput" placeholder="輸入你的問題...（支援跨系統複合查詢）" rows="2"></textarea>
      <button class="btn-primary" id="submitBtn" onclick="submitQuery()">發送</button>
    </div>
    <div class="examples">
      <button class="example-btn" onclick="setQuery('幫我找出上個月業績最差的三個客戶，並從退換貨管理辦法中總結出他們可能遇到的違約金規定')">📊 場景1: 業績+政策</button>
      <button class="example-btn" onclick="setQuery('列出所有 Critical 等級且尚未解決的 CR，並從技術文件中找出相關的設計規範和修復指引')">🔧 場景2: CR分析</button>
      <button class="example-btn" onclick="setQuery('分析 Modem 模組的 CR 趨勢，平均修復時間是多少？最新的 Release Note 有修了哪些問題？')">📋 場景2: Release Note</button>
      <button class="example-btn" onclick="setQuery('退貨率最高的客戶是誰？他們的信用評等會受到什麼影響？')">📑 場景1: 信用評等</button>
    </div>
    <div class="examples" style="margin-top:4px">
      <button class="example-btn" style="border-color:#2dd4bf" onclick="setQuery('5G NR handover 失敗要怎麼排查？有哪些常見地雷？')">🧠 場景3: Modem Know-How</button>
      <button class="example-btn" style="border-color:#2dd4bf" onclick="setQuery('手機待機功耗偏高，要怎麼一步步排查？')">🔋 場景3: Power Know-How</button>
      <button class="example-btn" style="border-color:#2dd4bf" onclick="setQuery('Wi-Fi 7 MLO 連線一直斷，怎麼 debug？')">📡 場景3: Connectivity Know-How</button>
      <button class="example-btn" style="border-color:#2dd4bf" onclick="setQuery('怎麼打包 patch 給客戶？完整流程是什麼？')">📦 場景3: Build Know-How</button>
    </div>
  </div>

  <div class="input-section" style="margin-bottom:16px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <h3 style="font-size:15px;font-weight:600">📤 上傳 Domain Know-How</h3>
      <button class="example-btn" onclick="loadKnowledgeList()" style="font-size:11px">查看知識庫</button>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px">
      <input id="khTitle" placeholder="標題" style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px;font-size:13px">
      <select id="khDomain" style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px;font-size:13px">
        <option value="modem">Modem</option><option value="power">Power</option>
        <option value="connectivity">Connectivity</option><option value="build">Build/Release</option>
        <option value="general">General</option>
      </select>
      <input id="khAuthor" placeholder="作者" style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px;font-size:13px">
    </div>
    <textarea id="khContent" placeholder="貼上你的 Know-How 內容..." rows="3" style="width:100%;margin-bottom:8px"></textarea>
    <div style="display:flex;gap:8px;align-items:center">
      <button class="btn-primary" onclick="uploadKnowledge()" style="padding:8px 16px;font-size:13px">上傳到知識庫</button>
      <span id="uploadStatus" style="font-size:12px;color:var(--green)"></span>
    </div>
    <div id="knowledgeList" style="display:none;margin-top:12px;background:var(--surface2);border-radius:8px;padding:12px;font-size:12px;max-height:200px;overflow-y:auto"></div>
  </div>

  <!-- Loading -->
  <div id="loadingArea" class="loading" style="display:none">
    <div class="spinner"></div>
    <p id="loadingText">Orchestrator 正在分析問題複雜度...</p>
  </div>

  <!-- Results -->
  <div id="resultArea" class="result-section">
    <!-- Workflow -->
    <div id="workflowSteps" class="workflow"></div>

    <!-- Final Report -->
    <div class="report-card">
      <h3>📝 整合報告 <span class="elapsed" id="elapsedTime"></span></h3>
      <div class="report-content" id="reportContent"></div>
    </div>

    <!-- Agent Detail Panels -->
    <div class="agent-panels" id="agentPanels"></div>
  </div>
</div>

<script>
function setQuery(q) {
  document.getElementById('queryInput').value = q;
}

async function submitQuery() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const btn = document.getElementById('submitBtn');
  btn.disabled = true;
  document.getElementById('loadingArea').style.display = 'block';
  document.getElementById('resultArea').classList.remove('show');

  const steps = ['Orchestrator 正在分析問題複雜度...', '透過 A2A 分派任務給專家代理...', 'SQL Agent 正在查詢資料庫...', 'IR Agent 正在搜尋文件庫...', '正在整合各代理回報結果...'];
  let si = 0;
  const timer = setInterval(() => {
    si = Math.min(si + 1, steps.length - 1);
    document.getElementById('loadingText').textContent = steps[si];
  }, 2000);

  try {
    const resp = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query}),
    });
    const data = await resp.json();
    clearInterval(timer);
    renderResult(data);
  } catch (e) {
    clearInterval(timer);
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    document.getElementById('loadingArea').style.display = 'none';
  }
}

function renderResult(data) {
  // Workflow steps
  const plan = data.plan || {};
  const wf = document.getElementById('workflowSteps');
  wf.innerHTML = '';

  // Step 1: Decomposition
  wf.innerHTML += `<div class="step">
    <div class="step-header">
      <span class="step-name">Step 1: 任務拆解</span>
      <span class="step-badge" style="background:${plan.complexity==='complex'?'rgba(251,191,36,0.2);color:#fbbf24':'rgba(74,222,128,0.2);color:#4ade80'}">${plan.complexity==='complex'?'🔥 複雜查詢（多代理協作）':'✅ 簡單查詢'}</span>
    </div>
    <div class="step-detail">${plan.reasoning||''}</div>
  </div>`;

  // Step 2: A2A dispatch
  (plan.tasks||[]).forEach((t,i) => {
    const icon = t.agent === 'sql_agent' ? '🗄️' : '📚';
    const color = t.agent === 'sql_agent' ? 'rgba(45,212,191,0.2);color:#2dd4bf' : 'rgba(167,139,250,0.2);color:#a78bfa';
    wf.innerHTML += `<div class="step">
      <div class="step-header">
        <span class="step-name">Step 2.${i+1}: A2A → ${t.agent}</span>
        <span class="step-badge" style="background:${color}">${icon} ${t.agent}</span>
      </div>
      <div class="step-detail">${t.description}</div>
    </div>`;
  });

  // Step 3: Synthesis
  wf.innerHTML += `<div class="step">
    <div class="step-header">
      <span class="step-name">Step 3: 結果整合</span>
      <span class="step-badge" style="background:rgba(108,140,255,0.2);color:#6c8cff">🤖 Orchestrator</span>
    </div>
    <div class="step-detail">將各代理回報統整為最終報告</div>
  </div>`;

  // Report
  document.getElementById('reportContent').textContent = data.report || '';
  document.getElementById('elapsedTime').textContent = `⏱️ ${data.elapsed_seconds}s`;

  // Agent panels
  const panels = document.getElementById('agentPanels');
  panels.innerHTML = '';
  (data.agent_results||[]).forEach(ar => {
    const result = ar.result || {};
    const agent = ar.assigned_agent;
    const icon = agent === 'sql_agent' ? '🗄️' : '📚';
    const title = agent === 'sql_agent' ? 'SQL Agent 詳細回報' : 'IR Agent 詳細回報';

    let detail = '';
    if (agent === 'sql_agent' && result.generated_sql) {
      detail = `-- Generated SQL:\n${result.generated_sql}\n\n-- Query Result:\n${JSON.stringify(result.data, null, 2)}`;
    } else if (agent === 'ir_agent') {
      detail = `-- Retrieved Docs:\n${JSON.stringify(result.retrieved_docs, null, 2)}\n\n-- Summary:\n${result.summary||''}`;
    } else {
      detail = JSON.stringify(result, null, 2);
    }

    panels.innerHTML += `<div class="agent-panel">
      <h4>${icon} ${title}</h4>
      <div style="margin-bottom:6px"><span class="badge badge-a2a" style="font-size:10px">A2A Messages: ${ar.messages?.length||0}</span></div>
      <pre>${escapeHtml(detail)}</pre>
    </div>`;
  });

  document.getElementById('resultArea').classList.add('show');
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function uploadKnowledge() {
  const title = document.getElementById('khTitle').value.trim();
  const domain = document.getElementById('khDomain').value;
  const author = document.getElementById('khAuthor').value.trim() || 'Anonymous';
  const content = document.getElementById('khContent').value.trim();
  if (!title || !content) { alert('請填寫標題和內容'); return; }
  const status = document.getElementById('uploadStatus');
  status.textContent = '上傳中...'; status.style.color = '#fbbf24';
  try {
    const resp = await fetch('/api/knowledge/upload', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({title, content, domain, author}),
    });
    const data = await resp.json();
    if (data.status === 'success') {
      status.textContent = '✅ ' + data.message; status.style.color = '#4ade80';
      document.getElementById('khTitle').value = '';
      document.getElementById('khContent').value = '';
      document.getElementById('knowledgeList').style.display = 'none';
      loadKnowledgeList();
    } else { status.textContent = '❌ ' + (data.error||'Failed'); status.style.color = '#f87171'; }
  } catch(e) { status.textContent = '❌ ' + e.message; status.style.color = '#f87171'; }
}

async function loadKnowledgeList() {
  const el = document.getElementById('knowledgeList');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
  if (el.style.display === 'none') return;
  el.innerHTML = 'Loading...';
  try {
    const resp = await fetch('/api/knowledge/list');
    const data = await resp.json();
    let html = '<strong>知識庫總覽（' + data.total + ' 篇）</strong><br><br>';
    for (const [domain, docs] of Object.entries(data.domains || {})) {
      html += '<div style="margin-bottom:10px"><strong style="color:#6c8cff">[' + domain.toUpperCase() + ']</strong> (' + docs.length + ' 篇)<br>';
      docs.forEach(d => {
        const isNew = d.id.startsWith('knowhow-upload-');
        const badge = isNew ? ' <span style="background:#4ade80;color:#000;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600">NEW</span>' : '';
        const clickHint = ' <span style="color:#9498b3;cursor:pointer" onclick="previewDoc(\'' + d.id + '\')">[預覽]</span>';
        html += '&nbsp;&nbsp;• ' + d.title + badge + clickHint + '<br>';
      });
      html += '</div>';
    }
    el.innerHTML = html;
  } catch(e) { el.innerHTML = 'Error: ' + e.message; }
}

async function previewDoc(docId) {
  try {
    const resp = await fetch('/api/knowledge/preview?id=' + encodeURIComponent(docId));
    const data = await resp.json();
    if (data.content) {
      const el = document.getElementById('knowledgeList');
      el.innerHTML += '<div style="margin-top:10px;padding:10px;background:#1a1d27;border:1px solid #2d3148;border-radius:6px;white-space:pre-wrap;font-size:12px;line-height:1.6"><strong>' + data.title + '</strong><br><br>' + data.content.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>';
    }
  } catch(e) { alert('Preview failed: ' + e.message); }
}

document.getElementById('queryInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitQuery(); }
});
</script>
</body>
</html>
"""


# ============================================================
# Startup
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AgentMaster PoC — Multi-Agent System")
    print("  Protocols: A2A (Google) + MCP (Anthropic)")
    print("  Agents: Orchestrator + SQL Agent + IR Agent")
    print("=" * 60)

    # Initialize demo data
    print("\n📦 Initializing demo databases...")
    init_database()
    init_vector_db()

    print("\n🚀 Starting Flask server...")
    print("   Web UI: http://localhost:5000")
    print("   API:    http://localhost:5000/api/query")
    print("   Health: http://localhost:5000/api/health")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
