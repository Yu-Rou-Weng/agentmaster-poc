"""
Lightweight Vector DB — replaces ChromaDB for low-memory environments.
Uses OpenAI embeddings + cosine similarity. Stores data in a JSON file.
Zero heavy dependencies, works fine on 512MB RAM.
"""
import os, json, math

DATA_FILE = os.path.join(os.path.dirname(__file__), "knowledge_store.json")


def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _get_embedding(text):
    """Get embedding from OpenAI API (text-embedding-3-small is cheap and fast)"""
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
    return resp.data[0].embedding


def _load_store():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": {}}


def _save_store(store):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)


def init_collection(documents: list[dict]):
    """Initialize the store with a list of documents (each has id, title, content, category, domain)"""
    store = {"documents": {}}
    for doc in documents:
        emb = _get_embedding(doc["content"][:2000])
        store["documents"][doc["id"]] = {
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "domain": doc.get("domain", "general"),
            "embedding": emb,
        }
    _save_store(store)
    return len(store["documents"])


def add_document(doc_id: str, title: str, content: str, category: str, domain: str):
    """Add a single document to the store"""
    store = _load_store()
    emb = _get_embedding(content[:2000])
    store["documents"][doc_id] = {
        "title": title, "content": content,
        "category": category, "domain": domain, "embedding": emb,
    }
    _save_store(store)


def search(query: str, n_results: int = 3, domain: str = None) -> list[dict]:
    """Search by cosine similarity, optionally filtered by domain"""
    store = _load_store()
    if not store["documents"]:
        return []
    q_emb = _get_embedding(query)
    scored = []
    for doc_id, doc in store["documents"].items():
        if domain and domain != "all" and doc["domain"] != domain:
            continue
        sim = _cosine_sim(q_emb, doc["embedding"])
        scored.append({"id": doc_id, "score": sim, **{k: v for k, v in doc.items() if k != "embedding"}})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_results]


def list_all() -> dict:
    """List all documents grouped by domain"""
    store = _load_store()
    domains = {}
    for doc_id, doc in store["documents"].items():
        d = doc.get("domain", "general")
        if d not in domains:
            domains[d] = []
        domains[d].append({"id": doc_id, "title": doc["title"], "category": doc["category"]})
    return {"domains": domains, "total": len(store["documents"])}


def get_document(doc_id: str) -> dict:
    """Get a single document by ID"""
    store = _load_store()
    doc = store["documents"].get(doc_id)
    if doc:
        return {"id": doc_id, "title": doc["title"], "content": doc["content"], "domain": doc["domain"]}
    return None


def is_initialized() -> bool:
    """Check if the store has data"""
    if not os.path.exists(DATA_FILE):
        return False
    store = _load_store()
    return len(store["documents"]) > 0
