from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import re
from flask_cors import CORS  # <-- add
import sqlite3
import threading
import random
import string

app = Flask(__name__)
CORS(app)  # <-- allow requests from your frontend (dev-friendly)


app = Flask(__name__)

# --- SQLite setup ---
DB_PATH = "tickets.db"
_db_lock = threading.Lock()


def init_db():
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    ticket_ref TEXT UNIQUE,
                    user_category TEXT,
                    category TEXT NOT NULL,
                    category_confidence REAL,
                    inferred_urgency TEXT,
                    urgency_score INTEGER,
                    priority INTEGER,
                    priority_rationale TEXT,
                    routing_queue TEXT,
                    suggested_sla_hours INTEGER,
                    auto_resolve_candidate INTEGER,
                    kb_link TEXT,
                    status TEXT DEFAULT 'open',
                    assigned_to TEXT
                )
                """
            )
            conn.commit()
            # Migration: ensure ticket_ref column exists for existing DBs
            # Migration for existing DBs: add column, then unique index
            cols = {r[1] for r in conn.execute("PRAGMA table_info(tickets)").fetchall()}
            if "ticket_ref" not in cols:
                # 1) Add column without UNIQUE (SQLite limitation)
                conn.execute("ALTER TABLE tickets ADD COLUMN ticket_ref TEXT")
                # 2) Create UNIQUE index (enforces uniqueness for non-null values)
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tickets_ticket_ref ON tickets(ticket_ref)")
                # 3) Optional: backfill refs for existing rows
                rows = conn.execute("SELECT id, category FROM tickets WHERE ticket_ref IS NULL").fetchall()
                for tid, cat in rows:
                    ref = generate_ticket_ref(cat or "other")
                    # ensure uniqueness
                    while conn.execute("SELECT 1 FROM tickets WHERE ticket_ref = ?", (ref,)).fetchone():
                        ref = generate_ticket_ref(cat or "other")
                    conn.execute("UPDATE tickets SET ticket_ref = ? WHERE id = ?", (ref, tid))
                conn.commit()
        finally:
            conn.close()



def db_execute(query: str, params: Tuple = ()):  # type: ignore[valid-type]
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.execute(query, params)
            conn.commit()
            return cur
        finally:
            conn.close()


def db_query(query: str, params: Tuple = ()):  # type: ignore[valid-type]
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            return rows
        finally:
            conn.close()


def _category_abbrev(category: str) -> str:
    m = {
        "billing": "Bil",
        "outage": "Out",
        "security": "Sec",
        "account": "Acc",
        "usage": "Use",
        "performance": "Per",
        "other": "Oth",
    }
    return m.get((category or "").lower(), "Oth")


def _random_code(n: int = 4) -> str:
    alphabet = string.ascii_uppercase
    return "".join(random.choice(alphabet) for _ in range(n))


def generate_ticket_ref(category: str) -> str:
    prefix = _category_abbrev(category)
    return f"{prefix}-{_random_code(4)}"


# --- Lightweight heuristics "AI" engine ---
# These keyword maps act as a deterministic fallback until you swap in a model.
CATEGORIES = {
    "billing": [
        "invoice",
        "billing",
        "refund",
        "charge",
        "payment",
        "credit",
        "card declined",
        "receipt",
        "subscription",
    ],
    "outage": [
        "down",
        "outage",
        "unreachable",
        "offline",
        "downtime",
        "major outage",
        "service down",
        "cannot connect",
    ],
    "security": [
        "breach",
        "hacked",
        "compromised",
        "phishing",
        "ransom",
        "leak",
        "unauthorized",
        "intrusion",
        "security incident",
    ],
    "account": [
        "login",
        "password",
        "account",
        "2fa",
        "mfa",
        "locked",
        "reset",
        "credential",
        "access",
    ],
    "usage": [
        "how to",
        "guide",
        "setup",
        "documentation",
        "question",
        "help",
        "feature",
        "walkthrough",
        "configure",
    ],
    "performance": [
        "slow",
        "latency",
        "timeout",
        "lag",
        "sluggish",
        "performance",
        "degraded",
        "high cpu",
    ],
}

SUGGESTIONS = {
    "billing": [
        "Verify recent invoices in billing portal.",
        "Check payment status and retry failed transactions.",
        "If duplicate charge, begin refund workflow."
    ],
    "outage": [
        "Check service status page and incident channel.",
        "Collect traceroute and ping outputs from user.",
        "Auto-create P1 incident if multiple reports in 15 min."
    ],
    "security": [
        "Force password reset and terminate active sessions.",
        "Open security incident; collect IOC details.",
        "Notify SecOps on-call immediately."
    ],
    "account": [
        "Guide user through password reset or MFA recovery.",
        "Verify identity via secure channel.",
        "Escalate if repeated lockouts > 3 in 24h."
    ],
    "usage": [
        "Provide relevant knowledge base article.",
        "Offer guided steps in-app.",
        "If unresolved, route to L1."
    ],
    "performance": [
        "Collect timestamps and region info.",
        "Check known latency events and resource metrics.",
        "If widespread, open problem record."
    ],
    "other": [
        "Acknowledge and request clarifying details.",
        "Suggest nearest KB article by keywords.",
        "Route to L1 queue."
    ]
}

ROUTING = {
    "billing": "Billing-Desk",
    "outage": "SRE-OnCall",
    "security": "SecOps-P1",
    "account": "Support-L1",
    "usage": "Support-L1",
    "performance": "SRE-Triage",
    "other": "Support-L1"
}

# Simple priority matrix
# Base priority by category; then adjusted by user-supplied urgency and keyword severity
BASE_PRIORITY = {
    "security": 1,  # P1 highest
    "outage": 1,
    "performance": 2,
    "billing": 3,
    "account": 3,
    "usage": 4,
    "other": 4
}

# Keywords that escalate priority
ESCALATE = ["urgent", "immediately", "critical", "p1", "down", "cannot", "breach", "outage"]

URGENCY_KEYWORDS = {
    "critical": [
        "production down",
        "prod down",
        "cannot login",
        "cannot access",
        "all users",
        "everyone",
        "major outage",
        "security incident",
        "data loss",
        "ransom",
        "breach",
        "compromised",
        "urgent",
        "immediately",
        "asap",
        "critical",
        "p1",
        "sev1",
        "severe",
    ],
    "high": [
        "failing",
        "failed",
        "error",
        "unable",
        "customers",
        "payment",
        "blocked",
        "deadlock",
        "down",
        "offline",
        "outage",
        "breached",
        "escalate",
        "priority",
    ],
    "medium": [
        "bug",
        "issue",
        "broken",
        "delay",
        "slow",
        "latency",
        "timeout",
        "degraded",
        "need help",
        "support",
        "fix",
        "investigate",
    ],
    "low": [
        "feature request",
        "enhancement",
        "would like",
        "nice to have",
        "question",
        "curious",
        "suggestion",
    ],
}

URGENCY_WEIGHTS = {
    "critical": 5,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def analyze_category(text: str) -> Dict[str, object]:
    """Return category prediction along with supporting signals."""

    normalized = _normalize(text)
    scores: Dict[str, int] = defaultdict(int)
    matched_keywords: Dict[str, List[str]] = defaultdict(list)

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword)}\b" if " " not in keyword else keyword
            if re.search(pattern, normalized):
                scores[category] += 1
                matched_keywords[category].append(keyword)

    if not scores:
        return {
            "category": "other",
            "scores": {},
            "matched_keywords": [],
            "confidence": 0.25,
        }

    top_category, top_score = max(scores.items(), key=lambda x: x[1])
    total = sum(scores.values())
    confidence = top_score / total if total else 0.25

    return {
        "category": top_category,
        "scores": dict(scores),
        "matched_keywords": matched_keywords[top_category],
        "confidence": round(confidence, 2),
    }


def infer_urgency(text: str, category: str) -> Dict[str, object]:
    """Infer urgency level automatically from free text and category."""

    normalized = _normalize(text)
    score = 0
    signals: List[str] = []

    for level, keywords in URGENCY_KEYWORDS.items():
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword)}\b" if " " not in keyword else keyword
            if re.search(pattern, normalized):
                score += URGENCY_WEIGHTS[level]
                signals.append(f"Matched '{keyword}' → {level}")

    # Category-informed adjustments
    if category in {"security", "outage"}:
        score += 6
        signals.append(f"Category '{category}' implies critical response")
    elif category == "performance":
        score += 2
        signals.append("Performance issues need faster follow-up")

    # Punctuation / formatting heuristics
    if normalized.count("!") >= 2:
        score += 2
        signals.append("Multiple exclamation marks detected")

    # User-provided urgency is ignored; auto-infer only

    # Determine label from score thresholds
    if score >= 11:
        label = "critical"
    elif score >= 7:
        label = "high"
    elif score >= 4:
        label = "medium"
    else:
        label = "low"

    return {
        "label": label,
        "score": score,
        "signals": signals,
    }


def compute_priority(
    category: str,
    inferred_urgency: str,
    user_urgency: str,
    text: str,
) -> Tuple[int, str]:
    # Base by category
    priority = BASE_PRIORITY.get(category, 4)
    rationale: List[str] = [f"Category baseline priority {priority}"]

    urgency_map = {"critical": 1, "high": 2, "medium": 3, "low": 4}
    inferred_priority = urgency_map.get(inferred_urgency, 4)
    if inferred_priority < priority:
        rationale.append(
            f"Inferred urgency '{inferred_urgency}' tightens priority to {inferred_priority}"
        )
    priority = min(priority, inferred_priority)

    mapped_user = (user_urgency or "").lower()
    if mapped_user in {"high", "urgent", "p1"}:
        rationale.append("Requester asked for high urgency – honoring as P1")
        priority = min(priority, 1)
    elif mapped_user in {"medium", "normal"}:
        rationale.append("Requester indicated medium urgency")
        priority = min(priority, 2) if priority > 2 else priority

    t = text.lower()
    if any(k in t for k in ESCALATE):
        rationale.append("Escalation keyword found – forcing P1")
        priority = 1

    return priority, "; ".join(rationale)


def suggest_sla_hours(priority: int) -> int:
    # Simple SLA suggestion (tweak as needed)
    return {1: 4, 2: 12, 3: 24, 4: 48}.get(priority, 48)


def can_auto_resolve(category: str, text: str) -> bool:
    # Cheap heuristic: usage/how-to with known keywords could be auto-resolved
    return category in ("usage", "account") and len(text) > 40


def kb_links(category: str):
    # Stubbed KB—replace with your real URLs/IDs
    base = "https://kb.example.com/"
    return {
        "billing": base + "billing-invoices",
        "outage": base + "service-status",
        "security": base + "security-incident",
        "account": base + "account-access",
        "usage": base + "getting-started",
        "performance": base + "performance-troubleshooting",
        "other": base + "search"
    }.get(category, base + "search")


@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, PATCH, PUT, DELETE, OPTIONS")
    return resp


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


@app.route("/support", methods=["GET"])
def support_ui():
    return send_from_directory(".", "support.html")


@app.route("/tickets", methods=["POST"])
def create_ticket():
    data = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    user_category = (data.get("user_category") or "").strip().lower()

    if not title or not description:
        return jsonify({"error": "title and description are required"}), 400

    text = f"{title}\n{description}"

    category_details = analyze_category(text)
    category = category_details["category"]
    urgency_details = infer_urgency(text, category)
    priority, rationale = compute_priority(
        category, urgency_details["label"], "", text
    )
    sla_hours = suggest_sla_hours(priority)
    auto_resolve = can_auto_resolve(category, text)
    route = ROUTING.get(category, "Support-L1")

    created_at = datetime.utcnow().isoformat() + "Z"
    # Generate unique ticket_ref
    ticket_ref = generate_ticket_ref(category)
    for _ in range(10):
        exists = db_query("SELECT 1 FROM tickets WHERE ticket_ref = ?", (ticket_ref,))
        if not exists:
            break
        ticket_ref = generate_ticket_ref(category)
    cur = db_execute(
        """
        INSERT INTO tickets (
            created_at, title, description, ticket_ref, user_category,
            category, category_confidence, inferred_urgency, urgency_score,
            priority, priority_rationale, routing_queue, suggested_sla_hours,
            auto_resolve_candidate, kb_link, status, assigned_to
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL)
        """,
        (
            created_at,
            title,
            description,
            ticket_ref,
            user_category or None,
            category,
            category_details["confidence"],
            urgency_details["label"],
            int(urgency_details["score"]),
            int(priority),
            rationale,
            route,
            int(sla_hours),
            1 if auto_resolve else 0,
            kb_links(category),
        ),
    )
    ticket_id = cur.lastrowid if cur else None

    resp = {
        "id": ticket_id,
        "ticket_ref": ticket_ref,
        "received_at": created_at,
        "title": title,
        "description": description,
        "user_category": user_category or None,
        "category": category,
        "category_confidence": category_details["confidence"],
        "category_signals": category_details["matched_keywords"],
        "priority": priority,  # 1=highest, 4=lowest
        "inferred_urgency": urgency_details["label"],
        "urgency_score": urgency_details["score"],
        "urgency_signals": urgency_details["signals"],
        "routing_queue": route,
        "suggested_sla_hours": sla_hours,
        "auto_resolve_candidate": auto_resolve,
        "suggestions": SUGGESTIONS.get(category, SUGGESTIONS["other"]),
        "kb_link": kb_links(category),
        "priority_rationale": rationale,
        "status": "open",
    }
    return jsonify(resp), 201


@app.route("/tickets", methods=["GET"])
def list_tickets():
    rows = db_query("SELECT * FROM tickets ORDER BY datetime(created_at) DESC")
    items = []
    for r in rows:
        obj = {k: r[k] for k in r.keys()}
        obj["auto_resolve_candidate"] = bool(obj.get("auto_resolve_candidate"))
        items.append(obj)
    return jsonify({"items": items}), 200


@app.route("/tickets/<int:ticket_id>", methods=["PATCH", "PUT", "POST"])
def update_ticket(ticket_id: int):
    data = request.get_json(force=True, silent=True) or {}
    status = data.get("status")
    assigned_to = data.get("assigned_to")
    updates = []
    params: List = []
    if status:
        updates.append("status = ?")
        params.append(status)
    if assigned_to is not None:
        updates.append("assigned_to = ?")
        params.append(assigned_to)
    if not updates:
        return jsonify({"error": "no valid fields to update"}), 400
    params.append(ticket_id)
    db_execute(f"UPDATE tickets SET {', '.join(updates)} WHERE id = ?", tuple(params))
    row = db_query("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    if not row:
        return jsonify({"error": "ticket not found"}), 404
    obj = {k: row[0][k] for k in row[0].keys()}
    obj["auto_resolve_candidate"] = bool(obj.get("auto_resolve_candidate"))
    return jsonify(obj), 200


@app.route("/tickets/<int:ticket_id>", methods=["GET"])
def get_ticket(ticket_id: int):
    row = db_query("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    if not row:
        return jsonify({"error": "ticket not found"}), 404
    obj = {k: row[0][k] for k in row[0].keys()}
    obj["auto_resolve_candidate"] = bool(obj.get("auto_resolve_candidate"))
    return jsonify(obj), 200


@app.route("/tickets/<int:ticket_id>", methods=["DELETE"])
def delete_ticket(ticket_id: int):
    row = db_query("SELECT id FROM tickets WHERE id = ?", (ticket_id,))
    if not row:
        return jsonify({"error": "ticket not found"}), 404
    db_execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
    return "", 204

@app.route("/triage", methods=["POST"])
def triage():
    data = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    # Ignore any caller-provided urgency; auto-infer only
    urgency = ""

    text = f"{title}\n{description}"

    category_details = analyze_category(text)
    category = category_details["category"]
    urgency_details = infer_urgency(text, category)
    priority, rationale = compute_priority(
        category, urgency_details["label"], urgency, text
    )
    sla_hours = suggest_sla_hours(priority)
    auto_resolve = can_auto_resolve(category, text)
    route = ROUTING.get(category, "Support-L1")

    resp = {
        "received_at": datetime.utcnow().isoformat() + "Z",
        "category": category,
        "category_confidence": category_details["confidence"],
        "category_signals": category_details["matched_keywords"],
        "priority": priority,  # 1=highest, 4=lowest
        "inferred_urgency": urgency_details["label"],
        "urgency_score": urgency_details["score"],
        "urgency_signals": urgency_details["signals"],
        "routing_queue": route,
        "suggested_sla_hours": sla_hours,
        "auto_resolve_candidate": auto_resolve,
        "suggestions": SUGGESTIONS.get(category, SUGGESTIONS["other"]),
        "kb_link": kb_links(category),
        "priority_rationale": rationale,
    }
    return jsonify(resp), 200


@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}


if __name__ == "__main__":
    # For local dev; use a proper WSGI server in production
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
