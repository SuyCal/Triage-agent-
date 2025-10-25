from flask import Flask, request, jsonify
from datetime import datetime
from collections import defaultdict
import re

app = Flask(__name__)

# --- Tiny "AI" rules engine (placeholder for a real model) ---
# You could replace this with a trained classifier later.
CATEGORIES = {
    "billing":  ["invoice", "billing", "refund", "charge", "payment", "credit"],
    "outage":   ["down", "outage", "unreachable", "offline", "downtime"],
    "security": ["breach", "hacked", "compromised", "phishing", "ransom", "leak"],
    "account":  ["login", "password", "account", "2fa", "mfa", "locked"],
    "usage":    ["how to", "guide", "setup", "documentation", "question", "help"],
    "performance": ["slow", "latency", "timeout", "lag", "sluggish"],
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

def classify(text: str) -> str:
    t = text.lower()
    # First match strongest category by keyword counts
    scores = defaultdict(int)
    for cat, kws in CATEGORIES.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", t):
                scores[cat] += 1
    if not scores:
        return "other"
    # choose category with max hits
    return max(scores.items(), key=lambda x: x[1])[0]

def compute_priority(category: str, urgency: str, text: str) -> int:
    # Base by category
    pr = BASE_PRIORITY.get(category, 4)
    # Adjust by urgency
    urgency = (urgency or "").lower()
    if urgency in ("high", "urgent", "p1"):
        pr = min(pr, 1)
    elif urgency in ("medium", "normal"):
        pr = min(pr, 2) if pr > 2 else pr
    else:
        pr = pr
    # Escalation keywords
    t = text.lower()
    if any(k in t for k in ESCALATE):
        pr = 1
    return pr

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

@app.route("/triage", methods=["POST"])
def triage():
    data = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    urgency = (data.get("urgency") or "low").strip()

    text = f"{title}\n{description}"
    category = classify(text)
    priority = compute_priority(category, urgency, text)
    sla_hours = suggest_sla_hours(priority)
    auto_resolve = can_auto_resolve(category, text)
    route = ROUTING.get(category, "Support-L1")

    resp = {
        "received_at": datetime.utcnow().isoformat() + "Z",
        "category": category,
        "priority": priority,  # 1=highest, 4=lowest
        "routing_queue": route,
        "suggested_sla_hours": sla_hours,
        "auto_resolve_candidate": auto_resolve,
        "suggestions": SUGGESTIONS.get(category, SUGGESTIONS["other"]),
        "kb_link": kb_links(category)
    }
    return jsonify(resp), 200

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}

if __name__ == "__main__":
    # For local dev; use a proper WSGI server in production
    app.run(host="0.0.0.0", port=5000, debug=True)
