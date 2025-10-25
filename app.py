from flask import Flask, request, jsonify
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import re
from flask_cors import CORS  # <-- add

app = Flask(__name__)
CORS(app)  # <-- allow requests from your frontend (dev-friendly)


app = Flask(__name__)

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


def infer_urgency(text: str, category: str, user_urgency: str) -> Dict[str, object]:
    """Infer urgency level from free text, category, and provided urgency."""

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

    # Provided urgency acts as a hint, not the final truth
    mapped_user = (user_urgency or "").lower()
    if mapped_user in {"high", "urgent", "p1"}:
        score += 3
        signals.append("Requester marked as high urgency")
    elif mapped_user in {"medium", "normal"}:
        score += 1
        signals.append("Requester marked as medium urgency")

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


@app.route("/triage", methods=["POST"])
def triage():
    data = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    urgency = (data.get("urgency") or "low").strip()

    text = f"{title}\n{description}"

    category_details = analyze_category(text)
    category = category_details["category"]
    urgency_details = infer_urgency(text, category, urgency)
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
    app.run(host="0.0.0.0", port=5000, debug=True)