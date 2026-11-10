"""Decides whether a request goes to the small in-house model or the frontier model.

The decision is rule-based and explainable: every route returns the reasons it chose. Rules are cheap and predictable, and
they can be tuned against the evaluation set before any model is trained.
"""
import re
from dataclasses import dataclass, field

COMPLEX_TERMS = {
    "architecture", "refactor", "debug", "race condition", "security", "vulnerability", "prove", "optimize",
    "migrate", "multi-file", "concurrency", "distributed", "memory leak", "performance", "design a system",
}
CODE_FENCE = re.compile(r"```")


@dataclass(frozen=True)
class Decision:
    tier: str  # "small" or "frontier"
    score: int
    reasons: list[str] = field(default_factory=list)


def classify(message: str, tool_count: int = 0, threshold: int = 3) -> Decision:
    text = message.lower()
    score = 0
    reasons: list[str] = []

    if len(message) > 1200:
        score += 2
        reasons.append("long request (+2)")
    elif len(message) > 400:
        score += 1
        reasons.append("medium-length request (+1)")

    matched = sorted(term for term in COMPLEX_TERMS if term in text)
    if matched:
        term_score = 2 if len(matched) == 1 else 3
        score += term_score
        reasons.append(f"complex terms: {', '.join(matched)} (+{term_score})")

    if CODE_FENCE.search(message) or len(re.findall(r"\bdef |\bclass |\bfunction\b|;\s*$", message, re.M)) >= 2:
        score += 2
        reasons.append("contains code (+2)")

    if tool_count >= 2:
        score += 3
        reasons.append(f"{tool_count} tools requested (+3)")

    if not reasons:
        reasons.append("short, simple request")

    tier = "frontier" if score >= threshold else "small"
    return Decision(tier=tier, score=score, reasons=reasons)
