"""Routes each request to a tier, falls back to the frontier model when the small model fails or looks unreliable,
and records latency, tokens and estimated cost for every decision."""
from dataclasses import asdict, dataclass, field

from .backends import BackendError, OpenAICompatibleBackend, Reply
from .classify import classify


@dataclass(frozen=True)
class Prices:
    """USD per 1 million tokens. Set these from your provider's current price list."""
    input_per_m: float = 0.15
    output_per_m: float = 0.60

    def cost(self, reply: Reply) -> float:
        return (reply.prompt_tokens * self.input_per_m + reply.completion_tokens * self.output_per_m) / 1_000_000


@dataclass
class RouteRecord:
    tier: str
    model: str
    fallback_used: bool
    score: int
    reasons: list[str]
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    answer: str = ""


@dataclass
class HybridRouter:
    frontier: OpenAICompatibleBackend
    small: OpenAICompatibleBackend | None = None
    prices: Prices = field(default_factory=Prices)
    threshold: int = 3
    min_small_chars: int = 2

    def _record(self, tier: str, backend, reply: Reply, fallback: bool, score: int, reasons: list[str]) -> RouteRecord:
        return RouteRecord(
            tier=tier,
            model=backend.model,
            fallback_used=fallback,
            score=score,
            reasons=reasons,
            latency_s=round(reply.latency_s, 4),
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            estimated_cost_usd=round(self.prices.cost(reply), 8),
            answer=reply.text,
        )

    def route(self, messages: list[dict], tool_count: int = 0) -> RouteRecord:
        user_text = messages[-1]["content"]
        decision = classify(user_text, tool_count=tool_count, threshold=self.threshold)

        if decision.tier == "small" and self.small is not None:
            try:
                reply = self.small.chat(messages)
                if len(reply.text) >= self.min_small_chars:
                    return self._record("small", self.small, reply, False, decision.score, decision.reasons)
                reasons = decision.reasons + ["small model answer too short; escalated"]
            except BackendError:
                reasons = decision.reasons + ["small model failed; escalated"]
            reply = self.frontier.chat(messages)
            return self._record("frontier", self.frontier, reply, True, decision.score, reasons)

        reply = self.frontier.chat(messages)
        return self._record("frontier", self.frontier, reply, False, decision.score, decision.reasons)


def summarize(records: list[RouteRecord]) -> dict:
    total = len(records)
    small = sum(1 for r in records if r.tier == "small")
    return {
        "requests": total,
        "small_share": round(small / total, 4) if total else 0.0,
        "fallbacks": sum(1 for r in records if r.fallback_used),
        "total_cost_usd": round(sum(r.estimated_cost_usd for r in records), 6),
        "mean_latency_s": round(sum(r.latency_s for r in records) / total, 4) if total else 0.0,
        "records": [asdict(r) for r in records],
    }
