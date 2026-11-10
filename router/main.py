"""HTTP API. Configure backends with environment variables; the frontier model is required, the small model optional."""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .backends import BackendError, OpenAICompatibleBackend
from .router import HybridRouter

app = FastAPI(title="Hybrid LLM Router", version="0.1.0")


def build_router() -> HybridRouter | None:
    frontier_key = os.getenv("FRONTIER_API_KEY", "").strip()
    if not frontier_key:
        return None
    frontier = OpenAICompatibleBackend(
        "frontier",
        os.getenv("FRONTIER_BASE_URL", "https://api.openai.com/v1"),
        os.getenv("FRONTIER_MODEL", "gpt-4o"),
        api_key=frontier_key,
    )
    small = None
    if os.getenv("SMALL_BASE_URL", "").strip():
        small = OpenAICompatibleBackend(
            "small",
            os.getenv("SMALL_BASE_URL"),  # e.g. http://localhost:11434/v1 for Ollama
            os.getenv("SMALL_MODEL", "qwen2.5-coder:7b"),
        )
    return HybridRouter(frontier=frontier, small=small)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    tool_count: int = Field(default=0, ge=0, le=20)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    router = build_router()
    if router is None:
        raise HTTPException(status_code=503, detail="FRONTIER_API_KEY is not set")
    try:
        record = router.route([{"role": "user", "content": request.message}], tool_count=request.tool_count)
    except BackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "answer": record.answer,
        "tier": record.tier,
        "model": record.model,
        "fallback_used": record.fallback_used,
        "reasons": record.reasons,
        "estimated_cost_usd": record.estimated_cost_usd,
    }
