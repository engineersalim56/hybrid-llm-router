"""OpenAI-compatible chat backends. Works for OpenAI, and for local models served by Ollama or vLLM,
which both expose the same /v1/chat/completions interface."""
import time
from dataclasses import dataclass

import httpx


class BackendError(RuntimeError):
    """The backend failed or returned an unusable answer. The router uses this to decide on fallback."""


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class OpenAICompatibleBackend:
    def __init__(self, name: str, base_url: str, model: str, api_key: str = "", timeout: float = 30,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    def chat(self, messages: list[dict]) -> Reply:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(f"{self.base_url}/chat/completions", headers=headers,
                                       json={"model": self.model, "messages": messages, "temperature": 0.2})
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name} unreachable: {exc.__class__.__name__}") from exc
        latency = time.perf_counter() - start
        if response.status_code >= 400:
            raise BackendError(f"{self.name} returned HTTP {response.status_code}")
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise BackendError(f"{self.name} returned an unexpected response") from exc
        return Reply(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
        )
