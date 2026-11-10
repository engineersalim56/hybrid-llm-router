# Hybrid LLM Router

Sends simple requests to a small in-house model and complex ones to a frontier model. It falls back to the frontier when the small model fails or gives an unusable answer. Each decision records latency, tokens, estimated cost, and the reasons for the route.

This covers the JD's "prototype hybrid routing that sends simple tasks to smaller in-house models and complex tasks to frontier models", along with the cost and latency optimization across providers.

## How a request is routed

1. **Classify:** a rule-based score for length, complex terms (refactor, race condition, security), code blocks, and tool count. The reasons are returned with each answer.
2. **Small first:** requests under the threshold go to the small model (for example Qwen2.5-Coder served by Ollama).
3. **Fallback:** if the small model errors or returns an empty answer, the request goes to the frontier model and the record shows `fallback_used`.
4. **Account:** token usage and cost are recorded from a price table you set.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                       # mocked backends; no API calls

export FRONTIER_API_KEY="<your key>"
export FRONTIER_MODEL="gpt-4o"            # optional
# optional local small model via Ollama:
ollama pull qwen2.5-coder:7b
export SMALL_BASE_URL="http://localhost:11434/v1"
export SMALL_MODEL="qwen2.5-coder:7b"
uvicorn router.main:app --reload
```

`POST /api/chat` with `{"message": "...", "tool_count": 0}` returns the answer, the tier used, whether fallback happened, the reasons, and the estimated cost.

## Why rules first

- Routing decisions are predictable and cheap, so they can be reviewed and tuned.
- The reasons go back with each answer, so you can see why a request was escalated.
- A learned router can replace the rules later, once you have labelled data from these records.

## Measuring it

`router.router.summarize(records)` reports the share of requests served by the small model, the fallback count, total estimated cost, and mean latency. Use it to compare a routing threshold against quality on your evaluation set.

## Limits

- The quality check is minimal (non-empty answer). Add task-level evaluation before relying on the small model for anything important.
- Prices are configured, not fetched. Update them from the provider's pricing page.
- Benchmark the small and frontier models on your own tasks; the router only helps if the small model is good enough for its share.

## Structure

```
router/
  classify.py   explainable rule-based tier decision
  backends.py   OpenAI-compatible client for OpenAI, Ollama and vLLM
  router.py     routing, fallback, cost and latency records, summary
  main.py       HTTP API
tests/          routing rules, fallback, cost, API
```
