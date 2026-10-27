# LLM Evaluation & Observability

A working first vertical slice for deterministic AI evaluation. It computes Recall@K, Precision@K, MRR, NDCG, citation precision, unsupported-claim rate, latency, and token usage, then applies CI-compatible thresholds with explicit failure names.

```bash
uv sync --extra dev
uv run pytest -q
uv run uvicorn app.main:app --reload
```

The initial release deliberately excludes subjective LLM-as-judge scoring. Dataset persistence, experiment comparison, dashboards, traces, and provider-run orchestration are later upgrades and are not claimed here.
