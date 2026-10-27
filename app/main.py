import math

from fastapi import FastAPI
from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    case_id: str
    relevant_ids: list[str]
    retrieved_ids: list[str]
    cited_ids: list[str]
    answer_claims: list[str] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class Metrics(BaseModel):
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg: float
    citation_precision: float
    unsupported_claim_rate: float
    latency_ms: float
    total_tokens: int


def evaluate(case: EvaluationCase) -> Metrics:
    relevant = set(case.relevant_ids)
    retrieved = case.retrieved_ids
    hits = [1 if item in relevant else 0 for item in retrieved]
    recall = sum(hits) / len(relevant) if relevant else 1.0
    precision = (
        sum(hits) / len(retrieved) if retrieved else (1.0 if not relevant else 0.0)
    )
    first = next(
        (index + 1 for index, item in enumerate(retrieved) if item in relevant), None
    )
    rr = 1 / first if first else 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(
        1 / math.log2(index + 2) for index in range(min(len(relevant), len(retrieved)))
    )
    citations = (
        sum(1 for item in case.cited_ids if item in relevant) / len(case.cited_ids)
        if case.cited_ids
        else 1.0
    )
    supported = set(case.supported_claims)
    unsupported = (
        sum(1 for claim in case.answer_claims if claim not in supported)
        / len(case.answer_claims)
        if case.answer_claims
        else 0.0
    )
    return Metrics(
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=rr,
        ndcg=dcg / ideal if ideal else 1.0,
        citation_precision=citations,
        unsupported_claim_rate=unsupported,
        latency_ms=case.latency_ms,
        total_tokens=case.prompt_tokens + case.completion_tokens,
    )


class GateRequest(BaseModel):
    metrics: Metrics
    minimum_citation_precision: float = 0.9
    maximum_unsupported_claim_rate: float = 0.05


app = FastAPI(title="LLM Evaluation & Observability", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/evaluate", response_model=Metrics)
def evaluate_case(case: EvaluationCase) -> Metrics:
    return evaluate(case)


@app.post("/api/v1/gates")
def gate(request: GateRequest) -> dict[str, object]:
    failures = []
    if request.metrics.citation_precision < request.minimum_citation_precision:
        failures.append("citation_precision")
    if request.metrics.unsupported_claim_rate > request.maximum_unsupported_claim_rate:
        failures.append("unsupported_claim_rate")
    return {"passed": not failures, "failures": failures}
