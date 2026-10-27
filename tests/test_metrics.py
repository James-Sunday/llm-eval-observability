from app.main import EvaluationCase, GateRequest, evaluate, gate


def test_deterministic_retrieval_and_claim_metrics():
    metrics = evaluate(
        EvaluationCase(
            case_id="1",
            relevant_ids=["a", "b"],
            retrieved_ids=["x", "a", "b"],
            cited_ids=["a"],
            answer_claims=["supported", "invented"],
            supported_claims=["supported"],
            latency_ms=12,
            prompt_tokens=10,
            completion_tokens=5,
        )
    )
    assert metrics.recall_at_k == 1
    assert metrics.precision_at_k == 2 / 3
    assert metrics.reciprocal_rank == 0.5
    assert metrics.citation_precision == 1
    assert metrics.unsupported_claim_rate == 0.5
    assert metrics.total_tokens == 15


def test_regression_gate_names_failed_thresholds():
    metrics = evaluate(
        EvaluationCase(
            case_id="1",
            relevant_ids=["a"],
            retrieved_ids=["x"],
            cited_ids=["x"],
            answer_claims=["invented"],
            latency_ms=1,
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    result = gate(GateRequest(metrics=metrics))
    assert result == {
        "passed": False,
        "failures": ["citation_precision", "unsupported_claim_rate"],
    }
