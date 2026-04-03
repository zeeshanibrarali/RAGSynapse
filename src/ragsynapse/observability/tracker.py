"""
MLflow query tracker — RAGSynapse v2
Logs every chat query with latency, token estimate, and cost.
Author: Zeeshan Ibrar
"""

import os
import time
import mlflow


# Cost per 1K tokens — matches model_factory.py
COST_PER_1K_TOKENS = {
    "gpt-4o-mini":               0.00015,
    "gpt-4o":                    0.005,
    "claude-3-5-haiku-20241022": 0.0008,
    "claude-3-5-sonnet-20241022":0.003,
    "tinyllama":                 0.0,
    "llama3.2":                  0.0,
    "llama3.2:latest":           0.0,
    "mistral":                   0.0,
    "codellama":                 0.0,
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _get_cost(model: str, total_tokens: int) -> float:
    rate = COST_PER_1K_TOKENS.get(model, 0.001)
    return round((total_tokens / 1000) * rate, 6)


def log_query(
    question: str,
    answer: str,
    provider: str,
    model: str,
    total_latency_ms: float,
    num_source_nodes: int = 0,
    context_texts: list[str] = None,
) -> None:
    """
    Log a single RAG query to MLflow.
    Called after every successful chat response.
    Fails silently so it never breaks the chat UI.
    """
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment("RAGSynapse-Queries")

        # Token estimates
        question_tokens = _estimate_tokens(question)
        answer_tokens = _estimate_tokens(answer)
        context_tokens = sum(
            _estimate_tokens(c) for c in (context_texts or [])
        )
        total_tokens = question_tokens + answer_tokens + context_tokens
        estimated_cost = _get_cost(model, total_tokens)

        run_name = f"{provider}_{int(time.time())}"

        with mlflow.start_run(run_name=run_name):
            # ── Metrics ───────────────────────────────────────────────────────
            mlflow.log_metrics({
                "query_latency_ms":    round(total_latency_ms, 2),
                "question_tokens":     question_tokens,
                "answer_tokens":       answer_tokens,
                "context_tokens":      context_tokens,
                "total_tokens":        total_tokens,
                "estimated_cost_usd":  estimated_cost,
                "num_source_nodes":    num_source_nodes,
            })

            # ── Parameters ────────────────────────────────────────────────────
            mlflow.log_params({
                "provider":        provider,
                "model":           model,
                "question_length": len(question),
                "answer_length":   len(answer),
            })

    except Exception as e:
        import streamlit as st
        print(f"MLflow tracking error: {e}")