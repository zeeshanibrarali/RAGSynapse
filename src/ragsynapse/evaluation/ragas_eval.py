"""
RAGAS evaluation pipeline — RAGSynapse v2
Uses Ollama as judge LLM — no OpenAI key required.
Author: Zeeshan Ibrar
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from datasets import Dataset
import mlflow


@dataclass
class EvalResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    num_questions: int = 0
    run_id: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "faithfulness":      round(self.faithfulness, 4),
            "answer_relevancy":  round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "num_questions":     self.num_questions,
        }

    def overall_score(self) -> float:
        return round(
            (self.faithfulness + self.answer_relevancy + self.context_precision) / 3,
            4
        )


def _get_ragas_llm():
    """
    Returns a RAGAS-compatible LLM wrapper.
    Uses Ollama if no OpenAI key, OpenAI if key is available.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if openai_key:
        from langchain_openai import ChatOpenAI
        return LangchainLLMWrapper(ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key
        ))
    else:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        return LangchainLLMWrapper(Ollama(
            model="tinyllama",
            base_url=ollama_url,
        ))


def _get_ragas_embeddings():
    """
    Returns RAGAS-compatible embeddings.
    Uses Ollama embeddings — no OpenAI key needed.
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    return LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model="tinyllama",
        base_url=ollama_url,
    ))


def run_evaluation(
    questions: list[str],
    ground_truths: list[str],
    chat_engine,
    run_name: str = None,
    log_to_mlflow: bool = True,
) -> EvalResult:
    """
    Run RAGAS evaluation using Ollama as judge LLM.
    Falls back gracefully if scoring fails.
    """
    if len(questions) != len(ground_truths):
        return EvalResult(error="Questions and ground truths must be same length.")

    if not questions:
        return EvalResult(error="No questions provided.")

    answers = []
    contexts = []

    # ── Query RAG pipeline for each question ─────────────────────────────────
    for q in questions:
        try:
            response = chat_engine.query(q)
            answers.append(str(response))
            ctx = [node.text for node in response.source_nodes] \
                if hasattr(response, "source_nodes") and response.source_nodes \
                else ["no context retrieved"]
            contexts.append(ctx)
        except Exception as e:
            answers.append("error")
            contexts.append(["no context retrieved"])

    # ── Build RAGAS dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    # ── Configure RAGAS to use Ollama ─────────────────────────────────────────
    try:
        ragas_llm = _get_ragas_llm()
        ragas_embeddings = _get_ragas_embeddings()

        metrics = [faithfulness, answer_relevancy, context_precision]
        for metric in metrics:
            metric.llm = ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = ragas_embeddings

        results = evaluate(
            dataset,
            metrics=metrics,
            raise_exceptions=False,
        )
        df = results.to_pandas()

    except Exception as e:
        return EvalResult(error=f"RAGAS evaluation failed: {e}")

    result = EvalResult(
        faithfulness=float(df["faithfulness"].fillna(0).mean()),
        answer_relevancy=float(df["answer_relevancy"].fillna(0).mean()),
        context_precision=float(df["context_precision"].fillna(0).mean()),
        num_questions=len(questions),
    )

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    if log_to_mlflow:
        try:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            )
            mlflow.set_experiment("RAGSynapse-Evaluation")
            with mlflow.start_run(run_name=run_name or f"eval_{int(time.time())}"):
                mlflow.log_metrics(result.as_dict())
                mlflow.log_param("num_questions", len(questions))
                mlflow.log_param("provider", os.getenv("LLM_PROVIDER", "ollama"))
                mlflow.log_param("judge_llm",
                    "openai" if os.getenv("OPENAI_API_KEY") else "ollama/tinyllama")
                result.run_id = mlflow.active_run().info.run_id
        except Exception:
            pass

    return result