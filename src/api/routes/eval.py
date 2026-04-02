from fastapi import APIRouter, HTTPException
from api.models import EvalRequest, EvalResponse
from api.dependencies import get_query_engine

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest):
    """
    Run RAGAS evaluation on provided question/answer pairs.
    Logs results to MLflow automatically.
    """
    if len(request.questions) != len(request.ground_truths):
        raise HTTPException(
            status_code=400,
            detail="questions and ground_truths must have equal length"
        )

    try:
        import sys
        sys.path.insert(0, "/ragsynapse/src")
        from ragsynapse.evaluation import run_evaluation as ragas_eval

        query_engine = get_query_engine(
            provider=request.provider,
            model=None,
        )

        result = ragas_eval(
            questions=request.questions,
            ground_truths=request.ground_truths,
            chat_engine=query_engine,
            run_name=request.run_name,
            log_to_mlflow=True,
        )

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        return EvalResponse(
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            overall_score=result.overall_score(),
            num_questions=result.num_questions,
            run_id=result.run_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_results():
    """Fetch latest 10 evaluation runs from MLflow."""
    try:
        import mlflow
        import os
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("RAGSynapse-Evaluation")
        if not experiment:
            return {"runs": [], "message": "No evaluation runs yet"}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=10,
            order_by=["start_time DESC"],
        )
        return {
            "runs": [
                {
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name,
                    "metrics": r.data.metrics,
                    "params": r.data.params,
                    "timestamp": r.info.start_time,
                }
                for r in runs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))