import os
from typing import Dict

from src.parameters import (
    TrainingPipelineParams,
    PathParams,
    PredictionPipelineParams,
)
from src.pipelines import (
    run_training_pipeline,
    run_prediction_pipeline,
)


def test_prediction(
    path_params: PathParams,
    training_pipeline_params: TrainingPipelineParams,
    prediction_pipeline_params: PredictionPipelineParams,
):
    metrics = run_training_pipeline(path_params, training_pipeline_params)
    assert isinstance(metrics, Dict)
    run_prediction_pipeline(prediction_pipeline_params)
    assert os.path.exists(prediction_pipeline_params.predictions_path)
