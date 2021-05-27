import os
from typing import Dict

from src.parameters import (
    PathParams,
    PreprocessingParams,
    TrainingParams,
    PredictionParams,
)
from src.pipelines import (
    run_training_pipeline,
    run_prediction_pipeline,
)


def test_prediction(
    path_params: PathParams,
    preprocessing_params: PreprocessingParams,
    training_params: TrainingParams,
    prediction_params: PredictionParams,
):
    metrics = run_training_pipeline(
        path_params,
        preprocessing_params,
        training_params,
    )
    assert isinstance(metrics, Dict)
    run_prediction_pipeline(prediction_params)
    assert os.path.exists(prediction_params.predictions_path)
