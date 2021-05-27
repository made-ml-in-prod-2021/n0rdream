import os

from src.parameters import (
    PathParams,
    PreprocessingParams,
    TrainingParams,
)
from src.pipelines import run_training_pipeline


def test_training(
    path_params: PathParams,
    preprocessing_params: PreprocessingParams,
    training_params: TrainingParams,
):
    metrics = run_training_pipeline(
        path_params,
        preprocessing_params,
        training_params,
    )
    for score in metrics.values():
        assert score >= 0
    assert os.path.exists(path_params.model)
    assert os.path.exists(path_params.transformer)
    assert os.path.exists(path_params.metrics)
