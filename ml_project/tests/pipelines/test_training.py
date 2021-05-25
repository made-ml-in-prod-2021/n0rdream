import os

from src.parameters import TrainingPipelineParams, PathParams
from src.pipelines import run_training_pipeline


def test_training(
    path_params: PathParams,
    training_pipeline_params: TrainingPipelineParams
):
    metrics = run_training_pipeline(path_params, training_pipeline_params)
    for score in metrics.values():
        assert score >= 0
    assert os.path.exists(path_params.model)
    assert os.path.exists(path_params.transformer)
    assert os.path.exists(path_params.metrics)
