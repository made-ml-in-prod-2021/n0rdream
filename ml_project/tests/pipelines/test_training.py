import os

from src.parameters import TrainingPipelineParams
from src.pipelines import run_training_pipeline


def test_training(training_pipeline_params: TrainingPipelineParams):
    metrics = run_training_pipeline(training_pipeline_params)
    for score in metrics.values():
        assert score >= 0
    assert os.path.exists(training_pipeline_params.model_path)
    assert os.path.exists(training_pipeline_params.transformer_path)
    assert os.path.exists(training_pipeline_params.metrics_path)
