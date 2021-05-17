from py._path.local import LocalPath
import pytest

from src.parameters import (
    TrainingPipelineParams,
    PredictionPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


@pytest.fixture()
def training_params() -> TrainingParams:
    return TrainingParams(model_type="LogisticRegression")


@pytest.fixture()
def training_pipeline_params(
    fake_train_dataset_path: LocalPath,
    model_path: LocalPath,
    transformer_path: LocalPath,
    metrics_path: LocalPath,
    splitting_params: SplittingParams,
    feature_params: FeatureParams,
    training_params: TrainingParams,
) -> TrainingPipelineParams:
    params = TrainingPipelineParams(
        input_data_path=fake_train_dataset_path,
        model_path=model_path,
        transformer_path=transformer_path,
        metrics_path=metrics_path,
        splitting_params=splitting_params,
        feature_params=feature_params,
        train_params=training_params,
    )
    return params


@pytest.fixture()
def prediction_pipeline_params(
    fake_test_dataset_path: LocalPath,
    model_path: LocalPath,
    transformer_path: LocalPath,
    predictions_path: LocalPath,
) -> PredictionPipelineParams:
    params = PredictionPipelineParams(
        dataset_path=fake_test_dataset_path,
        model_path=model_path,
        transformer_path=transformer_path,
        predictions_path=predictions_path,
    )
    return params
