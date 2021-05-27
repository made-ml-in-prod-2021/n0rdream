from py._path.local import LocalPath
import pytest

from src.parameters import (
    PathParams,
    PreprocessingParams,
    PredictionParams,
)
from src.parameters.preprocessing import SplittingParams, FeatureParams


@pytest.fixture()
def path_params(
    fake_train_dataset_path: LocalPath,
    model_path: LocalPath,
    transformer_path: LocalPath,
    metrics_path: LocalPath,
) -> PathParams:
    params = PathParams(
        dataset=fake_train_dataset_path,
        model=model_path,
        transformer=transformer_path,
        metrics=metrics_path,
    )
    return params


@pytest.fixture()
def preprocessing_params(
    splitting_params: SplittingParams,
    feature_params: FeatureParams,
) -> PreprocessingParams:
    params = PreprocessingParams(
        splitting_params=splitting_params,
        feature_params=feature_params,
    )
    return params


@pytest.fixture()
def prediction_params(
    fake_test_dataset_path: LocalPath,
    model_path: LocalPath,
    transformer_path: LocalPath,
    predictions_path: LocalPath,
) -> PredictionParams:
    params = PredictionParams(
        dataset_path=fake_test_dataset_path,
        model_path=model_path,
        transformer_path=transformer_path,
        predictions_path=predictions_path,
    )
    return params
