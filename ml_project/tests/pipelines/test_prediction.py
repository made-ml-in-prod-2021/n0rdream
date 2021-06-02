import os
from typing import Dict, List

from py._path.local import LocalPath
import pytest
import pandas as pd

from ..helpers import generate_fake_df
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


@pytest.fixture(scope="function")
def fake_test_dataset(
    feature_columns: List,
    target_col: str,
) -> pd.DataFrame:
    df = generate_fake_df(
        feature_columns,
        size=1111,
    )
    return df


@pytest.fixture(scope="function")
def fake_test_dataset_path(
    tmpdir: LocalPath,
    fake_test_dataset: pd.DataFrame,
) -> LocalPath:
    path = tmpdir.join("fake_test_data.csv")
    fake_test_dataset.to_csv(path, index=False)
    return path


@pytest.fixture(scope='function')
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
