from py._path.local import LocalPath
import pytest
from typing import List

from src.parameters import FeatureParams, SplittingParams
from .helpers import generate_fake_df


@pytest.fixture()
def model_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("model.pkl")


@pytest.fixture()
def transformer_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("transformer.pkl")


@pytest.fixture()
def metrics_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("metrics.json")


@pytest.fixture()
def predictions_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("predictions.csv")


@pytest.fixture()
def target_col() -> str:
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        'cp',
        'restecg',
        'slope',
        'ca',
        'thal',
        'sex',
        'fbs',
        'exang',
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak',
    ]


@pytest.fixture
def feature_columns() -> List[str]:
    return [
        'age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
    ]


@pytest.fixture()
def fake_train_dataset_path(
    tmpdir: LocalPath,
    feature_columns: List,
    target_col: str,
) -> LocalPath:
    fake_df = generate_fake_df(feature_columns, size=1919, target=target_col)
    file = tmpdir.join("fake_train_data.csv")
    fake_df.to_csv(file, index=False)
    return file


@pytest.fixture()
def fake_test_dataset_path(
    tmpdir: LocalPath,
    feature_columns: List,
) -> LocalPath:
    fake_df = generate_fake_df(feature_columns, size=1111)
    file = tmpdir.join("fake_test_data.csv")
    fake_df.to_csv(file, index=False)
    return file


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return params


@pytest.fixture()
def splitting_params() -> SplittingParams:
    return SplittingParams(val_size=0.2, random_state=239)
