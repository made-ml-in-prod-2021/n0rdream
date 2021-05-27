from typing import List

from py._path.local import LocalPath
import pytest
import pandas as pd

from src.parameters import PathParams, PreprocessingParams
from src.parameters.preprocessing import FeatureParams, SplittingParams
from src.parameters.training import LogisticRegressionParams
from .helpers import generate_fake_df


@pytest.fixture(scope="function")
def model_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("model.pkl")


@pytest.fixture(scope="function")
def transformer_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("transformer.pkl")


@pytest.fixture(scope="function")
def metrics_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("metrics.json")


@pytest.fixture(scope="function")
def predictions_path(tmpdir: LocalPath) -> LocalPath:
    return tmpdir.join("predictions.csv")


@pytest.fixture(scope="session")
def target_col() -> str:
    return "target"


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak',
    ]


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def fake_train_dataset(
    feature_columns: List,
    target_col: str,
) -> pd.DataFrame:
    df = generate_fake_df(
        feature_columns,
        size=1919,
        target=target_col,
    )
    return df


@pytest.fixture(scope="function")
def fake_train_dataset_path(
    tmpdir: LocalPath,
    fake_train_dataset: pd.DataFrame,
) -> LocalPath:
    path = tmpdir.join("fake_train_data.csv")
    fake_train_dataset.to_csv(path, index=False)
    return path


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def splitting_params() -> SplittingParams:
    params = SplittingParams(
        val_size=0.2,
        random_state=239,
    )
    return params


@pytest.fixture(scope="session")
def preprocessing_params(
    splitting_params: SplittingParams,
    feature_params: FeatureParams,
) -> PreprocessingParams:
    params = PreprocessingParams(
        splitting_params=splitting_params,
        feature_params=feature_params,
    )
    return params


@pytest.fixture(scope="session")
def training_params() -> LogisticRegressionParams:
    params = LogisticRegressionParams(
        model_type="LogisticRegression",
        penalty="l2",
        C=0.001,
        solver="lbfgs",
        random_state=0,
    )
    return params
