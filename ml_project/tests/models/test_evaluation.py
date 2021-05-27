from typing import Tuple

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import read_dataset
from src.features import (
    build_transformer,
    pop_target,
)
from src.models import train_model
from src.parameters.training import LogisticRegressionParams
from src.parameters.preprocessing import FeatureParams


@pytest.fixture
def features_and_target(
    feature_params: FeatureParams,
    fake_train_dataset_path: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    data = read_dataset(fake_train_dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = transformer.transform(data)
    target = pop_target(data, feature_params)
    return features, target


def test_train_model(
    features_and_target: Tuple[pd.DataFrame, pd.Series],
    training_params: LogisticRegressionParams,
):
    features, target = features_and_target
    model = train_model(features, target, training_params)
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape[0] == target.shape[0]
