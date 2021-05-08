import pandas as pd
from numpy.testing import assert_allclose

from src.data import read_dataset
from src.features import build_transformer, pop_target
from src.parameters import FeatureParams


def test_build_transformer(
    feature_params: FeatureParams,
    fake_train_dataset_path: str,
):
    data = read_dataset(fake_train_dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = transformer.transform(data)
    assert not pd.isnull(features).any().any()


def test_pop_target(
    feature_params: FeatureParams,
    fake_train_dataset_path: str,
):
    data = read_dataset(fake_train_dataset_path)
    expected_target = data[feature_params.target_col]
    obtained_target = pop_target(data, feature_params)
    assert_allclose(expected_target.to_numpy(), obtained_target.to_numpy())
