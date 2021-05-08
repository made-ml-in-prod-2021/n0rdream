import os

from py._path.local import LocalPath
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.transformers import (
    save_transformer_to_pickle,
    load_transformer_from_pickle,
)


def test_serialize_transformer(transformer_path: LocalPath):
    transformer = ColumnTransformer('scaler', remainder=StandardScaler())
    save_transformer_to_pickle(transformer, transformer_path)
    assert os.path.exists(transformer_path)
    transformer = load_transformer_from_pickle(transformer_path)
    assert isinstance(transformer, ColumnTransformer)
