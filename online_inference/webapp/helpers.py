from typing import Union, List
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


class CustomStandardScaler(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.to_numpy()
        x_transformed = (x - x.mean(axis=0)) / x.std(axis=0)
        return x_transformed


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "src.transformers.transformers":
            renamed_module = __name__
        return super().find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def load_transformer_from_pickle(output: str) -> ColumnTransformer:
    with open(output, "rb") as f:
        transformer = renamed_load(f)
    return transformer


def load_model_from_pickle(output: str) -> SklearnClassificationModel:
    with open(output, "rb") as f:
        model = pickle.load(f)
    return model


def read_dataset_from_json(data: str) -> pd.DataFrame:
    df = pd.read_json(data)
    return df


def predict_model(
    model: SklearnClassificationModel,
    features: pd.DataFrame,
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def make_predict(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
    model: SklearnClassificationModel,
) -> List:
    X = transformer.transform(df)
    preds = predict_model(model, X)
    return preds.tolist()
