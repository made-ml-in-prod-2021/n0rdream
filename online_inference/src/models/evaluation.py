from typing import Dict, Union, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from ..parameters import TrainingParams


SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_params: TrainingParams
) -> SklearnClassificationModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            C=1,
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnClassificationModel,
    features: pd.DataFrame,
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray,
    target: pd.Series,
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def make_predict(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
    model: SklearnClassificationModel,
) -> List:
    X = transformer.transform(df)
    preds = predict_model(model, X)
    return preds.tolist()
