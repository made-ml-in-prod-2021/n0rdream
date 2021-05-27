from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..parameters import TrainingParams
from .codes import (
    CODE_RANDOM_FOREST,
    CODE_LOGISTIC_REGRESSION,
)

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def get_rf_model(params: TrainingParams) -> SklearnClassificationModel:
    model = RandomForestClassifier(
        n_estimators=params.n_estimators,
        criterion=params.criterion,
        min_samples_leaf=params.min_samples_leaf,
        random_state=params.random_state,
    )
    return model


def get_lr_model(params: TrainingParams) -> SklearnClassificationModel:
    model = LogisticRegression(
        penalty=params.penalty,
        C=params.C,
        solver=params.solver,
        random_state=params.random_state,
    )
    return model


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    params: TrainingParams,
) -> SklearnClassificationModel:
    if params.model_type == CODE_RANDOM_FOREST:
        model = get_rf_model(params)
    elif params.model_type == CODE_LOGISTIC_REGRESSION:
        model = get_lr_model(params)
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
