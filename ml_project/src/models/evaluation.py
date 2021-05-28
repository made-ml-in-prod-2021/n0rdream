from typing import Dict, Union
import logging

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

logger = logging.getLogger()


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
    logger.info(f"Training model: {params.model_type}")
    logger.debug(f"Model params: {params}")
    if params.model_type == CODE_RANDOM_FOREST:
        model = get_rf_model(params)
    elif params.model_type == CODE_LOGISTIC_REGRESSION:
        model = get_lr_model(params)
    else:
        logger.error(f"Model {params.model_type} is not implemented")
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnClassificationModel,
    features: pd.DataFrame,
) -> np.ndarray:
    logger.info("Predicting labels")
    predicts = model.predict(features)
    logger.debug(f"Predicts shape: {predicts.shape}")
    return predicts


def evaluate_model(
    predicts: np.ndarray,
    target: pd.Series,
) -> Dict[str, float]:
    logger.info("Calculating metrics")
    accuracy = accuracy_score(target, predicts)
    f1 = f1_score(target, predicts)
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
    }
    logging.debug(f"Got metrics: {metrics}")
    return metrics
