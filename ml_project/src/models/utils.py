import pickle
import json
from typing import Dict
import logging

import numpy as np
import pandas as pd

from .evaluation import SklearnClassificationModel

logger = logging.getLogger()


def save_model_to_pickle(
    model: SklearnClassificationModel,
    path: str,
):
    logger.info(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model_from_pickle(path: str) -> SklearnClassificationModel:
    logger.info(f"Loading model from {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predicts_to_csv(
    predicts: np.ndarray,
    path: str,
):
    logger.info(f"Saving predicts to {path}")
    pd.DataFrame(predicts).to_csv(path)


def save_metrics_to_json(
    metrics: Dict,
    path: str,
):
    logger.info(f"Saving metrics to {path}")
    with open(path, "w") as f:
        json.dump(metrics, f)
