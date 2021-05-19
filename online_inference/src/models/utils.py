import pickle
import json
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .evaluation import SklearnClassificationModel


def save_model_to_pickle(
    model: SklearnClassificationModel,
    output: str,
):
    with open(output, "wb") as f:
        pickle.dump(model, f)


def load_model_from_pickle(output: str) -> SklearnClassificationModel:
    with open(output, "rb") as f:
        model = pickle.load(f)
    return model


def save_predicts_to_csv(
    predicts: Union[np.ndarray, List],
    output: str,
):
    pd.DataFrame(predicts).to_csv(output)


def save_metrics_to_json(
    metrics: Dict,
    path: str,
):
    with open(path, "w") as f:
        json.dump(metrics, f)
