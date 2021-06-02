from typing import Union, Dict
import logging

from marshmallow_dataclass import class_schema
import yaml

from .paths import PathParams
from .preprocessing import PreprocessingParams
from .training import TrainingParams, RandomForestParams, LogisticRegressionParams
from .prediction import PredictionParams
from ..models.codes import (
    CODE_RANDOM_FOREST,
    CODE_LOGISTIC_REGRESSION,
)

Params = Union[
    PathParams,
    PreprocessingParams,
    TrainingParams,
    PredictionParams,
]

logger = logging.getLogger()


def get_yaml_params(path: str) -> Dict:
    with open(path, "r") as input_stream:
        yaml_config = yaml.safe_load(input_stream)
        return yaml_config


def read_yaml(yaml_params: Dict, params: Params) -> Params:
    schema = class_schema(params)
    return schema().load(yaml_params)


def read_params(path: str, params: Params) -> Params:
    logger.debug(f"Loading {params} from {path}")
    yaml_params = get_yaml_params(path)
    return read_yaml(yaml_params, params)


def read_path_params(path: str) -> PathParams:
    return read_params(path, PathParams)


def read_preprocessing_params(path: str) -> PreprocessingParams:
    return read_params(path, PreprocessingParams)


def read_training_params(path: str) -> TrainingParams:
    yaml_params = get_yaml_params(path)
    if yaml_params["model_type"] == CODE_RANDOM_FOREST:
        return read_yaml(yaml_params, RandomForestParams)
    if yaml_params["model_type"] == CODE_LOGISTIC_REGRESSION:
        return read_yaml(yaml_params, LogisticRegressionParams)
    logger.error(f"Bad model type from {path} config")
    raise NotImplementedError()


def read_prediction_params(path: str) -> PredictionParams:
    return read_params(path, PredictionParams)
