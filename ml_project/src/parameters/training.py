from dataclasses import dataclass
from typing import Union


@dataclass()
class LogisticRegressionParams:
    model_type: str
    penalty: str
    C: float
    solver: str
    random_state: int


@dataclass()
class RandomForestParams:
    model_type: str
    n_estimators: int
    criterion: str
    min_samples_leaf: int
    random_state: int


TrainingParams = Union[
    LogisticRegressionParams,
    RandomForestParams,
]
