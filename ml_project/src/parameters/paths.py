from dataclasses import dataclass


@dataclass()
class PathParams:
    dataset: str
    model: str
    metrics: str
    transformer: str
