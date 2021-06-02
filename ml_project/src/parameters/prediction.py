from dataclasses import dataclass


@dataclass()
class PredictionParams:
    dataset_path: str
    model_path: str
    transformer_path: str
    predictions_path: str
