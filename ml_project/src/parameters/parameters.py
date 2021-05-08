from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=13)


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=255)


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    model_path: str
    metrics_path: str
    transformer_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


@dataclass()
class PredictionPipelineParams:
    dataset_path: str
    model_path: str
    transformer_path: str
    predictions_path: str
    feature_params: FeatureParams
