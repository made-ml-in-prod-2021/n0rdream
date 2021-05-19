from marshmallow_dataclass import class_schema
import yaml

from .parameters import TrainingPipelineParams, PredictionPipelineParams

TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
PredictionPipelineParamsSchema = class_schema(PredictionPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def read_prediction_pipeline_params(path: str) -> PredictionPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictionPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
