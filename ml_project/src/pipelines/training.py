import logging
import sys

from ..data import (
    read_dataset,
    split_dataset,
)
from ..features import (
    build_transformer,
    pop_target,
)
from ..parameters import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ..models import (
    train_model,
    predict_model,
    evaluate_model,
    save_model_to_pickle,
    save_metrics_to_json,
)
from ..transformers import save_transformer_to_pickle

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_training_pipeline(params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {params}")

    logger.info(f"Loading dataset from {params.input_data_path}")
    df = read_dataset(params.input_data_path)

    df_train, df_valid = split_dataset(df, params.splitting_params)

    logger.info("Preprocessing data")
    y_train = pop_target(df_train, params.feature_params)
    y_valid = pop_target(df_valid, params.feature_params)
    transformer = build_transformer(params.feature_params)
    transformer.fit(df_train)
    X_train = transformer.transform(df_train)
    X_valid = transformer.transform(df_valid)

    logger.info("Training model")
    model = train_model(X_train, y_train, params.train_params)
    y_vld_pred = predict_model(model, X_valid)

    logger.info("Evaluating metrics")
    metrics = evaluate_model(y_vld_pred, y_valid)
    logger.info(f"Metrics: {metrics}")

    logger.info("Saving files")
    save_model_to_pickle(model, params.model_path)
    save_transformer_to_pickle(transformer, params.transformer_path)
    save_metrics_to_json(metrics, params.metrics_path)

    logger.info("Training completed")

    return metrics
