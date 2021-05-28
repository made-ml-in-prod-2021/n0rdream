import logging

from ..data import (
    read_dataset,
    split_dataset,
)
from ..features import (
    build_transformer,
    pop_target,
)
from ..parameters import (
    PathParams,
    PreprocessingParams,
    TrainingParams,
)
from ..models import (
    train_model,
    predict_model,
    evaluate_model,
    save_model_to_pickle,
    save_metrics_to_json,
)
from ..transformers import save_transformer_to_pickle

logger = logging.getLogger()


def run_training_pipeline(
    paths: PathParams,
    preprocessing_params: PreprocessingParams,
    training_params: TrainingParams,
):
    logger.info(f"Starting training pipeline")

    logger.info("Preprocessing data")
    df = read_dataset(paths.dataset)
    df_train, df_valid = split_dataset(df, preprocessing_params.splitting_params)
    y_train = pop_target(df_train, preprocessing_params.feature_params)
    y_valid = pop_target(df_valid, preprocessing_params.feature_params)
    transformer = build_transformer(preprocessing_params.feature_params)
    transformer.fit(df_train)
    X_train = transformer.transform(df_train)
    X_valid = transformer.transform(df_valid)

    logger.info("Starting training")
    model = train_model(X_train, y_train, training_params)
    y_vld_pred = predict_model(model, X_valid)
    metrics = evaluate_model(y_vld_pred, y_valid)

    logger.info("Saving files")
    save_model_to_pickle(model, paths.model)
    save_transformer_to_pickle(transformer, paths.transformer)
    save_metrics_to_json(metrics, paths.metrics)

    logger.info("Training completed")

    return metrics
