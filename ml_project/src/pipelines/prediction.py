import logging

from ..data import read_dataset
from ..models import (
    load_model_from_pickle,
    predict_model,
    save_predicts_to_csv,
)
from ..parameters import PredictionParams
from ..transformers import load_transformer_from_pickle

logger = logging.getLogger()


def run_prediction_pipeline(params: PredictionParams):
    logger.info(f"Starting training pipeline")

    logger.info("Preprocessing data")
    df = read_dataset(params.dataset_path)
    transformer = load_transformer_from_pickle(params.transformer_path)
    X = transformer.transform(df)
    model = load_model_from_pickle(params.model_path)

    logger.info("Starting prediction")
    predicts = predict_model(model, X)

    logger.info(f"Saving results")
    save_predicts_to_csv(predicts, params.predictions_path)

    logger.info("Prediction completed")
