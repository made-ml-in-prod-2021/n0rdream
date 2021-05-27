import logging
import sys

from ..data import read_dataset
from ..models import (
    load_model_from_pickle,
    predict_model,
    save_predicts_to_csv,
)
from ..parameters import PredictionParams
from ..transformers import load_transformer_from_pickle

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_prediction_pipeline(params: PredictionParams):
    logger.info(f"Start prediction pipeline with params {params}")

    logger.info(f"Loading dataset from {params.dataset_path}")
    df = read_dataset(params.dataset_path)

    logger.info(f"Loading transformer from {params.transformer_path}")
    transformer = load_transformer_from_pickle(params.transformer_path)
    X = transformer.transform(df)

    logger.info(f"Loading model from {params.model_path}")
    model = load_model_from_pickle(params.model_path)

    logger.info("Making prediction")
    predicts = predict_model(model, X)

    logger.info(f"Saving results to {params.predictions_path}")
    save_predicts_to_csv(predicts, params.predictions_path)

    logger.info("Prediction completed")
