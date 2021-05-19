import logging
import sys
import os
from typing import List

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.models import load_model_from_pickle, make_predict
from src.transformers import load_transformer_from_pickle
from src.data import read_dataset_from_json

DEFAULT_PATH_TRANSFORMER = "artefacts/transformer.pkl"
DEFAULT_PATH_MODEL = "artefacts/model.pkl"
PATH_TRANSFORMER = os.getenv("PATH_TO_TRANSFORMER", DEFAULT_PATH_TRANSFORMER)
PATH_MODEL = os.getenv("PATH_TO_MODEL", DEFAULT_PATH_MODEL)
ARTEFACTS = {}

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = FastAPI()


@app.on_event("startup")
def load_transformer():
    logger.info("loading transformer")
    ARTEFACTS["transformer"] = load_transformer_from_pickle(PATH_TRANSFORMER)


@app.on_event("startup")
def load_model():
    logger.info("loading model")
    ARTEFACTS["model"] = load_model_from_pickle(PATH_MODEL)


@app.get("/")
def root():
    return "Service is working"


class InputDataModel(BaseModel):
    data: str


@app.get("/predict/", response_model=List)
def predict(request: InputDataModel) -> List:
    logger.info("reading dataset")
    df = read_dataset_from_json(request.data)
    logger.info("making prediction")
    preds = make_predict(df, ARTEFACTS["transformer"], ARTEFACTS["model"])
    return preds
