import os
from datetime import timedelta

from airflow.utils.dates import days_ago
from airflow.models import Variable

LOCAL_PATH_DATA = Variable.get("LOCAL_PATH_DATA")
DOCKER_PATH_DATA = Variable.get("DOCKER_PATH_DATA")

LOCAL_PATH_BEST_MODEL = Variable.get("LOCAL_PATH_BEST_MODEL")
DOCKER_PATH_BEST_MODEL = Variable.get("DOCKER_PATH_BEST_MODEL")

DIR_RAW_DATA = "raw/{{ ds }}/"
DIR_PROCESSED_DATA = "processed/{{ ds }}/"
DIR_MODELS = "models/{{ ds }}/"
DIR_PREDICTIONS = "predictions/{{ ds }}/"

PATH_RAW = os.path.join(DOCKER_PATH_DATA, DIR_RAW_DATA)
PATH_PROCESSED = os.path.join(DOCKER_PATH_DATA, DIR_PROCESSED_DATA)
PATH_MODELS = os.path.join(DOCKER_PATH_DATA, DIR_MODELS)
PATH_PREDICTIONS = os.path.join(DOCKER_PATH_DATA, DIR_PREDICTIONS)

VOLUMES = [
    f"{LOCAL_PATH_DATA}:/{DOCKER_PATH_DATA}",
    f"{LOCAL_PATH_BEST_MODEL}:/{DOCKER_PATH_BEST_MODEL}",
]

DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(seconds=int(Variable.get("RETRY_DELAY_SECONDS"))),
}

START_DATE = days_ago(int(Variable.get("START_DATE_DAYS_AGO")))
