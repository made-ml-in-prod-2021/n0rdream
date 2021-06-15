from airflow import DAG
from airflow.operators.dummy import DummyOperator

from global_variables import (
    DEFAULT_ARGS,
    START_DATE,
    PATH_RAW,
    DOCKER_PATH_BEST_MODEL,
    PATH_PREDICTIONS,
)
from helpers import get_docker_operator, get_file_sensor


with DAG(
    "prediction-pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=START_DATE,
) as dag:

    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    waiting_data = get_file_sensor(PATH_RAW, "data.csv")
    waiting_model = get_file_sensor(DOCKER_PATH_BEST_MODEL, "model.pkl")

    prediction = get_docker_operator(
        "prediction",
        f"{PATH_RAW} {DOCKER_PATH_BEST_MODEL} {PATH_PREDICTIONS}",
    )

    start >> [waiting_data, waiting_model] >> prediction >> end
