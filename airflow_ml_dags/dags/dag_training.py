from airflow import DAG
from airflow.operators.dummy import DummyOperator

from global_variables import (
    DEFAULT_ARGS,
    START_DATE,
    PATH_RAW,
    PATH_PROCESSED,
    PATH_MODELS,
)
from helpers import get_docker_operator, get_file_sensor


with DAG(
    "training-pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
    start_date=START_DATE,
) as dag:

    start = DummyOperator(task_id="start")
    preprocessing = DummyOperator(task_id="dummy-preprocessing")
    end = DummyOperator(task_id="end")

    waiting_data = get_file_sensor(PATH_RAW, "data.csv")
    waiting_target = get_file_sensor(PATH_RAW, "target.csv")

    preparation = get_docker_operator(
        "preparation",
        f"{PATH_RAW} {PATH_PROCESSED}",
    )
    splitting = get_docker_operator(
        "splitting",
        PATH_PROCESSED,
    )
    training = get_docker_operator(
        "training",
        f"{PATH_PROCESSED} {PATH_MODELS}",
    )
    validation = get_docker_operator(
        "validation",
        f"{PATH_PROCESSED} {PATH_MODELS}",
    )

    start >> [waiting_data, waiting_target] >> preparation
    preparation >> splitting >> preprocessing
    preprocessing >> training >> validation >> end
