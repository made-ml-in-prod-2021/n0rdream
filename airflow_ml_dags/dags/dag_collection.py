from airflow import DAG
from airflow.operators.dummy import DummyOperator

from global_variables import (
    DEFAULT_ARGS,
    START_DATE,
    PATH_RAW,
)
from helpers import get_docker_operator


with DAG(
    "data-collection",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=START_DATE,
) as dag:

    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    collection = get_docker_operator(
        "collection",
        PATH_RAW,
    )

    start >> collection >> end
