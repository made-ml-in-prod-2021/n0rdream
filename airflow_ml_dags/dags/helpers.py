import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from global_variables import VOLUMES


def get_docker_operator(label: str, command: str) -> DockerOperator:
    operator = DockerOperator(
        image=f"airflow-{label}",
        command=command,
        network_mode="bridge",
        task_id=f"docker-{label}",
        do_xcom_push=False,
        volumes=VOLUMES,
    )
    return operator


def get_file_sensor(path: str, filename: str) -> FileSensor:
    label, _ = filename.split(".")
    sensor = FileSensor(
        task_id=f"waiting-{label}",
        poke_interval=10,
        retries=100,
        filepath=os.path.join(path, filename),
    )
    return sensor
