import logging

import click

from src.parameters import read_prediction_params
from src.pipelines import run_prediction_pipeline
from src.helpers import setup_logging

CONFIG_LOGGING_PATH = "configs/logging/prediction.yml"

setup_logging(CONFIG_LOGGING_PATH)
logger = logging.getLogger()


@click.command(name="prediction_pipeline")
@click.argument("config_path")
def predict_command(config_path: str):
    params = read_prediction_params(config_path)
    run_prediction_pipeline(params)


if __name__ == "__main__":
    try:
        predict_command()
    except Exception as e:
        logging.error(e)
