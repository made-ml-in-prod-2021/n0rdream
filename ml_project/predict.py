import click

from src.parameters import read_prediction_pipeline_params
from src.pipelines import run_prediction_pipeline


@click.command(name="prediction_pipeline")
@click.argument("config_path")
def predict_command(config_path: str):
    params = read_prediction_pipeline_params(config_path)
    run_prediction_pipeline(params)


if __name__ == "__main__":
    predict_command()
