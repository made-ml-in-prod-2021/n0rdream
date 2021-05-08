import click

from src.parameters import read_training_pipeline_params
from src.pipelines import run_training_pipeline


@click.command(name="training_pipeline")
@click.argument("config_path")
def train_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    run_training_pipeline(params)


if __name__ == "__main__":
    train_command()
