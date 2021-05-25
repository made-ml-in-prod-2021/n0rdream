import click

from src.parameters import read_training_pipeline_params, read_path_params
from src.pipelines import run_training_pipeline


@click.command(name="training_pipeline")
@click.argument("config_path")
@click.argument("config_main")
def train_command(config_path: str, config_main: str):
    params = read_training_pipeline_params(config_main)
    paths = read_path_params(config_path)
    run_training_pipeline(paths, params)


if __name__ == "__main__":
    train_command()
