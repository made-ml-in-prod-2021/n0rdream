import click

from src.parameters import (
    read_path_params,
    read_preprocessing_params,
    read_training_params,
)
from src.pipelines import run_training_pipeline


@click.command(name="training_pipeline")
@click.argument("config_paths")
@click.argument("config_preprocessing")
@click.argument("config_training")
def train_command(
    config_paths: str,
    config_preprocessing: str,
    config_training: str,
):
    paths = read_path_params(config_paths)
    preprocessing_params = read_preprocessing_params(config_preprocessing)
    training_params = read_training_params(config_training)
    run_training_pipeline(paths, preprocessing_params, training_params)


if __name__ == "__main__":
    train_command()
