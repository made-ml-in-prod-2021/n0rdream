import os

import click
import pandas as pd


def prepare_data(path_raw: str, path_processed: str):
    df_data = pd.read_csv(os.path.join(path_raw, "data.csv"))
    df_target = pd.read_csv(os.path.join(path_raw, "target.csv"))
    df_data["target"] = df_target.values
    os.makedirs(path_processed, exist_ok=True)
    df_data.to_csv(os.path.join(path_processed, "train_data.csv"), index=False)


@click.command("preparing-data")
@click.argument("path_raw")
@click.argument("path_processed")
def prepare_data_command(path_raw, path_processed):
    prepare_data(path_raw, path_processed)


if __name__ == '__main__':
    prepare_data_command()
