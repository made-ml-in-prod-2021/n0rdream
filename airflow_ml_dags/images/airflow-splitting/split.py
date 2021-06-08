import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.3
RANDOM_STATE = 0


def split_data(path_processed: str):
    df = pd.read_csv(os.path.join(path_processed, "train_data.csv"))
    df_train, df_valid = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    y_train = df_train.pop("target")
    y_valid = df_valid.pop("target")

    df_train.to_csv(os.path.join(path_processed, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(path_processed, "y_train.csv"), index=False)

    df_valid.to_csv(os.path.join(path_processed, "X_valid.csv"), index=False)
    y_valid.to_csv(os.path.join(path_processed, "y_valid.csv"), index=False)


@click.command("splitting-data")
@click.argument("path_processed")
def split_data_command(path_processed):
    split_data(path_processed)


if __name__ == '__main__':
    split_data_command()
