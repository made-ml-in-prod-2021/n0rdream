import os

import click
import numpy as np
import pandas as pd

SIZE = 100


def generate_data(size: int) -> pd.DataFrame:
    X = pd.DataFrame()
    for col in ['sex', 'fbs', 'exang']:
        X[col] = np.random.randint(0, 2, size)
    X['age'] = np.random.randint(29, 78, size)
    X['cp'] = np.random.randint(0, 4, size)
    X['trestbps'] = np.random.randint(94, 201, size)
    X['chol'] = np.random.randint(126, 565, size)
    X['restecg'] = np.random.randint(0, 3, size)
    X['thalach'] = np.random.randint(71, 203, size)
    X['oldpeak'] = np.random.randint(0, 63, size) / 10
    X['slope'] = np.random.randint(0, 3, size)
    X['ca'] = np.random.randint(0, 5, size)
    X['thal'] = np.random.randint(0, 4, size)
    return X


def generate_target(size: int) -> pd.DataFrame:
    return pd.DataFrame(np.random.randint(0, 2, size))


def collect_data(path_raw):
    X = generate_data(SIZE)
    y = generate_target(SIZE)
    os.makedirs(path_raw, exist_ok=True)
    X.to_csv(os.path.join(path_raw, "data.csv"), index=False)
    y.to_csv(os.path.join(path_raw, "target.csv"), index=False)


@click.command("collecting-data")
@click.argument("path_raw")
def collect_data_command(path_raw):
    collect_data(path_raw)


if __name__ == '__main__':
    collect_data_command()
