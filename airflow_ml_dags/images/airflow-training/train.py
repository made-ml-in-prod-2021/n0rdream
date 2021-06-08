import os
import pickle

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train(path_processed: str, path_artefacts: str):
    X_train = pd.read_csv(os.path.join(path_processed, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(path_processed, "y_train.csv"))
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    os.makedirs(path_artefacts, exist_ok=True)
    with open(os.path.join(path_artefacts, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


@click.command("training-model")
@click.argument("path_processed")
@click.argument("path_artefacts")
def train_command(path_processed, path_artefacts):
    train(path_processed, path_artefacts)


if __name__ == '__main__':
    train_command()
