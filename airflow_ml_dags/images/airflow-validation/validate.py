import os
import pickle
import json

import click
import pandas as pd
from sklearn.metrics import accuracy_score


def validate(path_processed: str, path_artefacts: str):
    X_valid = pd.read_csv(os.path.join(path_processed, "X_valid.csv"))
    y_valid = pd.read_csv(os.path.join(path_processed, "y_valid.csv"))
    with open(os.path.join(path_artefacts, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    with open(os.path.join(path_artefacts, "metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy}, f)


@click.command("validating-model")
@click.argument("path_processed")
@click.argument("path_artefacts")
def validate_command(path_processed, path_artefacts):
    validate(path_processed, path_artefacts)


if __name__ == '__main__':
    validate_command()
