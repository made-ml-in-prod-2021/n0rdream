import os
import pickle

import click
import pandas as pd


def predict(path_raw: str, path_artefacts: str, path_predictions: str):
    X = pd.read_csv(os.path.join(path_raw, "data.csv"))
    with open(os.path.join(path_artefacts, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    predictions = pd.DataFrame(model.predict(X))
    os.makedirs(path_predictions, exist_ok=True)
    predictions.to_csv(os.path.join(path_predictions, "predictions.csv"), index=False)


@click.command("predicting-labels")
@click.argument("path_raw")
@click.argument("path_artefacts")
@click.argument("path_predictions")
def predict_command(path_raw, path_artefacts, path_predictions):
    predict(path_raw, path_artefacts, path_predictions)


if __name__ == '__main__':
    predict_command()
