import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def read_dataset_from_json(data: str) -> pd.DataFrame:
    df = pd.read_json(data)
    return df
