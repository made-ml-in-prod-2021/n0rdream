from dataclasses import dataclass

import pandas as pd
import requests
from marshmallow_dataclass import class_schema
import yaml
import click


@dataclass()
class RequestParams:
    url: str
    dataset_path: str
    predicts_path: str


def read_request_params(path: str) -> RequestParams:
    with open(path, "r") as input_stream:
        schema = class_schema(RequestParams)()
        return schema.load(yaml.safe_load(input_stream))


def get_json_data(path):
    df = pd.read_csv(path)
    return df.to_json()


def make_request(params: RequestParams):
    data = get_json_data(params.dataset_path)
    response = requests.get(params.url, json={"data": data})
    if response.status_code == 200:
        preds = response.json()
        pd.DataFrame(preds).to_csv(params.predicts_path)


@click.command(name="request")
@click.argument("config_path")
def make_request_command(config_path: str):
    params = read_request_params(config_path)
    make_request(params)


if __name__ == "__main__":
    make_request_command()
