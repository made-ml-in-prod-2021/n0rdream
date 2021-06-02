import logging.config

import yaml


def read_yaml_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(path):
    config = read_yaml_config(path)
    logging.config.dictConfig(config)
