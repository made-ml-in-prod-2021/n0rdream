import pickle
import logging

from sklearn.compose import ColumnTransformer

logger = logging.getLogger()


def save_transformer_to_pickle(
    transformer: ColumnTransformer,
    path: str,
):
    logger.info(f"Saving transformer to {path}")
    with open(path, "wb") as f:
        pickle.dump(transformer, f)


def load_transformer_from_pickle(path: str) -> ColumnTransformer:
    logger.info(f"Loading transformer from {path}")
    with open(path, "rb") as f:
        transformer = pickle.load(f)
    return transformer
