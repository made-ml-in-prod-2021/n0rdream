import pickle

from sklearn.compose import ColumnTransformer


def save_transformer_to_pickle(
    transformer: ColumnTransformer,
    output: str,
):
    with open(output, "wb") as f:
        pickle.dump(transformer, f)


def load_transformer_from_pickle(output: str) -> ColumnTransformer:
    with open(output, "rb") as f:
        transformer = pickle.load(f)
    return transformer
