import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..parameters.preprocessing import FeatureParams
from ..transformers import CustomStandardScaler

logger = logging.getLogger()


def build_categorical_pipeline() -> Pipeline:
    logging.debug("Building categorical pipeline")
    cat_pipeline = Pipeline(
        [
            ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return cat_pipeline


def build_numerical_pipeline() -> Pipeline:
    logging.debug("Building numerical pipeline")
    num_pipeline = Pipeline(
        [
            ("scaling", CustomStandardScaler()),
        ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    logging.info("Building transformer")
    logging.debug(f"Transformer params: {params}")
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def pop_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df.pop(params.target_col)
    return target
