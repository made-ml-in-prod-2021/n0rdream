import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..parameters import FeatureParams
from ..transformers import CustomStandardScaler


def build_categorical_pipeline() -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return cat_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("scaling", CustomStandardScaler()),
        ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
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
