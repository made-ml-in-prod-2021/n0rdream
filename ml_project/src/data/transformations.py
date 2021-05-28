from typing import Tuple
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ..parameters.preprocessing import SplittingParams

logger = logging.getLogger()


def split_dataset(
    data: pd.DataFrame,
    params: SplittingParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info(f"Splitting data")
    logging.debug(f"Splitting params: {params}")
    train_data, val_data = train_test_split(
        data,
        test_size=params.val_size,
        random_state=params.random_state,
    )
    logging.debug(f"Train data shape: {train_data.shape}, valid data shape: {val_data.shape}")
    return train_data, val_data
