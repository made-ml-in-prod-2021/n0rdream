import logging

import pandas as pd

logger = logging.getLogger()


def read_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    logger.debug(f"Dataset shape: {df.shape}")
    return df
