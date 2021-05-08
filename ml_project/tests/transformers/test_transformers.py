import numpy as np
import pandas as pd

from src.transformers import CustomStandardScaler


def test_custom_standard_scaler():
    X = pd.DataFrame(np.random.normal(11, 3.3, 100))
    scaler = CustomStandardScaler()
    x_transformed = scaler.fit_transform(X)
    assert not np.isclose(0, X.mean())
    assert not np.isclose(1, X.std())
    assert np.isclose(0, x_transformed.mean())
    assert np.isclose(1, x_transformed.std())
