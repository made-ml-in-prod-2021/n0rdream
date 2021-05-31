from typing import List

import pytest

from helpers import generate_fake_df


@pytest.fixture
def feature_columns() -> List[str]:
    return [
        'age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
    ]


@pytest.fixture()
def fake_test_dataset_json(feature_columns: List) -> str:
    fake_df = generate_fake_df(feature_columns, size=1111)
    return fake_df.to_json()
