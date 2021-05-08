from src.data import read_dataset, split_dataset
from src.parameters import SplittingParams


def test_split_dataset(fake_train_dataset_path: str, splitting_params):
    df = read_dataset(fake_train_dataset_path)
    df_train, df_valid = split_dataset(df, splitting_params)
    assert df_train.shape[0] > 1
    assert df_valid.shape[0] > 1
