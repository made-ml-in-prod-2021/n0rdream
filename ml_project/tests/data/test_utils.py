from src.data import read_dataset


def test_load_dataset(fake_train_dataset_path: str, target_col: str):
    data = read_dataset(fake_train_dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()
