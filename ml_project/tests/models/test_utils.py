import os

from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier

from src.models import save_model_to_pickle, load_model_from_pickle


def test_serialize_model(model_path: LocalPath):
    model = RandomForestClassifier()
    save_model_to_pickle(model, model_path)
    assert os.path.exists(model_path)
    model = load_model_from_pickle(model_path)
    assert isinstance(model, RandomForestClassifier)
