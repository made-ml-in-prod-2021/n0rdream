from docker.models.containers import Container
from requests import Session


def test_api_root(
    api_container: Container,
    requests_session: Session,
    url_root: str,
):
    response = requests_session.get(url_root)
    assert 200 == response.status_code
    assert "Service is working" in response.text


def test_api_predict(
    api_container: Container,
    requests_session: Session,
    fake_test_dataset_json: str,
    url_predict: str,
):
    data = {"data": fake_test_dataset_json}
    response = requests_session.get(url_predict, json=data)
    assert 200 == response.status_code
    preds = response.json()
    assert {0, 1} == set(preds)
