import pytest
from fastapi.testclient import TestClient

from webapp import app


@pytest.fixture()
def fastapi_client() -> TestClient:
    return TestClient(app)


def test_api_root(
    fastapi_client: TestClient,
    url_root: str,
):
    response = fastapi_client.get(url_root)
    assert 200 == response.status_code
    assert "Service is working" in response.text


def test_api_predict(
    fastapi_client: TestClient,
    fake_test_dataset_json: str,
    url_predict: str,
):
    data = {"data": fake_test_dataset_json}
    with fastapi_client:
        response = fastapi_client.get(url_predict, json=data)
    assert 200 == response.status_code
    preds = response.json()
    assert {0, 1} == set(preds)
