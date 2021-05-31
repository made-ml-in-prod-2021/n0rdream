import pytest
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

HOST = "0.0.0.0"
PORT = 8000


@pytest.fixture()
def url_root() -> str:
    return f"http://{HOST}:{PORT}/"


@pytest.fixture()
def url_predict(url_root) -> str:
    return f"{url_root}predict/"


@pytest.fixture()
def requests_session() -> Session:
    session = Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    return session
