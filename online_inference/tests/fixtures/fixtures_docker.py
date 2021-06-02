import os

import docker
from docker import DockerClient
from docker.models.containers import Container
import pytest
from py._path.local import LocalPath


@pytest.fixture(scope='session')
def docker_client() -> DockerClient:
    return docker.from_env(version='auto')


@pytest.fixture(scope="session")
def dockerfile_path() -> LocalPath:
    fixture_path = os.path.dirname(os.path.realpath(__file__))
    tests_path = os.path.dirname(fixture_path)
    return os.path.dirname(tests_path)


@pytest.fixture(scope="session")
def api_container(
    docker_client: DockerClient,
    dockerfile_path: LocalPath,
) -> Container:
    docker_client.images.build(
        path=dockerfile_path,
        tag='api',
        rm=True,
    )
    container = docker_client.containers.run(
        image='api',
        detach=True,
        network_mode='host',
    )
    yield container
    container.kill(signal=9)
    container.remove(force=True)
