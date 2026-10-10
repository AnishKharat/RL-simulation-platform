import pytest
from fastapi.testclient import TestClient

from rcl.api.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}", artifact_root=str(tmp_path / "a")
    )
    with TestClient(app) as c:
        yield c
