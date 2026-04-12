from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.db.sqlite_db import init_db


@pytest.fixture(autouse=True)
def isolate_db(tmp_path: Path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    init_db()
    yield
    os.chdir(old_cwd)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
