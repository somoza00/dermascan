import os

import pytest


@pytest.fixture(autouse=True)
def _allow_mock_inference_by_default(monkeypatch):
    """Testes que sobem a app via `TestClient` (ex. test_predict.py) disparam
    o lifespan, que chama `get_inference_service()` diretamente —
    `app.dependency_overrides` não alcança essa chamada. Em CI, sem o
    checkpoint `dermascan_v1.pt` (gitignored), a app só sobe com
    `ALLOW_MOCK_INFERENCE=true`. `os.environ.get(...)` preserva um valor já
    definido no ambiente (ex. por um teste que quer exercitar o gate em si,
    via `monkeypatch.setenv`/`delenv` depois desta fixture rodar).
    """
    monkeypatch.setenv("ALLOW_MOCK_INFERENCE", os.environ.get("ALLOW_MOCK_INFERENCE", "true"))
