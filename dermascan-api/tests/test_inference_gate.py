"""
Cobre o gate ALLOW_MOCK_INFERENCE (`_build_service`), os erros distinguíveis
de `_ensure_loaded`, e o campo `inference_mode` de `/health`.

Objetivo de negócio: a API nunca deve subir servindo predições sintéticas
por acidente (ex. MODEL_PATH mal configurado em produção) — só quando
ALLOW_MOCK_INFERENCE=true é definido explicitamente.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.services.inference as inference_module
from app.main import app
from app.services.inference import (
    DEFAULT_MODEL_PATH,
    MockInferenceService,
    RealInferenceService,
    _build_service,
)

# ─── Gate ALLOW_MOCK_INFERENCE ───


@pytest.mark.parametrize("flag_value", [None, "", "false", "False", "0"])
def test_build_service_raises_without_mock_gate(tmp_path, monkeypatch, flag_value):
    """Checkpoint ausente + flag não habilitada (ou ausente/falsy) -> recusa subir."""
    missing = tmp_path / "does-not-exist.pt"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    if flag_value is None:
        monkeypatch.delenv("ALLOW_MOCK_INFERENCE", raising=False)
    else:
        monkeypatch.setenv("ALLOW_MOCK_INFERENCE", flag_value)

    with pytest.raises(RuntimeError, match="ALLOW_MOCK_INFERENCE"):
        _build_service()


@pytest.mark.parametrize("flag_value", ["true", "True", "TRUE", "1", "yes", "on"])
def test_build_service_allows_mock_when_flag_truthy(tmp_path, monkeypatch, flag_value):
    """Confirma parsing correto de bool: só valores realmente truthy liberam o mock."""
    missing = tmp_path / "does-not-exist.pt"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    monkeypatch.setenv("ALLOW_MOCK_INFERENCE", flag_value)

    service = _build_service()

    assert isinstance(service, MockInferenceService)


def test_build_service_prefers_real_checkpoint_over_mock_gate(monkeypatch):
    """Checkpoint disponível vence independente de ALLOW_MOCK_INFERENCE."""
    if not Path(DEFAULT_MODEL_PATH).is_file():
        pytest.skip("checkpoint dermascan_v1.pt ausente")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("ALLOW_MOCK_INFERENCE", "true")

    service = _build_service()

    assert isinstance(service, RealInferenceService)


# ─── Erros distinguíveis de _ensure_loaded ───


def test_ensure_loaded_raises_for_missing_file(tmp_path):
    service = RealInferenceService(model_path=str(tmp_path / "missing.pt"))

    with pytest.raises(RuntimeError, match="não encontrado"):
        service._ensure_loaded()


def test_ensure_loaded_raises_for_corrupted_file(tmp_path):
    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"isto nao e um checkpoint valido")
    service = RealInferenceService(model_path=str(bad))

    with pytest.raises(RuntimeError, match="corrompido"):
        service._ensure_loaded()


def test_ensure_loaded_raises_for_permission_denied(tmp_path, monkeypatch):
    def _raise_permission_error(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(inference_module.torch, "load", _raise_permission_error)
    service = RealInferenceService(model_path=str(tmp_path / "whatever.pt"))

    with pytest.raises(RuntimeError, match="permissão"):
        service._ensure_loaded()


# ─── /health reporta o modo de inferência real ───


def test_health_reports_mock_mode(monkeypatch):
    monkeypatch.setattr(inference_module, "_inference_service", None)
    monkeypatch.setattr(inference_module, "_build_service", lambda: MockInferenceService())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["inference_mode"] == "mock"


def test_health_reports_real_mode(monkeypatch):
    class FakeRealService(RealInferenceService):
        def _ensure_loaded(self):
            return  # evita depender de um checkpoint real no lifespan

    monkeypatch.setattr(inference_module, "_inference_service", None)
    monkeypatch.setattr(inference_module, "_build_service", lambda: FakeRealService())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["inference_mode"] == "real"
