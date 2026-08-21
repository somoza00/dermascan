import io
from collections import defaultdict, deque

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas.prediction import PredictionResponse
from app.services.inference import get_inference_service


class FakeInferenceService:
    """Substitui o mock em teste — desacopla os testes de rota do random.choice."""

    async def predict(self, image_bytes: bytes) -> PredictionResponse:
        return PredictionResponse(
            risk_level="low",
            label="Nevo Benigno (Pinta)",
        confidence=0.95,
        recommendation="Apenas acompanhamento de rotina.",
        inference_mode="real",
        )


@pytest.fixture
def client():
    app.dependency_overrides[get_inference_service] = lambda: FakeInferenceService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _make_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buf, format="PNG")
    return buf.getvalue()


def test_predict_happy_path(client):
    files = {"file": ("lesion.png", _make_png_bytes(), "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "low"
    assert body["label"] == "Nevo Benigno (Pinta)"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["inference_mode"] == "real"


def test_predict_rejects_invalid_content_type(client):
    files = {"file": ("lesion.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400


def test_predict_rejects_empty_file(client):
    files = {"file": ("lesion.png", b"", "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400


def test_predict_rejects_oversized_file(client):
    too_big = b"\x00" * (10 * 1024 * 1024 + 1)
    files = {"file": ("lesion.png", too_big, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 413


def test_predict_rejects_fake_image_with_spoofed_content_type(client):
    """Content-Type mentindo que bytes arbitrários são um PNG real."""
    files = {"file": ("lesion.png", b"definitely-not-a-real-png", "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400


def test_predict_rejects_image_with_too_many_pixels(client, monkeypatch):
    """O limite é próprio, não depende de Pillow emitir apenas warning."""
    from app.routes import predict as predict_route

    class FakeImage:
        size = (5_001, 5_000)

        def verify(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(predict_route.Image, "open", lambda *_args: FakeImage())
    files = {"file": ("lesion.png", b"small-but-malicious", "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "dimensões" in response.json()["detail"]


def test_rate_limit_rejects_request_above_window(monkeypatch):
    from app.routes import predict as predict_route

    monkeypatch.setattr(predict_route, "MAX_PREDICTIONS_PER_WINDOW", 1)
    monkeypatch.setattr(predict_route, "_requests_by_client", defaultdict(deque))
    predict_route._enforce_rate_limit("test-client")

    with pytest.raises(HTTPException) as exc_info:
        predict_route._enforce_rate_limit("test-client")

    assert exc_info.value.status_code == 429


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
