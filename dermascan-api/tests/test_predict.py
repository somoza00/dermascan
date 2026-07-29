import io

import pytest
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


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
