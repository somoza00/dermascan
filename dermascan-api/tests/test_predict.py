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


def test_middleware_rejects_oversized_content_length():
    """Content-Length acima do limite -> 413 sem nunca chamar a app downstream."""
    import asyncio

    from app.main import MaxUploadSizeMiddleware
    from app.routes import predict as predict_route

    limit = predict_route.MAX_UPLOAD_SIZE + (1024 * 1024)
    downstream_called: list[str] = []

    async def dummy_downstream(scope, receive, send):
        downstream_called.append(scope["path"])

    middleware = MaxUploadSizeMiddleware(dummy_downstream)
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [(b"content-length", str(limit + 1).encode())],
    }
    asyncio.run(middleware(scope, receive, send))

    assert downstream_called == []
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 413


def test_middleware_caps_body_without_content_length():
    """Corpo chunked (sem Content-Length) ficando maior que o limite -> 413.

    O limite em chunks em predict.py protege a RAM, mas é o parser multipart
    que spoola o corpo num arquivo temporário antes da rota. Este teste
    garante que o middleware corta o corpo recebido durante o stream.
    """
    import asyncio

    from app.main import MaxUploadSizeMiddleware
    from app.routes import predict as predict_route

    limit = predict_route.MAX_UPLOAD_SIZE + (1024 * 1024)

    async def dummy_consumer(scope, receive, send):
        # Consome o corpo como o parser multipart faria, até o fim da stream.
        while True:
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body"):
                break

    middleware = MaxUploadSizeMiddleware(dummy_consumer)
    chunks = iter([
        {"type": "http.request", "body": b"x" * (limit + 8), "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
    ])
    messages: list[dict] = []

    async def receive():
        return next(chunks)

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": "POST", "path": "/predict", "headers": []}
    asyncio.run(middleware(scope, receive, send))

    starts = [m for m in messages if m["type"] == "http.response.start"]
    assert starts and starts[0]["status"] == 413
