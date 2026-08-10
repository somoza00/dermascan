"""
Testes do pipeline real de inferência: risco (regra de negócio) e smoke
test do modelo treinado via rota /predict.

Os testes de modelo pulam (skip) quando o checkpoint não existe — ele é
gitignored e só está presente em máquinas que receberam o .pt (como esta).
CI sem o checkpoint continua verde; o comportamento do mock cobre o resto.
"""

import io
from pathlib import Path

import pytest
from PIL import Image

from app.main import app
from app.services.inference import DEFAULT_MODEL_PATH, RealInferenceService
from app.services.risk import CLASS_NAMES, build_prediction, derive_risk_level

# ─── Regra de negócio (mesmos casos de dermascan-model/tests/test_risk.py) ───


def _probs(**overrides) -> dict:
    remaining_classes = [c for c in CLASS_NAMES if c not in overrides]
    remaining_mass = 1.0 - sum(overrides.values())
    each = remaining_mass / len(remaining_classes) if remaining_classes else 0.0
    probs = {c: each for c in remaining_classes}
    probs.update(overrides)
    return probs


def test_high_risk_when_mel_is_top_class():
    assert derive_risk_level(_probs(MEL=0.85)) == "high"


def test_high_risk_when_mel_above_threshold_even_if_not_argmax():
    # NV é a classe de maior probabilidade, mas MEL > 30% já basta.
    assert derive_risk_level(_probs(NV=0.45, MEL=0.35)) == "high"


def test_low_risk_when_benign_wins_and_mel_below_threshold():
    assert derive_risk_level(_probs(NV=0.80, MEL=0.05)) == "low"


@pytest.mark.parametrize("malignant_class", ["BCC", "SCC"])
def test_high_risk_when_other_malignant_class_wins(malignant_class):
    assert derive_risk_level(_probs(**{malignant_class: 0.70, "MEL": 0.05})) == "high"


def test_medium_risk_when_precancerous_class_wins():
    assert derive_risk_level(_probs(AK=0.60, MEL=0.05)) == "medium"


def test_raises_on_missing_class():
    incomplete = {c: 0.125 for c in CLASS_NAMES[:-1]}
    with pytest.raises(ValueError):
        derive_risk_level(incomplete)


def test_build_prediction_matches_api_response_shape():
    prediction = build_prediction(_probs(MEL=0.85))
    assert set(prediction) == {"risk_level", "label", "confidence", "recommendation"}
    assert prediction["risk_level"] == "high"
    assert prediction["label"] == "Melanoma"
    assert 0.0 <= prediction["confidence"] <= 1.0


# ─── Pipeline real (modelo treinado) ───

HAS_CHECKPOINT = Path(DEFAULT_MODEL_PATH).is_file()


def _make_png_bytes(size: int = 300) -> bytes:
    """Imagem sintética com gradiente — suficiente pra validar o pipeline
    (imagem -> pré-processamento -> modelo -> regra de risco)."""
    buf = io.BytesIO()
    img = Image.linear_gradient("L").convert("RGB").resize((size, size))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint dermascan_v1.pt ausente")
def test_real_service_loads_checkpoint_and_predicts():
    import asyncio

    service = RealInferenceService()
    result = asyncio.run(service.predict(_make_png_bytes()))

    # O serviço real sempre responde com o contrato do schema.
    assert result.risk_level in {"high", "medium", "low"}
    assert result.label
    assert 0.0 <= result.confidence <= 1.0
    assert result.recommendation


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint dermascan_v1.pt ausente")
def test_real_service_reports_model_metadata_from_checkpoint():
    """Fonte da verdade é o checkpoint: 8 classes, 300x300, ImageNet stats."""
    service = RealInferenceService()
    service._ensure_loaded()
    assert service._classes == CLASS_NAMES
    assert service._image_size == 300
    assert service._mean == [0.485, 0.456, 0.406]


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint dermascan_v1.pt ausente")
def test_predict_end_to_end_with_real_model():
    """E2E via rota HTTP (sem override do serviço): a app sobe o serviço
    real sozinha, porque o checkpoint existe."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/predict", files={"file": ("lesion.png", _make_png_bytes(), "image/png")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in {"high", "medium", "low"}
    assert body["label"]
    assert 0.0 <= body["confidence"] <= 1.0
