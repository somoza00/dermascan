import pytest

from src.risk import CLASS_NAMES, derive_risk_level, build_prediction


def _probs(**overrides) -> dict:
    """Distribuição de probabilidade sobre as 8 classes, com todo o resto
    dividido igualmente e os overrides sobrescrevendo classes específicas."""
    remaining_classes = [c for c in CLASS_NAMES if c not in overrides]
    remaining_mass = 1.0 - sum(overrides.values())
    each = remaining_mass / len(remaining_classes) if remaining_classes else 0.0
    probs = {c: each for c in remaining_classes}
    probs.update(overrides)
    return probs


def test_high_risk_when_mel_is_top_class():
    probs = _probs(MEL=0.85)
    assert derive_risk_level(probs) == "high"


def test_high_risk_when_mel_above_threshold_even_if_not_argmax():
    # NV é a classe de maior probabilidade, mas MEL > 30% já basta —
    # é exatamente a regra conservadora pedida para o produto.
    probs = _probs(NV=0.45, MEL=0.35)
    assert derive_risk_level(probs) == "high"


def test_low_risk_when_mel_below_threshold_and_benign_wins():
    probs = _probs(NV=0.80, MEL=0.05)
    assert derive_risk_level(probs) == "low"


@pytest.mark.parametrize("malignant_class", ["BCC", "SCC"])
def test_high_risk_when_other_malignant_class_wins(malignant_class):
    probs = _probs(**{malignant_class: 0.70, "MEL": 0.05})
    assert derive_risk_level(probs) == "high"


def test_medium_risk_when_precancerous_class_wins():
    probs = _probs(AK=0.60, MEL=0.05)
    assert derive_risk_level(probs) == "medium"


def test_raises_on_missing_class():
    incomplete = {c: 0.125 for c in CLASS_NAMES[:-1]}
    with pytest.raises(ValueError):
        derive_risk_level(incomplete)


def test_build_prediction_matches_api_response_shape():
    probs = _probs(MEL=0.85)
    prediction = build_prediction(probs)

    assert set(prediction) == {"risk_level", "label", "confidence", "recommendation"}
    assert prediction["risk_level"] == "high"
    assert prediction["label"] == "Melanoma"
    assert 0.0 <= prediction["confidence"] <= 1.0
