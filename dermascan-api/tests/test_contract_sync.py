"""
Garante que os "espelhos" em app/services/{risk,model}.py não divergem dos
originais em dermascan-model/src/{risk,model}.py.

A API é deployada como pacote separado (Railway builda só o contexto
dermascan-api/) e não pode importar de dermascan-model em runtime — por
isso risk.py e model.py existem fisicamente duplicados nos dois lugares
(ver os comentários "MANTENHA EM SINCRONIA" em cada arquivo). Um comentário
pedindo sincronia manual não impede ninguém de editar só um lado; este
teste transforma a divergência num erro de CI, em vez de um bug silencioso
na regra de negócio mais sensível do produto (classificação de risco de
lesão de pele).

Roda só quando o monorepo completo está presente (dermascan-model/ como
irmão de dermascan-api/) — num checkout que só tem a API (ex.: imagem de
deploy final), pula.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from efficientnet_pytorch import EfficientNet

MODEL_SRC = Path(__file__).resolve().parents[2] / "dermascan-model" / "src"

pytestmark = pytest.mark.skipif(
    not MODEL_SRC.is_dir(),
    reason="dermascan-model/ não está presente neste checkout "
    "(esperado em builds de deploy só da API)",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def original_risk():
    return _load_module("dermascan_model_src_risk", MODEL_SRC / "risk.py")


@pytest.fixture(scope="module")
def original_model():
    return _load_module("dermascan_model_src_model", MODEL_SRC / "model.py")


def test_risk_constants_match(original_risk):
    from app.services import risk as api_risk

    assert api_risk.CLASS_NAMES == original_risk.CLASS_NAMES
    assert api_risk.MALIGNANT_CLASSES == original_risk.MALIGNANT_CLASSES
    assert api_risk.PRECANCEROUS_CLASSES == original_risk.PRECANCEROUS_CLASSES
    assert api_risk.MELANOMA_HIGH_RISK_THRESHOLD == original_risk.MELANOMA_HIGH_RISK_THRESHOLD
    assert api_risk.LABEL_PT == original_risk.LABEL_PT
    assert api_risk.RECOMMENDATION_PT == original_risk.RECOMMENDATION_PT


@pytest.mark.parametrize(
    "probabilities",
    [
        {"MEL": 0.85},
        {"NV": 0.45, "MEL": 0.35},
        {"NV": 0.80, "MEL": 0.05},
        {"BCC": 0.70, "MEL": 0.05},
        {"SCC": 0.70, "MEL": 0.05},
        {"AK": 0.60, "MEL": 0.05},
        {"MEL": 0.30},  # limite exato do threshold — os dois lados devem concordar
        {"MEL": 0.30 + 1e-9},
    ],
)
def test_risk_decision_matches(original_risk, probabilities):
    """Mesma entrada -> mesma saída nos dois módulos, para os casos que
    definem a regra (limite do threshold de MEL incluído — é onde uma
    divergência de `>` vs `>=` entre os dois lados passaria despercebida)."""
    from app.services import risk as api_risk

    classes = api_risk.CLASS_NAMES
    remaining = [c for c in classes if c not in probabilities]
    remaining_mass = 1.0 - sum(probabilities.values())
    each = remaining_mass / len(remaining) if remaining else 0.0
    full_probs = {c: each for c in remaining}
    full_probs.update(probabilities)

    assert api_risk.build_prediction(full_probs) == original_risk.build_prediction(full_probs)


def test_model_architecture_matches(original_model, monkeypatch):
    """Compara os shapes do state_dict entre os dois `DermaScanModel` — se
    alguém mudar o head customizado (dropout, largura da camada oculta) só
    de um lado, um checkpoint exportado por um não carregaria (ou
    carregaria errado) no outro."""
    from app.services.model import DermaScanModel as ApiModel

    # Evita baixar pesos do ImageNet (rede, ~40MB) só pra comparar
    # arquitetura — from_name tem os mesmos shapes que from_pretrained,
    # só sem carregar os pesos.
    monkeypatch.setattr(EfficientNet, "from_pretrained", EfficientNet.from_name)

    api_model = ApiModel(num_classes=8)
    original = original_model.DermaScanModel(num_classes=8)

    api_shapes = {k: tuple(v.shape) for k, v in api_model.state_dict().items()}
    original_shapes = {k: tuple(v.shape) for k, v in original.state_dict().items()}
    assert api_shapes == original_shapes
