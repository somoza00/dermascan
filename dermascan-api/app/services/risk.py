"""
Regra de negócio: deriva o nível de risco (high/medium/low) a partir das
probabilidades por classe previstas pelo modelo, treinado nas 8 classes do
ISIC 2019.

ESPELHO DE `dermascan-model/src/risk.py`. A API é deployada como pacote
separado (Railway) e não pode importar de `dermascan-model` em runtime —
por isso a regra mais sensível do produto vive duplicada aqui. MANTENHA EM
SINCRONIA com o arquivo original; os testes de `tests/test_risk.py` na API
e no modelo devem continuar idênticos entre si.

Classes do ISIC 2019 (ordem alfabética = índice de saída do modelo):
    AK    - Ceratose Actínica       (pré-cancerosa)
    BCC   - Carcinoma Basocelular   (maligno)
    BKL   - Ceratose Benigna        (benigno)
    DF    - Dermatofibroma          (benigno)
    MEL   - Melanoma                (maligno, mais letal)
    NV    - Nevo Melanocítico       (benigno, classe majoritária do dataset)
    SCC   - Carcinoma Espinocelular (maligno)
    VASC  - Lesão Vascular          (benigno)
"""

from typing import Literal

RiskLevel = Literal["high", "medium", "low"]

# Ordem alfabética — DEVE bater com `SkinLesionDataset.classes` usado no
# treino (ver dermascan-model/src/dataset.py). O checkpoint exportado
# carrega a lista de classes; confira se vier diferente.
CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

MALIGNANT_CLASSES = {"MEL", "BCC", "SCC"}
PRECANCEROUS_CLASSES = {"AK"}

# Threshold conservador e proposital: mesmo que MEL não seja a classe de
# maior probabilidade, qualquer indício razoável de melanoma (>30% de
# confiança) já vira risco alto. Nesta aplicação, o custo de um falso
# positivo (uma consulta extra) é muito menor que o de um falso negativo
# (um melanoma não detectado).
MELANOMA_HIGH_RISK_THRESHOLD = 0.30

LABEL_PT = {
    "MEL": "Melanoma",
    "BCC": "Carcinoma Basocelular",
    "SCC": "Carcinoma Espinocelular",
    "AK": "Ceratose Actínica (Pré-cancerosa)",
    "NV": "Nevo Benigno (Pinta)",
    "BKL": "Ceratose Benigna",
    "DF": "Dermatofibroma",
    "VASC": "Lesão Vascular",
}

RECOMMENDATION_PT: dict[RiskLevel, str] = {
    "high": "Consulte um dermatologista com urgência para avaliação.",
    "medium": "Agende uma consulta dermatológica nos próximos meses.",
    "low": "Aparentemente benigno. Monitore alterações e consulte um dermatologista se mudar.",
}


def derive_risk_level(class_probabilities: dict[str, float]) -> RiskLevel:
    """
    Args:
        class_probabilities: mapa {nome_da_classe: probabilidade}, saída de
            um softmax sobre os logits do modelo (deve somar ~1.0 e conter
            todas as classes em `CLASS_NAMES`).

    Regra, em ordem de prioridade:
        1. P(MEL) > MELANOMA_HIGH_RISK_THRESHOLD -> "high", não importa o argmax.
        2. Classe de maior probabilidade é maligna (BCC/SCC) -> "high".
        3. Classe de maior probabilidade é pré-cancerosa (AK) -> "medium".
        4. Caso contrário (classe benigna vence) -> "low".
    """
    missing = set(CLASS_NAMES) - set(class_probabilities)
    if missing:
        raise ValueError(f"class_probabilities não contém as classes: {sorted(missing)}")

    if class_probabilities["MEL"] > MELANOMA_HIGH_RISK_THRESHOLD:
        return "high"

    top_class = max(class_probabilities, key=class_probabilities.get)

    if top_class in MALIGNANT_CLASSES:
        return "high"
    if top_class in PRECANCEROUS_CLASSES:
        return "medium"
    return "low"


def build_prediction(class_probabilities: dict[str, float]) -> dict:
    """Monta o payload completo (risk_level, label, confidence,
    recommendation) que a API expõe em `/predict` — mesmo formato do mock,
    pra troca mock -> real não exigir mudança no schema."""
    top_class = max(class_probabilities, key=class_probabilities.get)
    risk_level = derive_risk_level(class_probabilities)

    return {
        "risk_level": risk_level,
        "label": LABEL_PT[top_class],
        "confidence": round(class_probabilities[top_class], 3),
        "recommendation": RECOMMENDATION_PT[risk_level],
    }
