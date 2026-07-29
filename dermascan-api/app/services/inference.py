"""
Serviço de inferência.

`InferenceService` é o contrato (Protocol) que qualquer implementação
precisa seguir. Hoje só existe `MockInferenceService`; quando o modelo
real (EfficientNet-B3 treinado) estiver pronto, ele vira uma nova classe
que implementa o mesmo `predict()`, e a troca acontece só em
`get_inference_service()` — nenhuma rota precisa mudar.
"""

import logging
import random
from typing import Protocol

from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)


class InferenceService(Protocol):
    """Contrato: recebe bytes de imagem já validados, devolve uma predição."""

    async def predict(self, image_bytes: bytes) -> PredictionResponse: ...


# Lista de condições de pele que o mock simula
CONDICOES = [
    {"label": "Melanoma", "risk": "high", "conf_range": (0.75, 0.98),
     "recom": "Consulte um dermatologista com urgência para avaliação."},
    {"label": "Carcinoma Basocelular", "risk": "high", "conf_range": (0.70, 0.95),
     "recom": "Agende consulta com dermatologista o quanto antes."},
    {"label": "Nevo Benigno (Pinta)", "risk": "low", "conf_range": (0.80, 0.99),
     "recom": "Aparentemente benigno. Monitore alterações e consulte se mudar."},
    {"label": "Ceratose Seborreica", "risk": "low", "conf_range": (0.75, 0.98),
     "recom": "Lesão benigna comum. Apenas acompanhamento de rotina."},
    {"label": "Dermatofibroma", "risk": "low", "conf_range": (0.70, 0.95),
     "recom": "Lesão benigna. Não requer tratamento, mas observe."},
    {"label": "Carcinoma Espinocelular", "risk": "high", "conf_range": (0.72, 0.96),
     "recom": "Necessita avaliação dermatológica prioritária."},
    {"label": "Lesão Actínica (Pré-cancerosa)", "risk": "medium", "conf_range": (0.65, 0.90),
     "recom": "Acompanhamento dermatológico recomendado nos próximos meses."},
]


class MockInferenceService:
    """Implementação mock — usada enquanto o modelo real não está pronto."""

    async def predict(self, image_bytes: bytes) -> PredictionResponse:
        """
        Simula uma predição.
        No futuro: carrega o modelo, pré-processa a imagem, roda inferência.
        """
        # Mock: escolhe uma condição aleatória baseada no "conteúdo" da imagem
        # Quanto maior a imagem, maior a chance de ser algo "preocupante" (só pro mock)
        tamanho = len(image_bytes)
        if tamanho > 500_000:
            pool = [c for c in CONDICOES if c["risk"] == "high"]
        elif tamanho > 200_000:
            pool = [c for c in CONDICOES if c["risk"] in ("medium", "high")]
        else:
            pool = CONDICOES

        condicao = random.choice(pool)
        confidence = round(random.uniform(*condicao["conf_range"]), 3)

        logger.info(
            "predição mock: label=%s risk=%s confidence=%.3f",
            condicao["label"], condicao["risk"], confidence,
        )

        return PredictionResponse(
            risk_level=condicao["risk"],
            label=condicao["label"],
            confidence=confidence,
            recommendation=condicao["recom"],
        )


# Instância única (singleton) do serviço ativo hoje.
_inference_service: InferenceService = MockInferenceService()


def get_inference_service() -> InferenceService:
    """Dependency injection point usado pela rota via `Depends()`.

    Trocar mock -> modelo real é reatribuir `_inference_service` (ou ler de
    config); em testes, sobrescreva com
    `app.dependency_overrides[get_inference_service] = lambda: FakeService()`.
    """
    return _inference_service
