from typing import Literal

from pydantic import BaseModel

RiskLevel = Literal["high", "medium", "low"]
InferenceMode = Literal["real", "mock"]


class PredictionResponse(BaseModel):
    """Resposta da predição"""
    risk_level: RiskLevel
    label: str
    confidence: float
    recommendation: str
    # Nunca omita a procedência do resultado: o frontend precisa conseguir
    # impedir que uma simulação seja interpretada como avaliação do modelo.
    inference_mode: InferenceMode


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada"""
    detail: str
