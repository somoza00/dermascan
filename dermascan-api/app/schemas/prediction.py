from typing import Literal

from pydantic import BaseModel

RiskLevel = Literal["high", "medium", "low"]


class PredictionResponse(BaseModel):
    """Resposta da predição"""
    risk_level: RiskLevel
    label: str
    confidence: float
    recommendation: str


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada"""
    detail: str
