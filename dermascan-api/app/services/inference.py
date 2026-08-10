"""
Serviço de inferência.

`InferenceService` é o contrato (Protocol) que qualquer implementação
precisa seguir. Existem duas implementações:

- `MockInferenceService` — simula predição; usado como fallback quando o
  checkpoint treinado não está disponível (dev sem modelo, CI, etc).
- `RealInferenceService` — carrega o checkpoint `dermascan_v1.pt`
  (exportado por `dermascan-model/src/evaluate.py::export_model`) e roda
  inferência de verdade: reconstroi a arquitetura, aplica o mesmo
  pré-processamento do treino (resolução/normalização lidas DO CHECKPOINT,
  não hardcoded) e deriva risco via `risk.py::build_prediction`.

`get_inference_service()` escolhe automaticamente: se o checkpoint existe
localmente (ou `MODEL_PATH` aponta pra ele / uma URL http(s)), usa o real;
senão, cai pro mock com warning. Nenhuma rota precisa mudar — a troca já
estava prevista desde o início (Protocol + DI point).
"""

import asyncio
import io
import logging
import os
import random
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from torchvision import transforms

from app.schemas.prediction import PredictionResponse
from app.services.model import DermaScanModel
from app.services.risk import build_prediction

logger = logging.getLogger(__name__)

# Caminho padrão do checkpoint. Relativo ao CWD: local (uvicorn rodando da
# raiz de dermascan-api) e no Docker (WORKDIR /app) resolvem para
# dermascan-api/models/dermascan_v1.pt. Sobrescreva com MODEL_PATH.
DEFAULT_MODEL_PATH = "models/dermascan_v1.pt"


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
    """Implementação mock — fallback enquanto o modelo real não está carregado."""

    async def predict(self, image_bytes: bytes) -> PredictionResponse:
        """Simula uma predição baseada no tamanho do arquivo (só pro mock)."""
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


def build_transform(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    """Pré-processamento idêntico ao de validação do treino
    (`build_transform(image_size, train=False)` em
    `dermascan-model/src/dataset.py`): Resize mantendo proporção (lado
    menor vira `image_size`, sem esticar — assimetria/borda são sinal
    diagnóstico do ABCDE), CenterCrop quadrado, ToTensor, Normalize.

    `image_size`/`mean`/`std` são lidos do checkpoint (fonte da verdade),
    não hardcoded — assim o serviço replica exatamente o que o modelo viu
    no treino mesmo que esses valores mudem numa versão futura.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class RealInferenceService:
    """Inferência real com o checkpoint treinado (EfficientNet-B3, ISIC 2019).

    Carga é LAZY (primeira predição) e protegida por lock: a primeira
    request paga o custo de carregar ~46MB de pesos (alguns segundos em
    CPU); as seguintes reutilizam o modelo em memória.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH
        self._model: torch.nn.Module | None = None
        self._classes: list[str] = []
        self._image_size: int = 300
        self._mean: list[float] = [0.485, 0.456, 0.406]
        self._std: list[float] = [0.229, 0.224, 0.225]
        self._load_lock = threading.Lock()

    @staticmethod
    def is_available(model_path: str | None = None) -> bool:
        """True se o checkpoint está presente (ou é uma URL — nesse caso o
        download acontece sob demanda na primeira predição)."""
        path = model_path or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH
        return path.startswith(("http://", "https://")) or Path(path).is_file()

    def _resolve_local_path(self) -> Path:
        """Se `model_path` é uma URL, baixa o checkpoint para o cache local
        (/tmp) e retorna o caminho. Isso permite deploy sem o .pt no
        repositório (ex.: Railway com MODEL_PATH apontando pra um storage
        público) sem mudar mais nada.
        """
        if not self.model_path.startswith(("http://", "https://")):
            return Path(self.model_path)

        cache_path = Path(tempfile.gettempdir()) / "dermascan_v1.pt"
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            logger.info("checkpoint já em cache: %s", cache_path)
            return cache_path

        logger.info("baixando checkpoint de %s ...", self.model_path)
        urllib.request.urlretrieve(self.model_path, cache_path)
        logger.info("checkpoint baixado: %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)
        return cache_path

    def _ensure_loaded(self) -> None:
        """Carrega o checkpoint uma única vez (thread-safe)."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            local_path = self._resolve_local_path()
            # weights_only=True: o checkpoint só precisa conter tensors +
            # tipos primitivos (confirmado — nenhuma classe custom), então
            # não há motivo pra habilitar unpickling irrestrito. Isso importa
            # de verdade aqui porque `model_path` pode vir de uma URL
            # (`MODEL_PATH`) — sem essa flag, um checkpoint malicioso
            # servido por um host comprometido/MITM executaria código
            # arbitrário no load, não só pesos errados.
            checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)

            # Valida o contrato salvo por export_model() — se faltar alguma
            # chave, o checkpoint é de outra versão do pipeline e o erro é
            # claro em vez de um KeyError obscuro no meio do forward.
            required = {"model_state_dict", "classes", "num_classes", "image_size",
                        "normalize_mean", "normalize_std"}
            missing = required - set(checkpoint)
            if missing:
                raise RuntimeError(
                    f"Checkpoint {local_path} incompleto — faltam: {sorted(missing)}. "
                    "Re-exporte com export_model() do dermascan-model."
                )

            model = DermaScanModel(num_classes=checkpoint["num_classes"])
            # strict=True: qualquer shape diferente do treino falha aqui,
            # nunca silenciosamente no meio da predição.
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()

            self._model = model
            self._classes = list(checkpoint["classes"])
            self._image_size = int(checkpoint["image_size"])
            self._mean = list(checkpoint["normalize_mean"])
            self._std = list(checkpoint["normalize_std"])
            logger.info(
                "modelo carregado: %d classes, resolução %dx%d, device=cpu",
                len(self._classes), self._image_size, self._image_size,
            )

    async def predict(self, image_bytes: bytes) -> PredictionResponse:
        """Inferência roda em thread separada (asyncio.to_thread) — nunca
        bloqueia o event loop do FastAPI, que é compartilhado com outras
        requests."""
        return await asyncio.to_thread(self._predict_sync, image_bytes)

    def _predict_sync(self, image_bytes: bytes) -> PredictionResponse:
        self._ensure_loaded()
        model = self._model
        assert model is not None  # garantido por _ensure_loaded()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = build_transform(self._image_size, self._mean, self._std)
        tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].numpy()

        class_probabilities = dict(zip(self._classes, probabilities))
        prediction = build_prediction(class_probabilities)

        logger.info(
            "predição real: top=%s risk=%s confidence=%.3f",
            prediction["label"], prediction["risk_level"], prediction["confidence"],
        )
        return PredictionResponse(**prediction)


# Instância única (singleton) — resolvida na primeira chamada.
_inference_service: InferenceService | None = None
_service_lock = threading.Lock()


def _build_service() -> InferenceService:
    """Decide qual implementação usar. Se o checkpoint estiver disponível
    (arquivo local ou MODEL_PATH/URL), usa o real; senão, mock com warning
    bem visível — a API continua respondendo em qualquer ambiente, mas deixa
    claro que a resposta não é de um modelo treinado."""
    path = os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH
    if RealInferenceService.is_available(path):
        logger.info("RealInferenceService ativo (checkpoint: %s)", path)
        return RealInferenceService(path)

    logger.warning(
        "Checkpoint do modelo não encontrado em %r — usando MockInferenceService. "
        "Coloque o .pt em dermascan-api/models/ ou defina MODEL_PATH.",
        path,
    )
    return MockInferenceService()


def get_inference_service() -> InferenceService:
    """Dependency injection point usado pela rota via `Depends()`.

    Em testes, sobrescreva com
    `app.dependency_overrides[get_inference_service] = lambda: FakeService()`.
    """
    global _inference_service
    if _inference_service is None:
        with _service_lock:
            if _inference_service is None:
                _inference_service = _build_service()
    return _inference_service
