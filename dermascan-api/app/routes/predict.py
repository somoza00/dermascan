import io
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.schemas.prediction import ErrorResponse, PredictionResponse
from app.services.inference import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predição"])

ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
_READ_CHUNK_SIZE = 1024 * 1024  # 1MB
# O limite de bytes não protege contra PNG/JPEG altamente comprimido com
# dezenas de milhões de pixels. 25 MP permite fotos grandes (ex. 5.000 x
# 5.000), mas limita o pico de RAM antes da conversão para RGB/inferência.
MAX_IMAGE_PIXELS = 25_000_000
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_PREDICTIONS_PER_WINDOW = int(os.getenv("MAX_PREDICTIONS_PER_MINUTE", "20"))
_requests_by_client: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()

if MAX_PREDICTIONS_PER_WINDOW < 1:
    raise RuntimeError("MAX_PREDICTIONS_PER_MINUTE deve ser maior que zero.")


async def _read_within_limit(file: UploadFile, max_size: int) -> bytes:
    """Lê o upload em chunks, abortando assim que ultrapassa `max_size`.

    Ler o corpo inteiro antes de checar o tamanho permitiria que um cliente
    forçasse o servidor a bufferizar um upload arbitrariamente grande antes
    de rejeitá-lo (vetor de DoS barato). Lendo em chunks, o request é
    cortado assim que excede o limite.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Imagem muito grande. Máx {max_size // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_is_image(image_bytes: bytes) -> None:
    """Valida o conteúdo real do arquivo (magic bytes via Pillow), não só o
    header `Content-Type` — que é definido pelo cliente e é trivial de
    falsificar. Também rejeita decompression bombs (imagem pequena em bytes
    mas com dimensões de pixel absurdas). Não dependemos só do warning
    padrão do Pillow: warnings não interrompem a request.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Imagem excede o limite de dimensões permitido "
                        f"({MAX_IMAGE_PIXELS:,} pixels)."
                    ),
                )
            img.verify()
    except Image.DecompressionBombError as exc:
        raise HTTPException(
            status_code=400,
            detail="Imagem excede o limite de dimensões permitido.",
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado não é uma imagem válida.",
        ) from exc


def _enforce_rate_limit(client_id: str) -> None:
    """Limite por processo para evitar abuso acidental ou trivial.

    Para múltiplas réplicas, o mesmo limite deve existir no gateway/CDN, que
    enxerga todos os processos. Esta barreira protege inclusive o modo local
    e não registra imagem ou outros dados de saúde.
    """
    now = time.monotonic()
    with _rate_limit_lock:
        recent = _requests_by_client[client_id]
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while recent and recent[0] <= cutoff:
            recent.popleft()
        if len(recent) >= MAX_PREDICTIONS_PER_WINDOW:
            raise HTTPException(
                status_code=429,
                detail="Muitas análises em pouco tempo. Aguarde um minuto e tente novamente.",
            )
        recent.append(now)


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File()],
    inference_service: Annotated[InferenceService, Depends(get_inference_service)],
):
    """
    Envie uma imagem de lesão de pele e receba a classificação.

    - **file**: imagem (.jpg, .jpeg, .png), máx 10MB
    """
    client_id = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_id)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Envie JPEG ou PNG.",
        )

    image_bytes = await _read_within_limit(file, MAX_UPLOAD_SIZE)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Imagem vazia.")

    _validate_is_image(image_bytes)

    logger.info("predição solicitada: filename=%s size=%d bytes", file.filename, len(image_bytes))

    return await inference_service.predict(image_bytes)
