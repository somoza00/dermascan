import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.schemas.prediction import ErrorResponse, PredictionResponse
from app.services.inference import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predição"])

ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
_READ_CHUNK_SIZE = 1024 * 1024  # 1MB


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
    mas com dimensões de pixel absurdas), via o limite padrão do Pillow.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
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


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
    },
)
async def predict(
    file: Annotated[UploadFile, File()],
    inference_service: Annotated[InferenceService, Depends(get_inference_service)],
):
    """
    Envie uma imagem de lesão de pele e receba a classificação.

    - **file**: imagem (.jpg, .jpeg, .png), máx 10MB
    """
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
