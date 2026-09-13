import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import predict as predict_route
from app.services.inference import RealInferenceService, get_inference_service

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class _PayloadTooLarge(Exception):
    """Sinal interno: o corpo do request ultrapassou o limite durante o stream."""


# Buffer reservado para o envelope multipart (boundaries + headers de cada
# parte) além do limite de arquivo imposto em routes/predict.MAX_UPLOAD_SIZE.
# Garante que qualquer imagem que passe o limite de arquivo da rota também
# caiba no corpo (que inclui o overhead do multipart), sem abrir espaço pra
# um spool descontrolado no disco do servidor.
_UPLOAD_ENVELOPE_BUFFER = 1_024 * 1024  # 1MB


def _upload_body_limit() -> int:
    """Limite do corpo do request: arquivo + envelope multipart."""
    return predict_route.MAX_UPLOAD_SIZE + _UPLOAD_ENVELOPE_BUFFER


class MaxUploadSizeMiddleware:
    """Rejeita uploads acima do limite ANTES do parser multipart.

    O parser do Starlette já spoola o corpo do multipart num arquivo
    temporário (que rola pra disco acima de ~500KB) antes de a rota ler o
    UploadFile — então o limite de 10MB em chunks em routes/predict.py
    protege a RAM, mas não o disco. Este middleware corta no nível do
    protocolo:
    (1) responde 413 de imediato quando o header Content-Length excede o
    limite, sem parse/spool nenhum;
    (2) como defesa contra corpos chunked sem Content-Length, limita os
    bytes recebidos durante o stream, contendo o spool do disco no limite.
    """

    def __init__(self, app):
        self.app = app

    async def _send_413(self, send):
        body = json.dumps({"detail": "Imagem muito grande. Máx 10MB."}).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = _upload_body_limit()
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > limit:
                await self._send_413(send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _PayloadTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _PayloadTooLarge:
            await self._send_413(send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o checkpoint no boot em vez de na 1ª request: sem isso, o
    # primeiro usuário real pagaria o custo de carregar (e possivelmente
    # baixar via MODEL_PATH) o modelo — alguns segundos ou mais. Também
    # falha cedo, com log claro no startup, se o checkpoint estiver
    # ausente/corrompido, em vez de na primeira predição de alguém.
    service = get_inference_service()
    if isinstance(service, RealInferenceService):
        await asyncio.to_thread(service._ensure_loaded)
    yield


app = FastAPI(
    title="DermaScan API",
    description="API de classificação de lesões de pele",
    version="0.1.0",
    lifespan=lifespan,
)

# Origens liberadas para o frontend. Em dev, cai para localhost por padrão;
# em produção, defina CORS_ORIGINS (lista separada por vírgula) com a URL
# real do frontend (ex.: o domínio do Vercel) — nunca "*" com upload de
# arquivo exposto publicamente.
_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:3000"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(MaxUploadSizeMiddleware)

app.include_router(predict_route.router)


@app.get("/health")
async def health_check():
    # Reporta o modo de inferência de verdade em vez de um "ok" estático:
    # ALLOW_MOCK_INFERENCE permite subir em modo mock intencionalmente (dev/
    # CI/demo), e sem isso no corpo da resposta não haveria como distinguir
    # de fora se a API está servindo o modelo treinado ou dados sintéticos.
    service = get_inference_service()
    mode = "real" if isinstance(service, RealInferenceService) else "mock"
    return {"status": "ok", "version": app.version, "inference_mode": mode}
