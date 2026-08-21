import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router
from app.services.inference import RealInferenceService, get_inference_service

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


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

app.include_router(predict_router)


@app.get("/health")
async def health_check():
    # Reporta o modo de inferência de verdade em vez de um "ok" estático:
    # ALLOW_MOCK_INFERENCE permite subir em modo mock intencionalmente (dev/
    # CI/demo), e sem isso no corpo da resposta não haveria como distinguir
    # de fora se a API está servindo o modelo treinado ou dados sintéticos.
    service = get_inference_service()
    mode = "real" if isinstance(service, RealInferenceService) else "mock"
    return {"status": "ok", "version": app.version, "inference_mode": mode}
