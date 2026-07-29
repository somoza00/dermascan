import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="DermaScan API",
    description="API de classificação de lesões de pele",
    version="0.1.0",
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
    return {"status": "ok", "version": app.version}
