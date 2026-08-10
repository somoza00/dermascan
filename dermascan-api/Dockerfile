# Dockerfile da DermaScan API — build multi-stage, imagem final roda non-root

FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# --chown independe da permissão que o arquivo tinha no host: sem isso, um
# arquivo commitado com permissão restritiva (ex.: 600, dono-only) fica
# ilegível pro usuário non-root abaixo, e o container não sobe.
COPY --chown=app:app app/ ./app/
# Checkpoint treinado (gitignored — não vai pro repositório, mas entra na
# imagem). Em deploy sem o arquivo local, use MODEL_PATH=<url> (a API
# baixa sob demanda; ver services/inference.py).
COPY --chown=app:app models/ ./models/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health', timeout=2)" || exit 1

# Honra HOST/PORT injetados pela plataforma de deploy (ex.: Railway define $PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
