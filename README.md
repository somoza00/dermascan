# 🔬 DermaScan

Análise preliminar de lesões de pele com IA.

> ⚠️ **Este projeto é uma ferramenta de triagem visual e não substitui diagnóstico médico.** Os resultados não devem ser usados como base única para decisões de saúde — consulte um dermatologista.

## Estrutura

```
dermascan/
├── dermascan-model/   # Treinamento do modelo (PyTorch + EfficientNet)
├── dermascan-api/     # Backend FastAPI
└── dermascan-web/     # Frontend React + TypeScript
```

## Como rodar

### 1. API
```bash
cd dermascan-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Documentação interativa: http://localhost:8000/docs

### 2. Frontend
```bash
cd dermascan-web
npm install
npm run dev
```

Acessa em http://localhost:5173

### 3. Modelo
O treinamento roda no Google Colab:
1. Abre `dermascan-model/notebooks/colab_training.ipynb` no Colab
2. Runtime → Change runtime type → GPU
3. Executa célula por célula
4. O modelo salvo vai pra `dermascan-api/models/`

### Alternativa: Docker
Sobe API + frontend juntos, sem instalar Python/Node localmente:
```bash
docker compose up --build
```
- Frontend: http://localhost:5173
- API: http://localhost:8000

Sem o checkpoint treinado (`dermascan-api/models/dermascan_v1.pt`, gitignored
por ser ~46MB), a API sobe em modo mock. Pra usar o modelo real, coloque o
arquivo nesse caminho antes do `docker compose up --build`, ou defina
`MODEL_PATH=https://.../dermascan_v1.pt` (ex.: um GitHub Release) — a API
baixa sozinha na primeira predição.

## API

`POST /predict` — envia imagem, recebe classificação:
- `risk_level`: high | medium | low
- `label`: nome da condição
- `confidence`: confiança (0-1)
- `recommendation`: recomendação médica

O serviço escolhe sozinho entre modelo real e mock: se
`dermascan-api/models/dermascan_v1.pt` existir (ou a env `MODEL_PATH`
apontar pra um arquivo/URL), roda o EfficientNet-B3 treinado; senão, cai
pro mock com um warning no log. Detalhes em
`dermascan-api/app/services/inference.py`.

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Modelo | PyTorch + EfficientNet-B3 |
| Treino | Google Colab (GPU) |
| API | FastAPI + Uvicorn |
| Frontend | React + TypeScript + Vite |
| Deploy API | Railway |
| Deploy Web | Vercel |
| Dataset | ISIC 2019 |
