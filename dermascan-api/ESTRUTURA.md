dermascan-api
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app principal
│   ├── routes/
│   │   ├── __init__.py
│   │   └── predict.py     # POST /predict
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference.py   # Protocol InferenceService + Mock (fallback) + RealInferenceService (modelo treinado)
│   │   ├── model.py       # Arquitetura DermaScanModel — espelho de dermascan-model/src/model.py
│   │   └── risk.py        # Regra de negócio de risco — espelho de dermascan-model/src/risk.py
│   └── schemas/
│       ├── __init__.py
│       └── prediction.py  # Modelos Pydantic (request/response)
├── models/
│   └── dermascan_v1.pt    # Checkpoint treinado (gitignored — não vai pro git)
├── tests/
│   ├── test_predict.py
│   └── test_inference_real.py  # Regra de risco + smoke test do modelo real (skip sem checkpoint)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
└── .env_example
