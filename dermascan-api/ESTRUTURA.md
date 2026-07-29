dermascan-api
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app principal
│   ├── routes/
│   │   ├── __init__.py
│   │   └── predict.py     # POST /predict
│   ├── services/
│   │   ├── __init__.py
│   │   └── inference.py   # Protocol InferenceService + MockInferenceService (depois: RealInferenceService)
│   └── schemas/
│       ├── __init__.py
│       └── prediction.py  # Modelos Pydantic (request/response)
├── tests/
│   └── test_predict.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
└── .env_example
