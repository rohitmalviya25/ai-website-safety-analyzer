
from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import extract_features, FEATURE_NAMES
from virustotal_check import check_virustotal


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "website_safety_model.joblib"
SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"

app = FastAPI(
    title="AI Website Safety Analyzer",
    description="ML-based URL phishing risk assessment API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ai-website-safety-analyzer.vercel.app",
        "https://ai-website-safety-analyzer-hut1v534e.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_error = None


@app.on_event("startup")
def load_model():
    global model, model_error

    try:
        if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
            raise FileNotFoundError("Model artifacts are missing.")

        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            schema = json.load(file)

        if schema.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Model feature schema mismatch.")

        model = joblib.load(MODEL_PATH)
        model_error = None

    except Exception as error:
        model = None
        model_error = str(error)


class ScanRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
    }


@app.post("/api/scan")
def scan_url(request: ScanRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The ML model is unavailable. Check model artifacts.",
        )

    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    try:
        features = extract_features(url)

        X = pd.DataFrame(
            [[features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )

        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        phishing_index = list(model.classes_).index(1)
        phishing_score = float(probabilities[phishing_index]) * 100
        safety_score = 100 - phishing_score

        # Reputation lookup is independent of the ML prediction.
        # A failed lookup must not prevent the ML result from returning.
        try:
            reputation = check_virustotal(url)
        except Exception:
            reputation = {
                "status": "unavailable",
                "message": "VirusTotal reputation check failed.",
            }

        return {
            "url": url,
            "prediction": (
                "likely_phishing"
                if prediction == 1
                else "likely_legitimate"
            ),
            "phishing_score": round(phishing_score, 2),
            "safety_score": round(safety_score, 2),
            "score_type": "uncalibrated_model_score",
            "features": features,
            "virustotal": reputation,
            "disclaimer": (
                "The ML score is an estimate, not a guarantee of safety. "
                "VirusTotal results reflect available security-engine reports."
            ),
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
