
import json
import joblib
import pandas as pd
from pathlib import Path

from features import extract_features, FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "website_safety_model.joblib"
SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"

if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
    raise FileNotFoundError(
        "Model or feature schema not found. Run train_models.py first."
    )

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema = json.load(file)

if schema["feature_names"] != FEATURE_NAMES:
    raise ValueError("Feature schema does not match features.py")

# These are illustrative test inputs, not verified safe websites.
test_urls = [
    "https://example.com/login",
    "http://192.168.1.1/account",
    "https://example.org/products?id=123",
]

for url in test_urls:
    features = extract_features(url)

    X = pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    # Training convention: 1 = phishing, 0 = legitimate.
    phishing_index = list(model.classes_).index(1)
    phishing_score = float(probabilities[phishing_index]) * 100

    print("\nURL:", url)
    print(
        "Prediction:",
        "Likely phishing" if prediction == 1 else "Likely legitimate"
    )
    print(f"Estimated phishing score: {phishing_score:.2f}/100")
    print("Note: This is a model estimate, not a safety guarantee.")
