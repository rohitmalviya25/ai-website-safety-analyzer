
import sys
from pathlib import Path

import pandas as pd
import joblib

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from features import FEATURE_NAMES, extract_features

MODEL_PATH = BACKEND / "artifacts" / "website_safety_model.joblib"

# These were the two false negatives in your report.
urls = [
    "https://krakgenloges.gitbook.io/",
    "https://raw.githubusercontent.com/BlueSkyXN/ChangeSource/master/besttrace",
]

model = joblib.load(MODEL_PATH)

for url in urls:
    print("\n" + "=" * 65)
    print("URL:", url)

    features = extract_features(url)
    feature_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    prediction = int(model.predict(feature_df)[0])
    probabilities = model.predict_proba(feature_df)[0]

    print("\nExtracted features:")
    for name, value in features.items():
        print(f"{name}: {value}")

    print("\nModel prediction:",
          "PHISHING" if prediction == 1 else "LEGITIMATE")
    print("Estimated phishing score:",
          round(float(probabilities[1]) * 100, 2), "%")

print("\nInspection complete. No URLs were opened.")
