
import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "website_safety_model.joblib"
SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema = json.load(file)

# Handle either a list of feature names or a dictionary schema.
if isinstance(schema, list):
    feature_names = schema
elif "feature_names" in schema:
    feature_names = schema["feature_names"]
else:
    raise ValueError(
        "Could not find feature_names in feature_schema.json. "
        "Please inspect that file."
    )

# Random Forest exposes feature_importances_ directly.
importances = model.feature_importances_

if len(feature_names) != len(importances):
    raise ValueError(
        f"Feature mismatch: schema has {len(feature_names)} names, "
        f"but model has {len(importances)} importances."
    )

results = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

results = results.sort_values("importance", ascending=False)

print("\nFEATURE IMPORTANCE (highest to lowest)")
print(results.to_string(index=False))

print("\nTotal importance:", results["importance"].sum())
