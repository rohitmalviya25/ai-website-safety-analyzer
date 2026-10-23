
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make the backend folder importable.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from features import FEATURE_NAMES, extract_features


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "url_features_extracted1.csv"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

URL_COLUMN = "URL"
LABEL_COLUMN = "ClassLabel"
RANDOM_STATE = 42


def get_hostname(url):
    """Extract a hostname without visiting the URL."""
    if not isinstance(url, str):
        return ""

    value = url.strip()

    if "://" not in value:
        value = "http://" + value

    try:
        return (urlparse(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def get_domain_group(hostname):
    """
    Approximate registrable domain for grouping related URLs.
    A proper Public Suffix List is needed for full accuracy.
    """
    if not hostname:
        return ""

    parts = hostname.split(".")

    # Handle common multi-label public suffixes.
    two_level_suffixes = {
        "co.uk", "org.uk", "ac.uk",
        "com.au", "net.au", "org.au",
        "co.in", "firm.in", "net.in",
        "co.jp", "co.nz", "com.br",
    }

    if len(parts) >= 3 and ".".join(parts[-2:]) in two_level_suffixes:
        return ".".join(parts[-3:])

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return hostname


def normalize_url(url):
    """Conservative normalization for duplicate removal."""
    if not isinstance(url, str):
        return ""

    value = url.strip().lower()
    value = re.sub(r"^https?://", "", value)
    return value.rstrip("/")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {URL_COLUMN, LABEL_COLUMN}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required_columns - set(df.columns)}"
        )

    print("Original dataset shape:", df.shape)

    # Keep only usable URLs and known binary labels.
    df = df.dropna(subset=[URL_COLUMN, LABEL_COLUMN]).copy()
    df[URL_COLUMN] = df[URL_COLUMN].astype(str).str.strip()
    df = df[df[URL_COLUMN] != ""]

    numeric_labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df[numeric_labels.isin([0, 1])].copy()
    df[LABEL_COLUMN] = numeric_labels.loc[df.index].astype(int)

    # Normalize and remove duplicate URLs before splitting.
    df["normalized_url"] = df[URL_COLUMN].map(normalize_url)
    df = df.drop_duplicates(subset=["normalized_url"]).copy()

    # Dataset convention: 0 = phishing, 1 = legitimate.
    # Our ML target: 1 = phishing, 0 = legitimate.
    y = (df[LABEL_COLUMN] == 0).astype(int)

    # Group related hostnames together to reduce leakage.
    df["hostname"] = df[URL_COLUMN].map(get_hostname)
    df["domain_group"] = df["hostname"].map(get_domain_group)

    valid = (
        df["hostname"].ne("")
        & df["domain_group"].ne("")
    )
    df = df.loc[valid].copy()
    y = y.loc[df.index]

    print("Rows after cleaning:", len(df))
    print("Phishing examples:", int(y.sum()))
    print("Legitimate examples:", int((y == 0).sum()))
    print("Unique domain groups:", df["domain_group"].nunique())

    # IMPORTANT: the same extractor is used for training and inference.
    
    print("\nExtracting URL features...")

    feature_rows = []
    valid_indices = []
    failed_urls = []

    for idx, url in df[URL_COLUMN].items():
        try:
            features = extract_features(url)
            feature_rows.append(features)
            valid_indices.append(idx)

        except ValueError as error:
            failed_urls.append((url, str(re.error)))

# Keep the feature rows and labels aligned.
    df = df.loc[valid_indices].copy()
    y = y.loc[valid_indices]

    X = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES,
        index=valid_indices,
)

    print("Successfully extracted features:", len(X))
    print("URLs skipped:", len(failed_urls))

    if failed_urls:
        print("\nFirst 10 URLs that could not be processed:")

        for url, reason in failed_urls[:10]:
            print("URL:", repr(url))
            print("Reason:", reason)
            print("-" * 40)

    if len(X) == 0:
        raise ValueError("No valid URLs remain after feature extraction.")


    # GroupShuffleSplit keeps a domain group on one side of the split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=df["domain_group"])
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = set(df["domain_group"].iloc[train_idx])
    test_groups = set(df["domain_group"].iloc[test_idx])

    assert train_groups.isdisjoint(test_groups)

    print("\nTraining rows:", len(train_idx))
    print("Testing rows:", len(test_idx))
    print("Overlapping domain groups:", len(train_groups & test_groups))

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    print("\nTraining and evaluating models...")

    for name, model in models.items():
        print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "average_precision": average_precision_score(
                y_test, probabilities
            ),
        }

        results.append(metrics)
        trained_models[name] = model

        for metric, value in metrics.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

        print("\nConfusion matrix [actual rows, predicted columns]:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0,
        ))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["recall", "f1"],
        ascending=False,
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    # Select using recall first, then F1.
    # This is a preliminary selection rule, not a final security guarantee.
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACT_DIR / "website_safety_model.joblib"
    schema_path = ARTIFACT_DIR / "feature_schema.json"
    metrics_path = REPORT_DIR / "model_comparison.csv"

    joblib.dump(best_model, model_path)

    schema = {
        "feature_names": FEATURE_NAMES,
        "target_positive_class": 1,
        "target_positive_class_meaning": "phishing",
        "selected_model": best_name,
        "dataset": "LegitPhish Version 2",
        "split_method": "GroupShuffleSplit by approximate domain",
        "random_state": RANDOM_STATE,
    }

    schema_path.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    results_df.to_csv(metrics_path, index=False)

    print("\nSelected model:", best_name)
    print("Saved model:", model_path)
    print("Saved feature schema:", schema_path)
    print("Saved metrics:", metrics_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
