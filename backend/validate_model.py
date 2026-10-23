
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit

# Make backend imports work when launched from the project root.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from features import FEATURE_NAMES, extract_features


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "url_features_extracted1.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

URL_COLUMN = "URL"
LABEL_COLUMN = "ClassLabel"
RANDOM_STATE = 42


def get_hostname(url):
    """Extract the hostname without visiting the website."""
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
    This does not fully handle every public suffix.
    """
    parts = hostname.split(".")

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
    """Basic normalization for duplicate removal."""
    value = str(url).strip().lower()
    value = value.removeprefix("https://")
    value = value.removeprefix("http://")
    return value.rstrip("/")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if URL_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError("Dataset must contain URL and ClassLabel columns.")

    # Remove missing URLs and labels.
    df = df.dropna(subset=[URL_COLUMN, LABEL_COLUMN]).copy()
    df[URL_COLUMN] = df[URL_COLUMN].astype(str).str.strip()
    df = df[df[URL_COLUMN] != ""]

    labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df[labels.isin([0, 1])].copy()
    df[LABEL_COLUMN] = labels.loc[df.index].astype(int)

    # Deduplicate before splitting.
    df["_normalized_url"] = df[URL_COLUMN].map(normalize_url)
    df = df.drop_duplicates("_normalized_url").copy()

    # Dataset: 0 = phishing, 1 = legitimate.
    # Experiment: 1 = phishing, 0 = legitimate.
    y = (df[LABEL_COLUMN] == 0).astype(int)

    df["_hostname"] = df[URL_COLUMN].map(get_hostname)
    df["_group"] = df["_hostname"].map(get_domain_group)

    valid = df["_hostname"].ne("") & df["_group"].ne("")
    df = df.loc[valid].copy()
    y = y.loc[df.index]

    print("Cleaned rows:", len(df))
    print("Phishing:", int(y.sum()))
    print("Legitimate:", int((y == 0).sum()))
    print("Approximate domain groups:", df["_group"].nunique())

    # Use the same feature extractor as the live application.
    print("\nExtracting URL features...")
    feature_rows = []
    valid_indices = []

    for idx, url in df[URL_COLUMN].items():
        try:
            feature_rows.append(extract_features(url))
            valid_indices.append(idx)
        except ValueError:
            pass

    df = df.loc[valid_indices].copy()
    y = y.loc[valid_indices]

    X = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES,
        index=valid_indices,
    )

    print("Usable URLs:", len(X))
    print("URLs skipped:", len(df) - len(X))  # Usable rows after extraction.

    # One shared split for a fair comparison.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=df["_group"])
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = set(df["_group"].iloc[train_idx])
    test_groups = set(df["_group"].iloc[test_idx])

    overlap = train_groups & test_groups

    print("\nTraining rows:", len(train_idx))
    print("Testing rows:", len(test_idx))
    print("Overlapping approximate domains:", len(overlap))

    if overlap:
        raise RuntimeError("Domain overlap detected; review the grouping logic.")

    # Baseline predicts only the majority class in training data.
    experiments = {
        "Majority-class baseline": (
            DummyClassifier(strategy="most_frequent"),
            list(X.columns),
        ),
        "Random Forest - all features": (
            RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            list(X.columns),
        ),
        "Random Forest - without HTTPS": (
            RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            [column for column in X.columns if column != "https_flag"],
        ),
    }

    results = []

    for name, (model, selected_features) in experiments.items():
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        model.fit(
            X_train[selected_features],
            y_train,
        )

        predictions = model.predict(
            X_test[selected_features]
        )

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
        }

        results.append(metrics)

        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.4f}")

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Legitimate", "Phishing"],
                zero_division=0,
            )
        )

    results_df = pd.DataFrame(results)

    print("\nFINAL COMPARISON")
    print(results_df.to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "validation_comparison.csv"
    results_df.to_csv(report_path, index=False)

    print("\nComparison saved to:", report_path)
    print("Experiment complete. Existing model was not changed.")


if __name__ == "__main__":
    main()
