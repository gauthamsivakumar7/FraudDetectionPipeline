from pathlib import Path
import json
import joblib
import pandas as pd
import logging
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.preprocessing import build_preprocessor
from ml_pipeline.evaluate import compute_classification_metrics
from ml_pipeline.thresholding import (
    find_threshold_for_min_precision,
    find_threshold_for_min_recall,
)

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Fraud (merged train+test from prepare_fraud.py): use labeled rows only via dropna below.
DATA_PATH = "data/processed/fraud_merged.csv"
TARGET_COL = "isFraud"
# Adult smoke test: DATA_PATH = "data/processed/adult.csv", TARGET_COL = "target"

# 1. load the dataset
# 2. define feature columns and target
# 3. call the training code
# 4. call evaluation/calibration/threshold selection
# 5. save the trained pipeline and results

def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("fraud_pipeline")

    t0 = time.perf_counter()
    log.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    log.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    n_all = len(df)
    df = df.dropna(subset=[TARGET_COL])
    log.info(
        "Using %d labeled rows for train/val (dropped %d without %s)",
        len(df),
        n_all - len(df),
        TARGET_COL,
    )

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    null_threshold = 0.6
    high_null_cols = [c for c in X.columns if X[c].isna().mean() > null_threshold]
    log.info("Dropping %d columns with >%.0f%% nulls", len(high_null_cols), null_threshold * 100)
    X = X.drop(columns=high_null_cols)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    log.info(
        "Detected %d numeric features and %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )

    log.info("Splitting train/val (test_size=0.2, stratify=y)")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    log.info("Split sizes: train=%d, val=%d", X_train.shape[0], X_val.shape[0])

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    models = {
        "logreg": LogisticRegression(
            l1_ratio=1.0,
            solver="saga",
            max_iter=3000,
            class_weight="balanced",
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results = {}

    for model_name, estimator in models.items():
        log.info("==== Training model: %s ====", model_name)
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("clf", estimator),
        ])

        t_fit0 = time.perf_counter()
        log.info("Fitting pipeline...")
        pipe.fit(X_train, y_train)
        log.info("Fit complete in %.2fs", time.perf_counter() - t_fit0)

        t_pred0 = time.perf_counter()
        log.info("Predicting on validation set...")
        y_pred = pipe.predict(X_val)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        log.info("Prediction complete in %.2fs", time.perf_counter() - t_pred0)

        log.info("Computing metrics + thresholds...")
        metrics = compute_classification_metrics(y_val, y_pred, y_prob)

        threshold_precision = find_threshold_for_min_precision(
            y_val, y_prob, min_precision=0.90
        )
        threshold_recall = find_threshold_for_min_recall(
            y_val, y_prob, min_recall=0.95
        )

        results[model_name] = {
            "metrics": metrics,
            "threshold_for_precision_90": threshold_precision,
            "threshold_for_recall_95": threshold_recall,
        }

        log.info("Saving trained pipeline to artifacts/%s_pipeline.joblib", model_name)
        joblib.dump(pipe, ARTIFACT_DIR / f"{model_name}_pipeline.joblib")
        log.info("Saved.")

    log.info("Writing results to artifacts/results.json")
    with open(ARTIFACT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("Done in %.2fs total", time.perf_counter() - t0)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()