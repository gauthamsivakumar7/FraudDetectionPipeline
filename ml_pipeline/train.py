import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.preprocessing import build_preprocessor
from ml_pipeline.evaluate import compute_classification_metrics
from ml_pipeline.thresholding import find_threshold_for_min_precision

# split correctly
# preprocess consistently
# compare baseline and stronger model
# evaluate with multiple metrics
# save artifacts

def train_models(csv_path, target_col, artifacts_dir="artifacts"):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )
    }

    all_results = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("clf", estimator)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        y_prob = pipe.predict_proba(X_val)[:, 1]

        metrics = compute_classification_metrics(y_val, y_pred, y_prob)
        threshold_choice = find_threshold_for_min_precision(y_val, y_prob, min_precision=0.90)

        all_results[name] = {
            "metrics": metrics,
            "threshold_choice": threshold_choice
        }

        joblib.dump(pipe, f"{artifacts_dir}/{name}_pipeline.joblib")

    with open(f"{artifacts_dir}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))