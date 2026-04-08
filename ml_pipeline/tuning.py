from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform

# OPTIONAL: tuning script to add later
# repeated cross-validated fitting under a chosen metric

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000
    ))
])

param_dist = {
    "clf__C": loguniform(1e-3, 1e2),
    "clf__penalty": ["l1", "l2"]
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=20,
    scoring="average_precision",
    cv=5,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)