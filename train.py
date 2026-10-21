
"""Train and evaluate a Credit Card Fraud Detection model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "CreditCard.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 2. Load data
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

df = df.dropna(subset=["IsFraud"]).copy()
df["IsFraud"] = pd.to_numeric(df["IsFraud"], errors="raise").astype(int)

if not df["IsFraud"].isin([0, 1]).all() or df["IsFraud"].nunique() != 2:
    raise ValueError("IsFraud must contain both 0 and 1.")

df["Transaction Date"] = pd.to_datetime(
    df["Transaction Date"], errors="coerce"
)
df["month"] = df["Transaction Date"].dt.month
df["day_of_week"] = df["Transaction Date"].dt.dayofweek

# 3. Features
CATEGORICAL = [
    "Merchant Name", "State", "Card Type", "Bank",
    "Transaction Category", "Merchant Location",
]
NUMERIC = ["Transaction Amount (INR)", "month", "day_of_week"]
FEATURES = CATEGORICAL + NUMERIC

X = df[FEATURES].copy()
y = df["IsFraud"]

for col in CATEGORICAL:
    X[col] = X[col].astype("string")

for col in NUMERIC:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Preprocessing and model
preprocess = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", __import__("sklearn").impute.SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL),
    ("num", Pipeline([
        ("imputer", __import__("sklearn").impute.SimpleImputer(
            strategy="median"
        )),
    ]), NUMERIC),
])

pipeline = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=1
    )),
])

params = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    pipeline, params, scoring="f1", cv=cv, n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# 6. Evaluate
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("\nBest parameters:", grid.best_params_)
print("Cross-validation F1:", round(grid.best_score_, 4))
print("\n", classification_report(
    y_test, pred, target_names=["Legitimate", "Fraud"], zero_division=0
))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

# 7. Baseline comparisons
baselines = {}
for name, constant in [("always_legitimate", 0), ("always_fraud", 1)]:
    dummy = DummyClassifier(strategy="constant", constant=constant)
    dummy.fit(X_train, y_train)
    p = dummy.predict(X_test)
    baselines[name] = {
        "accuracy": float(accuracy_score(y_test, p)),
        "precision": float(precision_score(y_test, p, zero_division=0)),
        "recall": float(recall_score(y_test, p, zero_division=0)),
        "f1": float(f1_score(y_test, p, zero_division=0)),
    }

# 8. Save model and metadata
joblib.dump(model, MODEL_DIR / "fraud_pipeline.joblib")

metadata = {
    "sklearn_version": sklearn.__version__,
    "features": FEATURES,
    "categorical_options": {
        c: sorted(df[c].dropna().astype(str).unique().tolist())
        for c in CATEGORICAL
    },
    "amount_range": [
        float(df[NUMERIC[0]].min()),
        float(df[NUMERIC[0]].max()),
    ],
}

with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

metrics = {
    "best_params": grid.best_params_,
    "cv_f1": float(grid.best_score_),
    "test": {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_fraud": float(precision_score(y_test, pred, zero_division=0)),
        "recall_fraud": float(recall_score(y_test, pred, zero_division=0)),
        "f1_fraud": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    },
    "baselines": baselines,
}

with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model, metadata and metrics to:", MODEL_DIR)
