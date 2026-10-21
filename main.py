"""FastAPI app. Run from the project root:  uvicorn app.main:app --reload"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
if metadata["sklearn_version"] != sklearn.__version__:
    warnings.warn(
        f"Model trained with scikit-learn {metadata['sklearn_version']} but "
        f"running {sklearn.__version__}. Re-run train.py to avoid problems."
    )
model = joblib.load(MODEL_DIR / "fraud_pipeline.joblib")

app = FastAPI(title="Credit Card Fraud Risk API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Transaction(BaseModel):
    merchant_name: str
    state: str
    card_type: str
    bank: str
    transaction_category: str
    merchant_location: str
    transaction_amount: float = Field(gt=0, description="Amount in INR")
    transaction_date: str = Field(description="YYYY-MM-DD")


@app.get("/options")
def options():
    """Dropdown values the frontend should offer."""
    return {"categories": metadata["categorical_options"],
            "amount_range": metadata["amount_range"]}


@app.post("/predict")
def predict(t: Transaction):
    try:
        date = pd.to_datetime(t.transaction_date)
    except Exception:
        raise HTTPException(status_code=422, detail="transaction_date must be YYYY-MM-DD")
    row = pd.DataFrame([{
        "Merchant Name": t.merchant_name,
        "State": t.state,
        "Card Type": t.card_type,
        "Bank": t.bank,
        "Transaction Category": t.transaction_category,
        "Merchant Location": t.merchant_location,
        "Transaction Amount (INR)": t.transaction_amount,
        "month": date.month,
        "day_of_week": date.dayofweek,
    }])[metadata["features"]]
    prob = float(model.predict_proba(row)[0, 1])
    return {"fraud_probability": round(prob, 4),
            "prediction": "Fraud" if prob >= 0.5 else "Legitimate"}


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")