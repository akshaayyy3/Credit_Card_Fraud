from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# Find project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
model = joblib.load(
    BASE_DIR / "fraud_detection_model.pkl"
)


# Create FastAPI application
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0"
)


# Input data structure
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    # Convert input to dictionary
    data = transaction.model_dump()

    # Convert dictionary to DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get fraud probability
    probability = model.predict_proba(input_data)[0][1]

    # Convert prediction to readable result
    result = "Fraud" if prediction == 1 else "Legitimate"

    return {
        "prediction": result,
        "fraud_probability": round(float(probability), 4)
    }