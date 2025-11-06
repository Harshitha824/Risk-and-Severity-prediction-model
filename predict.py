# predict.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Fever Risk & Severity API")

# Pydantic model with example values so Swagger shows a clear example
class PatientFeatures(BaseModel):
    age: int = Field(..., example=45)
    temperature: float = Field(..., example=38.5)
    heart_rate: int = Field(..., example=95)
    bp_sys: int = Field(..., example=130)
    bp_dia: int = Field(..., example=85)
    oxygen_sat: int = Field(..., example=96)
    symptoms_count: int = Field(..., example=3)
    chronic_conditions: int = Field(..., example=1)
    region_risk_index: float = Field(..., example=0.4)
    days_since_onset: int = Field(..., example=4)

# Load models (adjust paths if needed)
RF_RISK_PATH = "models/rf_risk.joblib"
RF_SEV_PATH = "models/rf_severity.joblib"
SCALER_PATH = "models/scaler.joblib"

# Lazy load with simple fallback messages if files are missing
def load_models():
    try:
        rf_risk = joblib.load(RF_RISK_PATH)
        rf_sev = joblib.load(RF_SEV_PATH)
        scaler = joblib.load(SCALER_PATH)
        return rf_risk, rf_sev, scaler
    except Exception as e:
        # Provide a clear error so /docs still loads
        print("Model loading error:", e)
        return None, None, None

rf_risk, rf_sev, scaler = load_models()

@app.get("/")
def home():
    return {"message": "Fever Risk & Severity Prediction API is running. Use /docs to test."}

@app.post("/predict")
def predict(payload: PatientFeatures):
    """Predict risk and severity. Expects a JSON body matching PatientFeatures."""
    if rf_risk is None or rf_sev is None or scaler is None:
        return {"error": "Models not loaded. Ensure models/*.joblib exist."}

    # Convert to DataFrame with correct column order
    df = pd.DataFrame([payload.dict()])
    # Note: scaler expects the same feature order used during training
    X_scaled = scaler.transform(df)
    risk_pred = int(rf_risk.predict(X_scaled)[0])
    sev_pred = int(rf_sev.predict(X_scaled)[0])

    return {
        "risk": risk_pred,
        "risk_label": "High" if risk_pred == 1 else "Low",
        "severity": sev_pred,
        "severity_label": ["Mild", "Moderate", "Severe"][sev_pred]
    }

@app.get("/predict/test")
def predict_test():
    """Return a sample prediction using example values (useful for quick checks)."""
    sample = PatientFeatures(
        age=45, temperature=38.5, heart_rate=95, bp_sys=130, bp_dia=85,
        oxygen_sat=96, symptoms_count=3, chronic_conditions=1,
        region_risk_index=0.4, days_since_onset=4
    )
    return predict(sample)

# Optional: allow quick CLI test when running python predict.py
if __name__ == "__main__":
    # quick local CLI test (no server)
    rf_risk, rf_sev, scaler = load_models()
    sample = PatientFeatures(
        age=45, temperature=38.5, heart_rate=95, bp_sys=130, bp_dia=85,
        oxygen_sat=96, symptoms_count=3, chronic_conditions=1,
        region_risk_index=0.4, days_since_onset=4
    )
    print("Sample input:", sample.json())
    if rf_risk is not None and rf_sev is not None and scaler is not None:
        import pandas as pd
        df = pd.DataFrame([sample.dict()])
        X_scaled = scaler.transform(df)
        print("Risk:", int(rf_risk.predict(X_scaled)[0]))
        print("Severity:", int(rf_sev.predict(X_scaled)[0]))
    else:
        print("Models not loaded; place rf_risk.joblib, rf_severity.joblib, scaler.joblib into models/")