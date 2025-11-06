# Risk-and-Severity-prediction-model
Fever Severity &amp; Risk Prediction API — An AI-powered health prediction system that analyzes patient data (age, temperature, heart rate, oxygen level, etc.) to classify fever severity (Mild/Moderate/Severe) and risk level (Low/High) using Machine Learning and FastAPI.
---

# Fever Severity & Risk Prediction API

An **AI-powered health prediction system** that analyzes patient health data to classify **fever severity** (Mild / Moderate / Severe) and **risk level** (Low / High).
Built using **Python, FastAPI, scikit-learn, and Streamlit**, this project demonstrates how ML models can support early health risk assessment.

---

## Features

*  **ML Models** (Random Forest) trained on synthetic patient data
*  **REST API** built with FastAPI for real-time predictions
*  **Streamlit Dashboard** for interactive input & visualization
*  **Clean Validation** using Pydantic models for safe input handling

---

## Tech Stack

* **Python 3.9+**
* **FastAPI** – backend API framework
* **scikit-learn** – machine learning models
* **Uvicorn** – ASGI server for FastAPI
* **Streamlit** – frontend dashboard
* **Pandas, NumPy, Joblib** – data handling & model persistence

---

## Project Structure

```
ML_Model/
│
├── datasets/                # Raw or processed data
├── models/                  # Trained ML models (.joblib)
│
├── train_model.py           # Script to train models
├── predict.py               # FastAPI backend for predictions
├── streamlit_app.py         # Streamlit dashboard UI
│
└── requirements.txt         # Dependencies
```

---

##  Installation & Setup

### 1️ Clone this repository

```bash
git clone https://github.com/<your-username>/fever-risk-api.git
cd fever-risk-api
```

### 2️ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # for Windows
# or
source venv/bin/activate   # for Mac/Linux
```

### 3️ Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the ML Models

Run this command to generate and train models:

```bash
python train_model.py
```

This will create:

* `rf_risk.joblib` → predicts **risk level**
* `rf_severity.joblib` → predicts **fever severity**
* `scaler.joblib` → feature normalizer

All saved inside the `models/` folder.

---

## Run the FastAPI Server

Start the backend API:

```bash
uvicorn predict:app --reload
```

Then open your browser at 👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

You’ll see the interactive Swagger UI where you can test predictions with JSON input.

**Example Input:**

```json
{
  "age": 45,
  "temperature": 38.5,
  "heart_rate": 95,
  "bp_sys": 130,
  "bp_dia": 85,
  "oxygen_sat": 96,
  "symptoms_count": 3,
  "chronic_conditions": 1,
  "region_risk_index": 0.4,
  "days_since_onset": 4
}
```

**Example Output:**

```json
{
  "risk": 0,
  "risk_label": "Low",
  "severity": 1,
  "severity_label": "Moderate"
}
```

---

## Future Enhancements

* Integrate **real anonymized medical data**
* Add **model explainability** (SHAP / feature importance)
* Deploy on **Render / Hugging Face Spaces / AWS**
* Build a **mobile frontend** or integrate with a health chatbot

---

## Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to modify.

---

Would you like me to include a **diagram (architecture image)** for your GitHub readme too — showing how the system flows from input → model → API → dashboard?
