import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ======================================================
# 1. Load or create dataset
# ======================================================
data_path = "datasets/processed/final_dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path}, shape = {df.shape}")
else:
    print("No dataset found — generating synthetic data for demo...")
    np.random.seed(42)
    N = 2000
    df = pd.DataFrame({
        "age": np.random.randint(0, 100, N),
        "temperature": np.random.normal(37, 1.5, N).round(1),
        "heart_rate": np.random.randint(60, 120, N),
        "bp_sys": np.random.randint(90, 180, N),
        "bp_dia": np.random.randint(60, 110, N),
        "oxygen_sat": np.random.randint(85, 100, N),
        "symptoms_count": np.random.randint(0, 8, N),
        "chronic_conditions": np.random.randint(0, 4, N),
        "region_risk_index": np.random.rand(N).round(2),
        "days_since_onset": np.random.randint(0, 14, N)
    })

    df["risk"] = ((df["age"] > 65) | (df["oxygen_sat"] < 92) | (df["chronic_conditions"] >= 2)).astype(int)
    df["severity"] = pd.qcut(
        ((df["temperature"] - 37).clip(0) + (100 - df["oxygen_sat"]) * 0.2 + df["chronic_conditions"] * 0.5),
        3,
        labels=[0, 1, 2]
    ).astype(int)

# ======================================================
# 2. Prepare data
# ======================================================
features = ["age","temperature","heart_rate","bp_sys","bp_dia","oxygen_sat",
            "symptoms_count","chronic_conditions","region_risk_index","days_since_onset"]

X = df[features]
y_risk = df["risk"]
y_severity = df["severity"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_risk_train, y_risk_test = train_test_split(X_scaled, y_risk, test_size=0.2, random_state=42)
_, _, y_sev_train, y_sev_test = train_test_split(X_scaled, y_severity, test_size=0.2, random_state=42)

# ======================================================
# 3. Train Models
# ======================================================
rf_risk = RandomForestClassifier(n_estimators=200, random_state=42)
rf_risk.fit(X_train, y_risk_train)

rf_sev = RandomForestClassifier(n_estimators=300, random_state=42)
rf_sev.fit(X_train, y_sev_train)

# ======================================================
# 4. Evaluate
# ======================================================
print("\n=== Risk Classification ===")
y_pred_risk = rf_risk.predict(X_test)
print("Accuracy:", accuracy_score(y_risk_test, y_pred_risk))
print(classification_report(y_risk_test, y_pred_risk))

print("\n=== Severity Classification ===")
y_pred_sev = rf_sev.predict(X_test)
print("Accuracy:", accuracy_score(y_sev_test, y_pred_sev))
print(classification_report(y_sev_test, y_pred_sev))

# ======================================================
# 5. Save Models
# ======================================================
os.makedirs("models", exist_ok=True)
joblib.dump(rf_risk, "models/rf_risk.joblib")
joblib.dump(rf_sev, "models/rf_severity.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\n Models saved to 'models/' folder.")