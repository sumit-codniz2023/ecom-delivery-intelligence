import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

def run_packaging():
    os.makedirs("models", exist_ok=True)
    
    print("Loading modelling table...")
    df = pd.read_csv("data/modelling_table.csv")
    
    # 1. Compute imputation medians from the training distribution
    print("Calculating training imputation medians...")
    medians = {
        "distance_km": df["distance_km"].median(),
        "parcel_weight_kg": df["parcel_weight_kg"].median(),
        "log_declared_value": df["log_declared_value"].median(),
    }
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 2. Replicate split to identify optimal threshold on validation set
    print("Performing stratified split to lock in optimal threshold...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Train final Random Forest classifier
    print("Training final Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 4. Determine optimal threshold using validation recall criterion (>= 0.80)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    thresholds = np.arange(0.1, 0.95, 0.05)
    optimal_threshold = 0.50
    
    for t in thresholds:
        preds_t = (val_probs >= t).astype(int)
        rec = recall_score(y_val, preds_t, zero_division=0)
        if rec >= 0.80:
            optimal_threshold = round(t, 2)
            
    print(f"Locked Operational Threshold: {optimal_threshold}")
    
    # 5. Package artifacts into a single dictionary container
    package = {
        "model": model,
        "scaler": scaler,
        "threshold": optimal_threshold,
        "features": FEATURES,
        "medians": medians
    }
    
    model_filename = "models/delay_model.pkl"
    joblib.dump(package, model_filename)
    print(f"Successfully serialized model package to {model_filename}")

if __name__ == "__main__":
    run_packaging()