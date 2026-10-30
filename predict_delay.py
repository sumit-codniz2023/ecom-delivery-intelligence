import os
import joblib
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

def run_prediction():
    os.makedirs("outputs", exist_ok=True)
    
    print("Loading serialized model package...")
    package = joblib.load("models/delay_model.pkl")
    model = package["model"]
    scaler = package["scaler"]
    threshold = package["threshold"]
    features = package["features"]
    medians = package["medians"]
    
    print("Loading new shipments and hub metadata...")
    raw_new = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    # Load hub segments if available (fallback to Standard if Part D hasn't run yet)
    if os.path.exists("outputs/hub_segments.csv"):
        hub_segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "operating_segment"]]
    else:
        hub_segments = hubs[["hub_id"]].copy()
        hub_segments["operating_segment"] = "Standard"
        
    print("Cleaning new shipments using stored training medians...")
    df = raw_new.copy()
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    if "declared_value" in df.columns:
        df["declared_value_inr"] = df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip()
        df["declared_value_inr"] = pd.to_numeric(df["declared_value_inr"], errors="coerce")
        df = df.drop(columns=["declared_value"])
        
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # Impute missing numerical fields using stored training medians
    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
            
    # Merge hub attributes
    df = df.merge(hubs, on="hub_id", how="left")
    
    # Build exact features
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    X_new = df[features]
    X_new_scaled = scaler.transform(X_new)
    
    print("Running model inference and threshold evaluation...")
    probs = model.predict_proba(X_new_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    df["delay_probability"] = probs.round(3)
    df["predicted_delay"] = preds
    
    # Merge hub operating segments
    df = df.merge(hub_segments, on="hub_id", how="left")
    
    # Assign operational action items based on risk and hub segment
    def assign_action(row):
        if row["predicted_delay"] == 1 and row["operating_segment"] == "Escalate":
            return "Urgent: Proactive Customer Call & Hub Dispatch Priority"
        elif row["predicted_delay"] == 1:
            return "Monitor: Flagged for Potential Delay"
        else:
            return "Standard Dispatch"
            
    df["operational_action"] = df.apply(assign_action, axis=1)
    
    # Select final columns for operational export
    output_cols = [
        "shipment_id", "hub_id", "service_type", "operating_segment", 
        "delay_probability", "predicted_delay", "operational_action"
    ]
    actions_df = df[output_cols]
    
    actions_output_path = "outputs/new_shipment_actions.csv"
    actions_df.to_csv(actions_output_path, index=False)
    print(f"Batch inference complete! Generated action list for {len(actions_df)} shipments saved to {actions_output_path}")

if __name__ == "__main__":
    run_prediction()