import os
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

def clean_shipments(raw, medians=None):
    df = raw.drop_duplicates().copy()
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    if "declared_value" in df.columns:
        df["declared_value_inr"] = df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip()
        df["declared_value_inr"] = pd.to_numeric(df["declared_value_inr"], errors="coerce")
        df = df.drop(columns=["declared_value"])
        
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    if medians is None:
        medians = {
            "distance_km": df["distance_km"].median(),
            "parcel_weight_kg": df["parcel_weight_kg"].median(),
            "declared_value_inr": df["declared_value_inr"].median(),
        }
        
    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
            
    return df, medians

def build_features(clean, hubs):
    df = clean.merge(hubs, on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df[FEATURES]

def run_part_a():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Loading raw shipments and hubs...")
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    initial_raw_len = len(raw)
    
    print("Cleaning shipments (deduplication, imputation, text standardization)...")
    clean, medians = clean_shipments(raw)
    clean_len = len(clean)
    
    clean['promised_hours'] = clean['service_type'].map(PROMISED_HOURS)
    clean['is_delayed'] = (clean['delivery_hours'] > clean['promised_hours']).astype(int)
    
    print("Building engineered features and merging hub attributes...")
    X_features = build_features(clean, hubs)
    
    modelling_table = clean[['shipment_id', 'hub_id', 'service_type', 'delivery_hours', 'is_delayed']].copy()
    for col in FEATURES:
        modelling_table[col] = X_features[col]
        
    modelling_table.to_csv("data/modelling_table.csv", index=False)
    print(f"Saved modeling table with shape {modelling_table.shape} to data/modelling_table.csv")
    
    print("Generating EDA checks and performance reports...")
    
    # Audit checks
    checks = pd.DataFrame({
        "metric": ["initial_raw_rows", "deduplicated_clean_rows", "duplicates_dropped", "final_modelling_rows"],
        "value": [initial_raw_len, clean_len, initial_raw_len - clean_len, len(modelling_table)]
    })
    checks.to_csv("outputs/part_a_checks.csv", index=False)
    
    # Delay rate by service group
    delay_group = modelling_table.groupby('service_type').agg(
        total_shipments=('is_delayed', 'count'),
        delayed_count=('is_delayed', 'sum'),
        delay_rate=('is_delayed', 'mean')
    ).reset_index()
    delay_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    # Correlation with delivery hours
    numeric_cols = modelling_table.select_dtypes(include=[np.number]).columns
    corr = modelling_table[numeric_cols].corr()['delivery_hours'].reset_index()
    corr.columns = ['feature', 'correlation_with_delivery_hours']
    corr.to_csv("outputs/part_a_correlations.csv", index=False)
    
    print("Part A completed successfully! All check files saved to outputs/.")

if __name__ == "__main__":
    run_part_a()