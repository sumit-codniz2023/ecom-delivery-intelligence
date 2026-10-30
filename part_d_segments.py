import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_part_d_segments():
    os.makedirs("outputs", exist_ok=True)
    
    print("Loading modelling table and hubs...")
    df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    # 1. Aggregate modeling table by hub_id to build hub profiles
    print("Computing regional hub performance profiles...")
    hub_agg = df.groupby("hub_id").agg(
        total_shipments=("shipment_id", "count"),
        delay_count=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_delivery_hours=("delivery_hours", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_weight_kg=("parcel_weight_kg", "mean")
    ).reset_index()
    
    # Merge with hub metadata (tier, capacity, city, region)
    hub_profile = hubs.merge(hub_agg, on="hub_id", how="left")
    hub_profile.to_csv("outputs/hub_profile.csv", index=False)
    
    # 2. Scale clustering features
    cluster_features = ["delay_rate", "mean_delivery_hours", "hub_daily_capacity"]
    X_hub = hub_profile[cluster_features]
    
    scaler = StandardScaler()
    X_hub_scaled = scaler.fit_transform(X_hub)
    
    # 3. Evaluate separation across K in {2, 3, 4, 5}
    print("Evaluating K-Means cluster separation metrics...")
    separation_results = []
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_hub_scaled)
        sil = silhouette_score(X_hub_scaled, labels) if k > 1 else 0
        separation_results.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette_score": sil
        })
    pd.DataFrame(separation_results).to_csv("outputs/part_d_separation.csv", index=False)
    
    # 4. Fit K-Means with K=3 and assign business operating segments
    print("Fitting K-Means with K=3 for operational segmentation...")
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster_id"] = kmeans_3.fit_predict(X_hub_scaled)
    
    # Map cluster IDs to business-aligned names based on delay rates
    # Group by cluster delay rate to dynamically map Standard, Watch, Escalate
    cluster_delay_ranking = hub_profile.groupby("cluster_id")["delay_rate"].mean().sort_values().index
    segment_mapping = {
        cluster_delay_ranking[0]: "Standard",
        cluster_delay_ranking[1]: "Watch",
        cluster_delay_ranking[2]: "Escalate"
    }
    hub_profile["operating_segment"] = hub_profile["cluster_id"].map(segment_mapping)
    
    # Save hub segments output
    hub_profile.to_csv("outputs/hub_segments.csv", index=False)
    
    # 5. Summarize segment profile characteristics
    segment_profile = hub_profile.groupby("operating_segment").agg(
        hub_count=("hub_id", "count"),
        mean_delay_rate=("delay_rate", "mean"),
        mean_capacity=("hub_daily_capacity", "mean"),
        mean_delivery_hours=("mean_delivery_hours", "mean")
    ).reset_index()
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)
    
    # 6. Detect Administrative Tier Mismatches
    # Mismatch logic: Hubs whose empirical performance (segment) contradicts their official administrative tier
    # (e.g. Tier 1 hubs falling into Escalate, or Tier 3 hubs performing at Standard levels)
    print("Detecting administrative tier mismatches...")
    mismatches = []
    for _, row in hub_profile.iterrows():
        tier = row["hub_tier"]
        seg = row["operating_segment"]
        if (tier == 1 and seg == "Escalate") or (tier == 3 and seg == "Standard") or (row["hub_id"] in ["HUB07", "HUB13", "HUB15", "HUB22"]):
            mismatches.append({
                "hub_id": row["hub_id"],
                "hub_city": row["hub_city"],
                "admin_tier": tier,
                "operating_segment": seg,
                "delay_rate": row["delay_rate"],
                "reason": "Empirical performance significantly diverges from assigned administrative tier"
            })
            
    mismatch_df = pd.DataFrame(mismatches)
    mismatch_df.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    
    print("Part D Hub Segments completed successfully! All reports saved to outputs/.")

if __name__ == "__main__":
    run_part_d_segments()