import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

def run_part_b():
    os.makedirs("outputs", exist_ok=True)
    
    print("Loading modelling table...")
    df = pd.read_csv("data/modelling_table.csv")
    
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 1. Two-Step Split: 70% Train (840), 15% Validation (180), 15% Test (180)
    print("Performing strict two-step data split...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"Split sizes -> Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # 2. Leakage-Safe Scaling (Fitted STRICTLY on Training Set Only)
    print("Fitting StandardScaler on training data only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaling verification checks
    scaling_checks = pd.DataFrame({
        "feature": FEATURES,
        "train_mean": X_train_scaled.mean(axis=0),
        "train_std": X_train_scaled.std(axis=0)
    })
    scaling_checks.to_csv("outputs/part_b_checks.csv", index=False)
    
    # 3. Baseline Evaluation (Predicting mean delivery hours on validation set)
    val_baseline_pred = np.full_like(y_val, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_val, val_baseline_pred))
    baseline_mae = mean_absolute_error(y_val, val_baseline_pred)
    print(f"Baseline Validation RMSE: {baseline_rmse:.2f}, MAE: {baseline_mae:.2f}")
    
    # 4. Tree-Depth Diagnostics (d in 1..12, None)
    print("Running Decision Tree depth diagnostics...")
    depths = list(range(1, 13)) + [None]
    depth_results = []
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train_scaled, y_train)
        preds = dt.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        depth_results.append({"max_depth": str(d), "val_rmse": rmse, "val_mae": mae})
    
    pd.DataFrame(depth_results).to_csv("outputs/part_b_tree_depth.csv", index=False)
    
    # 5. Candidate Model Benchmarks
    print("Training candidate models (Linear, Tree, Random Forest, Gradient Boosting)...")
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    model_metrics = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        model_metrics.append({"model": name, "val_rmse": rmse, "val_mae": mae, "val_r2": r2})
        
    metrics_df = pd.DataFrame(model_metrics)
    
    # 6. Gradient Boosting Learning Rate Tuning
    print("Tuning Gradient Boosting learning rates...")
    lr_results = []
    for lr in [0.01, 0.05, 0.1, 0.2, 0.3]:
        gb = GradientBoostingRegressor(learning_rate=lr, n_estimators=100, random_state=42)
        gb.fit(X_train_scaled, y_train)
        preds = gb.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        lr_results.append({"learning_rate": lr, "val_rmse": rmse})
    pd.DataFrame(lr_results).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    # 7. 5-Fold Cross-Validation on Top 2 Models (RandomForest & GradientBoosting)
    print("Running 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for name in ["RandomForest", "GradientBoosting"]:
        scores = cross_val_score(models[name], X_train_scaled, y_train, scoring="neg_root_mean_squared_error", cv=cv)
        cv_results.append({"model": name, "cv_rmse_mean": -scores.mean(), "cv_rmse_std": scores.std()})
    pd.DataFrame(cv_results).to_csv("outputs/part_b_cv.csv", index=False)
    
    # 8. Final Test Evaluation using Best Performing Model (Gradient Boosting)
    print("Evaluating winning model on test set...")
    best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    best_model.fit(X_train_scaled, y_train)
    test_preds = best_model.predict(X_test_scaled)
    
    final_test_metrics = pd.DataFrame([{
        "model": "GradientBoosting_Test",
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
        "test_mae": mean_absolute_error(y_test, test_preds),
        "test_r2": r2_score(y_test, test_preds)
    }])
    
    # Combine validation benchmark metrics and final test evaluation
    final_metrics_output = pd.concat([metrics_df, final_test_metrics], ignore_index=True)
    final_metrics_output.to_csv("outputs/part_b_metrics.csv", index=False)
    
    print("Part B completed successfully! All regression reports saved to outputs/.")

if __name__ == "__main__":
    run_part_b()