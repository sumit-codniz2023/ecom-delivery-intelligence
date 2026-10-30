import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

def run_part_c():
    os.makedirs("outputs", exist_ok=True)
    
    print("Loading modelling table for classification...")
    df = pd.read_csv("data/modelling_table.csv")
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 1. Stratified Two-Step Split: 70% Train, 15% Validation, 15% Test
    print("Performing stratified data split...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    print(f"Split sizes -> Train: {len(X_train)} ({y_train.mean():.2%} positive), "
          f"Val: {len(X_val)} ({y_val.mean():.2%} positive), "
          f"Test: {len(X_test)} ({y_test.mean():.2%} positive)")
    
    # Scaling (Fitted STRICTLY on Training Set Only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Candidate Models Benchmark
    print("Training candidate classification models...")
    models = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    model_metrics = []
    # Blind Baseline (majority class predictor)
    baseline_preds = np.zeros_like(y_val)
    model_metrics.append({
        "model": "BlindBaseline",
        "val_accuracy": accuracy_score(y_val, baseline_preds),
        "val_precision": precision_score(y_val, baseline_preds, zero_division=0),
        "val_recall": recall_score(y_val, baseline_preds, zero_division=0),
        "val_f1": f1_score(y_val, baseline_preds, zero_division=0),
        "val_roc_auc": 0.50
    })
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        preds = model.predict(X_val_scaled)
        probs = model.predict_proba(X_val_scaled)[:, 1]
        
        model_metrics.append({
            "model": name,
            "val_accuracy": accuracy_score(y_val, preds),
            "val_precision": precision_score(y_val, preds, zero_division=0),
            "val_recall": recall_score(y_val, preds, zero_division=0),
            "val_f1": f1_score(y_val, preds, zero_division=0),
            "val_roc_auc": roc_auc_score(y_val, probs)
        })
        
    pd.DataFrame(model_metrics).to_csv("outputs/part_c_models.csv", index=False)
    
    # 3. Feature Selection & Hyperparameter Tuning
    print("Performing feature selection with SelectKBest...")
    selector = SelectKBest(score_func=f_classif, k=7)
    selector.fit(X_train_scaled, y_train)
    
    scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": selector.scores_,
        "p_value": selector.pvalues_,
        "selected": selector.get_support()
    }).sort_values(by="f_score", ascending=False)
    scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    # Hyperparameter Grid Search on Random Forest
    print("Running hyperparameter tuning (GridSearchCV)...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
    grid.fit(X_train_scaled, y_train)
    
    tuning_results = pd.DataFrame(grid.cv_results_)[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'mean_test_score', 'std_test_score']]
    tuning_results.to_csv("outputs/part_c_tuning.csv", index=False)
    
    best_rf = grid.best_estimator_
    print(f"Best Hyperparameters: {grid.best_params_}")
    
    # 4. Threshold Optimization (Operations Rule: Max threshold with validation recall >= 0.80)
    print("Optimizing classification decision threshold...")
    val_probs = best_rf.predict_proba(X_val_scaled)[:, 1]
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    thresh_records = []
    
    for t in thresholds:
        preds_t = (val_probs >= t).astype(int)
        rec = recall_score(y_val, preds_t, zero_division=0)
        prec = precision_score(y_val, preds_t, zero_division=0)
        f1 = f1_score(y_val, preds_t, zero_division=0)
        thresh_records.append({"threshold": round(t, 2), "recall": rec, "precision": prec, "f1": f1})
        
    thresh_df = pd.DataFrame(thresh_records)
    thresh_df.to_csv("outputs/part_c_thresholds.csv", index=False)
    
    # Apply Operations Rule: Highest threshold where recall >= 0.80
    valid_thresh_rows = thresh_df[thresh_df["recall"] >= 0.80]
    if not valid_thresh_rows.empty:
        optimal_threshold = valid_thresh_rows.iloc[-1]["threshold"] # Highest threshold meeting criterion
    else:
        optimal_threshold = 0.50 # Fallback
    
    print(f"Selected Optimal Operational Threshold: {optimal_threshold}")
    
    # 5. Final Test Evaluation at Optimal Threshold
    print("Evaluating tuned model on test set...")
    test_probs = best_rf.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    final_test_results = pd.DataFrame([{
        "model": "RandomForest_Optimized",
        "threshold": optimal_threshold,
        "test_accuracy": accuracy_score(y_test, test_preds),
        "test_precision": precision_score(y_test, test_preds, zero_division=0),
        "test_recall": recall_score(y_test, test_preds, zero_division=0),
        "test_f1": f1_score(y_test, test_preds, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, test_probs)
    }])
    final_test_results.to_csv("outputs/part_c_final_test.csv", index=False)
    
    print("Part C completed successfully! All classification outputs saved to outputs/.")

if __name__ == "__main__":
    run_part_c()