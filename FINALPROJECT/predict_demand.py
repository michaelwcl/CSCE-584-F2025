# predict_demand.py

import json
import numpy as np
import pandas as pd
from joblib import load

# ========= FILE NAMES (adjust as needed) =========

WEEKLY_FEATURES_FILE = "data/item_weekly_features.csv"
CLASSIFIER_MODEL_FILE = "mldata/demand_nonzero_classifier.joblib"
REGRESSOR_MODEL_FILE = "mldata/demand_positive_regressor.joblib"
MODEL_METADATA_FILE = "mldata/demand_model_metadata.json"

OUTPUT_PREDICTIONS_FILE = "data/predicted_demand_next_4_weeks.csv"

# =================================================


def load_weekly_features(file_path: str) -> pd.DataFrame:
    """
    Load weekly features and rebuild any simple calendar features
    needed by the model (e.g. week_of_year).
    """
    df = pd.read_csv(file_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    return df


def apply_feature_filling(df: pd.DataFrame, feature_cols):
    """
    Apply the same missing-value handling used in training.
    Returns an array-like X with no NaNs for the given feature columns.
    """
    X = df[feature_cols].copy()

    for col in ["lag1_sales", "lag2_sales", "lag4_sales"]:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)

    if "weeks_since_last_sale" in X.columns:
        X["weeks_since_last_sale"] = X["weeks_since_last_sale"].fillna(999.0)

    if "on_hand_start" in X.columns:
        X["on_hand_start"] = X["on_hand_start"].fillna(0.0)

    if "week_of_year" in X.columns:
        X["week_of_year"] = X["week_of_year"].fillna(0).astype(int)

    if "avg_online_price" in X.columns:
        X["avg_online_price"] = X["avg_online_price"].fillna(method="ffill").fillna(method="bfill")

    if "land_price_per_acre" in X.columns:
        X["land_price_per_acre"] = X["land_price_per_acre"].fillna(method="ffill").fillna(method="bfill")

    for col in ["rolling_4w_avg_sales", "rolling_8w_avg_sales", "sku_avg_sales", "sku_encoded"]:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)

    return X


def build_future_weeks(df: pd.DataFrame, clf, regr, feature_cols, base_week: pd.Timestamp, weeks_ahead: int = 4) -> pd.DataFrame:
    """
    Build future week rows with recursive predictions.
    Uses model predictions for lag features when actual data unavailable.
    """
    all_skus = df["SKU"].unique()
    future_rows = []
    future_predictions = {}
    
    for sku in all_skus:
        sku_df = df[df["SKU"] == sku].sort_values("week_start")
        latest_row = sku_df.iloc[-1].copy()
        
        sales_history = deque(sku_df[["week_start", "units_sold"]].tail(4).to_dict("records"), maxlen=4)
        predicted_sales_queue = deque(maxlen=4)
        recent_sales = list(sku_df["units_sold"].tail(8).values)
        recent_avg4 = float(sku_df["rolling_4w_avg_sales"].iloc[-1])
        recent_avg8 = float(sku_df["rolling_8w_avg_sales"].iloc[-1])
        
        sku_avg = latest_row.get("sku_avg_sales", float(sku_df["units_sold"].mean()))
        sku_code = latest_row.get("sku_encoded", 0.0)

        for week_offset in range(1, weeks_ahead + 1):
            future_week_start = base_week + pd.Timedelta(weeks=week_offset)
            future_week_of_year = future_week_start.isocalendar()[1]
            
            # Lag features: use actual history, then switch to predictions
            if week_offset == 1:
                lag1 = latest_row["units_sold"] if latest_row["units_sold"] > 0 else 0.0
                lag2 = sales_history[2]["units_sold"] if len(sales_history) > 2 else 0.0
                lag4 = sales_history[0]["units_sold"] if len(sales_history) > 3 else 0.0
            else:
                # Use recursive predictions for older lags
                lag1 = predicted_sales_queue[-1] if len(predicted_sales_queue) > 0 else 0.0
                lag2 = predicted_sales_queue[-2] if len(predicted_sales_queue) > 1 else 0.0
                lag4 = predicted_sales_queue[-4] if len(predicted_sales_queue) > 3 else 0.0
            
            # Rolling averages: compute from recent sales or use smoothed value
            rolling4 = np.mean(recent_sales[-4:]) if len(recent_sales) else recent_avg4
            rolling8 = np.mean(recent_sales[-8:]) if len(recent_sales) else recent_avg8

            future_row = {
                "SKU": sku,
                "week_start": future_week_start,
                "week_of_year": future_week_of_year,
                "on_hand_start": latest_row.get("on_hand_end", latest_row.get("on_hand_start", 0.0)),
                "lag1_sales": lag1,
                "lag2_sales": lag2,
                "lag4_sales": lag4,
                "avg_online_price": latest_row.get("avg_online_price", np.nan),
                "land_price_per_acre": latest_row.get("land_price_per_acre", np.nan),
                "rolling_4w_avg_sales": rolling4,
                "rolling_8w_avg_sales": rolling8,
                "sku_avg_sales": sku_avg,
                "sku_encoded": sku_code,
                "units_sold": np.nan,
            }
            future_rows.append(future_row)
            
            # Make prediction for this week to use in next week's lags
            X_this_week = pd.DataFrame([future_row])
            X_this_week_filled = apply_feature_filling(X_this_week, feature_cols)
            
            prob_sale = clf.predict_proba(X_this_week_filled)[:, 1][0]
            log_units = regr.predict(X_this_week_filled)[0]
            pred_units = np.expm1(log_units)
            expected_units = prob_sale * pred_units
            
            predicted_sales_queue.append(expected_units)
            future_predictions[(sku, week_offset)] = expected_units
            recent_sales.append(expected_units)
            if len(recent_sales) > 8:
                recent_sales.pop(0)
    
    df_future = pd.DataFrame(future_rows)
    return df_future, future_predictions


def main():
    print("Loading data...")
    df = load_weekly_features(WEEKLY_FEATURES_FILE)
    
    with open(MODEL_METADATA_FILE, "r") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    
    clf = load(CLASSIFIER_MODEL_FILE)
    regr = load(REGRESSOR_MODEL_FILE)
    print("Loaded trained models.")
    
    latest_week = df["week_start"].max()
    
    df_future, future_preds = build_future_weeks(df, clf, regr, feature_cols, latest_week, weeks_ahead=4)
    
    X_future = apply_feature_filling(df_future, feature_cols)
    
    prob_nonzero = clf.predict_proba(X_future)[:, 1]
    log_pred = regr.predict(X_future)
    pred_units_cond = np.expm1(log_pred)
    expected_units = prob_nonzero * pred_units_cond
    
    predictions = df_future[["SKU", "week_start"]].copy()
    predictions["probability_any_sale"] = prob_nonzero
    predictions["predicted_units_given_sale"] = pred_units_cond
    predictions["expected_units_sold"] = expected_units
    
    predictions.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
    print(f"\nSaved 4-week demand forecast to {OUTPUT_PREDICTIONS_FILE}")
    print(predictions.head(12))


if __name__ == "__main__":
    from collections import deque
    main()
