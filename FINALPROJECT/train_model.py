import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    classification_report,
)
from joblib import dump

WEEKLY_FEATURES_FILE = "data/item_weekly_features.csv"

CLASSIFIER_MODEL_FILE = "mldata/demand_nonzero_classifier.joblib"
REGRESSOR_MODEL_FILE = "mldata/demand_positive_regressor.joblib"
MODEL_METADATA_FILE = "mldata/demand_model_metadata.json"

FEATURE_COLS = [
    "on_hand_start",
    "lag1_sales",
    "lag2_sales",
    "lag4_sales",
    "rolling_4w_avg_sales",
    "rolling_8w_avg_sales",
    "sku_avg_sales",
    "sku_encoded",
    "week_of_year",
    "avg_online_price",
    "land_price_per_acre",
]

RANDOM_STATE = 42
TRAIN_YEARS = 9


def load_weekly_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)

    return df


def prepare_training_data(df: pd.DataFrame):
    df["units_sold"] = df["units_sold"].astype(float)
    df["nonzero_flag"] = (df["units_sold"] > 0).astype(int)

    for col in ["lag1_sales", "lag2_sales", "lag4_sales"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    if "weeks_since_last_sale" in df.columns:
        df["weeks_since_last_sale"] = df["weeks_since_last_sale"].fillna(999.0)

    if "on_hand_start" in df.columns:
        df["on_hand_start"] = df["on_hand_start"].fillna(0.0)

    if "avg_online_price" in df.columns:
        df["avg_online_price"] = df.groupby("SKU")["avg_online_price"].ffill().bfill()

    if "land_price_per_acre" in df.columns:
        df["land_price_per_acre"] = df["land_price_per_acre"].ffill().bfill()

    for col in ["rolling_4w_avg_sales", "rolling_8w_avg_sales", "sku_avg_sales", "sku_encoded"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    X = df[FEATURE_COLS].copy()
    y_units = df["units_sold"].copy()
    y_nonzero = df["nonzero_flag"].copy()

    positive_mask = y_units > 0
    y_reg = np.log1p(y_units[positive_mask])
    X_reg = X[positive_mask].copy()

    return X, y_units, y_nonzero, X_reg, y_reg


def train_test_year_split(df: pd.DataFrame, train_years: int):
    df_sorted = df.sort_values("week_start")
    min_year = df_sorted["week_start"].dt.year.min()
    max_year = df_sorted["week_start"].dt.year.max()
    
    split_year = min_year + train_years - 1
    
    if split_year >= max_year:
        raise ValueError(
            f"Not enough years in data. Have {max_year - min_year + 1} years, "
            f"but requested {train_years} years for training."
        )
    
    train_mask = df["week_start"].dt.year <= split_year
    test_mask = df["week_start"].dt.year > split_year
    
    split_week = df[train_mask]["week_start"].max()
    
    return train_mask, test_mask, split_week


def main():
    df = load_weekly_data(WEEKLY_FEATURES_FILE)
    print(f"Loaded {len(df)} weekly item records from {WEEKLY_FEATURES_FILE}")

    train_mask, test_mask, split_week = train_test_year_split(df, train_years=TRAIN_YEARS)
    print(f"Train on years up to {split_week.year}, "
          f"test on year(s) after that.")
    print(f"Train set: {train_mask.sum()} rows, Test set: {test_mask.sum()} rows")

    X, y_units, y_nonzero, X_reg, y_reg = prepare_training_data(df)

    X_train_clf = X[train_mask].copy()
    y_train_clf = y_nonzero[train_mask].copy()
    X_test_clf = X[test_mask].copy()
    y_test_clf = y_nonzero[test_mask].copy()

    reg_train_mask = train_mask & (y_units > 0)
    reg_test_mask = test_mask & (y_units > 0)

    X_train_reg = X[reg_train_mask].copy()
    y_train_reg = np.log1p(y_units[reg_train_mask].values)

    X_test_reg = X[reg_test_mask].copy()
    y_test_reg_actual = y_units[reg_test_mask].values

    unique_classes = np.unique(y_train_clf)
    if len(unique_classes) > 1:
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=y_train_clf
        )
        weight_map = {cls: w for cls, w in zip(unique_classes, class_weights)}
        sample_weights = y_train_clf.map(weight_map).values
    else:
        sample_weights = None

    clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train_clf, y_train_clf, sample_weight=sample_weights)
    print("Trained nonzero-demand classifier.")

    regr = GradientBoostingRegressor(random_state=RANDOM_STATE)
    regr.fit(X_train_reg, y_train_reg)
    print("Trained positive-demand regressor.")

    y_pred_clf = clf.predict(X_test_clf)
    print("\n=== Classification report: nonzero demand (test set) ===")
    print(classification_report(y_test_clf, y_pred_clf, digits=3))

    prob_nonzero = clf.predict_proba(X[test_mask])[:, 1]
    log_pred_units = regr.predict(X[test_mask])
    pred_units_conditional = np.expm1(log_pred_units)
    y_pred_expected = prob_nonzero * pred_units_conditional

    y_test_units_all = y_units[test_mask].values

    mae = mean_absolute_error(y_test_units_all, y_pred_expected)
    r2 = r2_score(y_test_units_all, y_pred_expected)

    print("\n=== Demand regression metrics on test set (expected demand) ===")
    print(f"MAE (mean absolute error): {mae:.3f}")
    print(f"R^2: {r2:.3f}")

    dump(clf, CLASSIFIER_MODEL_FILE)
    dump(regr, REGRESSOR_MODEL_FILE)
    print(f"\nSaved classifier to: {CLASSIFIER_MODEL_FILE}")
    print(f"Saved regressor to:  {REGRESSOR_MODEL_FILE}")

    metadata = {
        "feature_columns": FEATURE_COLS,
        "weekly_features_file": WEEKLY_FEATURES_FILE,
        "train_years": TRAIN_YEARS,
        "split_week": str(split_week.date()),
        "random_state": RANDOM_STATE,
    }

    with open(MODEL_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model metadata to: {MODEL_METADATA_FILE}")


if __name__ == "__main__":
    main()
