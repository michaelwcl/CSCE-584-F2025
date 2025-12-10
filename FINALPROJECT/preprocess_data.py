import pandas as pd
import numpy as np

RAW_TRANSACTIONS_FILE = "inputs/dummy_warehouse_transactions.csv"
ONLINE_PRICES_FILE = "inputs/dummy_online_prices.csv"
LAND_PRICES_FILE = "inputs/dummy_land_prices.csv"
DAILY_STOCK_FILE = "data/daily_stock_levels.csv"
WEEKLY_FEATURES_FILE = "data/item_weekly_features.csv"


def load_and_clean_transactions(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    df.columns = [c.strip().upper() for c in df.columns]

    required_cols = {"SKU", "DATE", "QUANTITY", "SALE/RECEIPT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df["DATE"] = pd.to_datetime(df["DATE"])

    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")
    if df["QUANTITY"].isna().any():
        raise ValueError("Some QUANTITY values could not be converted to numbers.")

    df["SALE/RECEIPT"] = df["SALE/RECEIPT"].astype(str).str.strip().str.upper()

    df["qty_sold"] = np.where(
        df["SALE/RECEIPT"] == "SALE", df["QUANTITY"].abs(), 0.0
    )
    df["qty_received"] = np.where(
        df["SALE/RECEIPT"] == "RECEIPT", df["QUANTITY"].abs(), 0.0
    )

    df = df[["SKU", "DATE", "qty_sold", "qty_received"]]

    return df


def load_online_prices(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.rename(columns={"ONLINE_UNIT_PRICE": "online_price"})
    return df[["DATE", "SKU", "online_price"]]


def load_land_prices(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"] + "-01")
    df = df.rename(columns={"LAND_PRICE_PER_ACRE": "land_price_per_acre"})
    return df[["DATE", "land_price_per_acre"]]


def build_daily_stock_table(transactions: pd.DataFrame, online_prices: pd.DataFrame, land_prices: pd.DataFrame) -> pd.DataFrame:
    daily_flows = (
        transactions.groupby(["SKU", "DATE"], as_index=False)[["qty_sold", "qty_received"]]
        .sum()
    )

    all_skus = daily_flows["SKU"].unique()
    start_date = daily_flows["DATE"].min()
    end_date = daily_flows["DATE"].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    full_index = pd.MultiIndex.from_product(
        [all_skus, all_dates], names=["SKU", "DATE"]
    )
    base = pd.DataFrame(index=full_index).reset_index()

    daily = base.merge(
        daily_flows,
        on=["SKU", "DATE"],
        how="left",
    ).fillna({"qty_sold": 0.0, "qty_received": 0.0})

    daily = daily.sort_values(["SKU", "DATE"])
    daily["net_flow"] = daily["qty_received"] - daily["qty_sold"]
    daily["on_hand_end"] = daily.groupby("SKU")["net_flow"].cumsum()
    daily["on_hand_start"] = daily["on_hand_end"] - daily["net_flow"]
    daily["stockout_flag"] = (daily["on_hand_start"] <= 0) & (daily["qty_sold"] > 0)

    daily = daily.merge(
        online_prices,
        on=["SKU", "DATE"],
        how="left"
    )

    daily["month_start"] = daily["DATE"].dt.to_period("M").dt.start_time
    land_prices["month_start"] = land_prices["DATE"].dt.to_period("M").dt.start_time
    
    daily = daily.merge(
        land_prices[["month_start", "land_price_per_acre"]],
        on="month_start",
        how="left"
    )
    
    daily = daily.drop(columns=["month_start"])

    return daily


def build_weekly_features(daily_stock: pd.DataFrame) -> pd.DataFrame:
    daily = daily_stock.copy()
    daily = daily.set_index("DATE")

    weekly_parts = []

    for sku, grp in daily.groupby("SKU"):
        weekly = grp.resample("W-MON").agg(
            {
                "qty_sold": "sum",
                "qty_received": "sum",
                "on_hand_start": "first",
                "on_hand_end": "last",
                "online_price": "mean",
                "land_price_per_acre": "first",
            }
        )

        weekly = weekly.reset_index().rename(columns={"DATE": "week_start"})
        weekly["SKU"] = sku
        weekly_parts.append(weekly)

    weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly = weekly.sort_values(["SKU", "week_start"]).reset_index(drop=True)

    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month.astype(int)
    weekly["quarter"] = weekly["week_start"].dt.quarter.astype(int)
    weekly["seasonal_sin"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52)
    weekly["seasonal_cos"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52)

    weekly["lag1_sales"] = weekly.groupby("SKU")["qty_sold"].shift(1)
    weekly["lag2_sales"] = weekly.groupby("SKU")["qty_sold"].shift(2)
    weekly["lag4_sales"] = weekly.groupby("SKU")["qty_sold"].shift(4)

    def compute_weeks_since_last_sale(x: pd.Series) -> pd.Series:
        last_nonzero_idx = -1
        result = []
        for i, val in enumerate(x):
            if val > 0:
                last_nonzero_idx = i
                result.append(0)
            else:
                if last_nonzero_idx == -1:
                    result.append(None)
                else:
                    result.append(i - last_nonzero_idx)
        return pd.Series(result, index=x.index)

    weekly["weeks_since_last_sale"] = (
        weekly.groupby("SKU")["qty_sold"].transform(compute_weeks_since_last_sale)
    )

    weekly = weekly.rename(
        columns={
            "qty_sold": "units_sold",
            "qty_received": "units_received",
            "online_price": "avg_online_price",
        }
    )

    weekly["rolling_4w_avg_sales"] = (
        weekly.groupby("SKU")["units_sold"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )
    weekly["rolling_8w_avg_sales"] = (
        weekly.groupby("SKU")["units_sold"]
        .transform(lambda s: s.rolling(window=8, min_periods=1).mean())
    )

    weekly["sku_avg_sales"] = weekly.groupby("SKU")["units_sold"].transform("mean")
    weekly["sku_encoded"] = weekly["SKU"].astype("category").cat.codes

    def _smooth_lag(col):
        fallback = weekly["rolling_4w_avg_sales"].fillna(0.0)
        capped = np.minimum(
            weekly[col].fillna(fallback),
            np.maximum(fallback, weekly["rolling_4w_avg_sales"] * 1.6)
        )
        weekly[col] = capped

    _smooth_lag("lag1_sales")
    _smooth_lag("lag2_sales")
    _smooth_lag("lag4_sales")

    return weekly


def main():
    transactions = load_and_clean_transactions(RAW_TRANSACTIONS_FILE)
    
    online_prices = load_online_prices(ONLINE_PRICES_FILE)
    land_prices = load_land_prices(LAND_PRICES_FILE)

    daily_stock = build_daily_stock_table(transactions, online_prices, land_prices)
    daily_stock.to_csv(DAILY_STOCK_FILE, index=False)

    weekly_features = build_weekly_features(daily_stock)
    weekly_features.to_csv(WEEKLY_FEATURES_FILE, index=False)

    print(f"Saved daily stock data to {DAILY_STOCK_FILE}")
    print(f"Saved weekly ML features to {WEEKLY_FEATURES_FILE}")


if __name__ == "__main__":
    main()
