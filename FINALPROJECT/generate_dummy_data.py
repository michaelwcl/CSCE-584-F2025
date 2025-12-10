import random
from datetime import datetime, timedelta
import pandas as pd
import math
from pathlib import Path

INPUTS_DIR = Path("inputs")
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_FILE = INPUTS_DIR / "dummy_warehouse_transactions.csv"
ONLINE_PRICES_CSV_FILE = INPUTS_DIR / "dummy_online_prices.csv"
LAND_PRICES_CSV_FILE = INPUTS_DIR / "dummy_land_prices.csv"

START_DATE_STR = "2015-01-01"
END_DATE_STR = "2025-12-31"

SKU_CONFIGS = [
    {
        "sku": "WIRE_2_5MM",
        "sale_frequency_days": 3,
        "sale_quantities": [40, 60, 80, 100, 120],
        "initial_stock": 500,
        "reorder_point": 150,
        "restock_qty": 500,
        "base_price": 2.50,
    },
    {
        "sku": "TRANSFORMER_5KVA",
        "sale_frequency_days": 7,
        "sale_quantities": [5, 10, 15, 20],
        "initial_stock": 100,
        "reorder_point": 30,
        "restock_qty": 300,
        "base_price": 450.00,
    },
    {
        "sku": "OUTLET_20A",
        "sale_frequency_days": 4,
        "sale_quantities": [20, 40, 60, 80, 100],
        "initial_stock": 400,
        "reorder_point": 120,
        "restock_qty": 300,
        "base_price": 8.75,
    },
    {
        "sku": "BREAKER_63A",
        "sale_frequency_days": 10,
        "sale_quantities": [8, 12, 16, 20, 24],
        "initial_stock": 80,
        "reorder_point": 25,
        "restock_qty": 200,
        "base_price": 65.00,
    },
]

RESTOCK_TRIGGER_PROB = 0.75
SUMMER_PEAK_MONTH = 7
SUMMER_PEAK_FACTOR = 1.08
CHRISTMAS_TROUGH_MONTHS = [11, 12]
CHRISTMAS_TROUGH_FACTOR = 0.90
BASE_SEASONAL_AMPLITUDE = 0.08
HOLIDAY_SOFT_DIP = 0.92
HOLIDAY_START_MONTH = 11
HOLIDAY_END_MONTH = 12
NOISE_PROBABILITY = 0.02
DAILY_SALE_JITTER = 0.08
MAX_NO_SALE_DAYS = 6
LAND_PRICE_BASE = 15000


def daterange(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def get_trend_multiplier(current_date: datetime, start_date: datetime, end_date: datetime) -> float:
    total_days = (end_date - start_date).days
    days_elapsed = (current_date - start_date).days
    progress = days_elapsed / total_days if total_days > 0 else 0
    multiplier = 1.0 + 0.15 * (progress ** 1.5)
    return multiplier


def get_seasonal_multiplier(current_date: datetime) -> float:
    day_of_year = current_date.timetuple().tm_yday
    season_phase = 2 * math.pi * ((day_of_year - 180) / 365.0)
    smooth_wave = 1.0 + BASE_SEASONAL_AMPLITUDE * math.sin(season_phase)

    if HOLIDAY_START_MONTH <= current_date.month <= HOLIDAY_END_MONTH:
        months_into_holiday = current_date.month - HOLIDAY_START_MONTH + (current_date.day / 30.0)
        holiday_weight = min(1.0, max(0.0, months_into_holiday / 2.0))
        holiday_multiplier = 1.0 * (1 - holiday_weight) + HOLIDAY_SOFT_DIP * holiday_weight
        seasonal_multiplier = smooth_wave * holiday_multiplier
    else:
        seasonal_multiplier = smooth_wave

    return max(0.85, min(1.15, seasonal_multiplier))


def get_online_price(base_price: float, trend_multiplier: float) -> float:
    online_price = base_price / trend_multiplier
    return round(online_price, 2)


def generate_days_until_next_sale(frequency_days: int, trend_multiplier: float, seasonal_multiplier: float) -> int:
    adjusted_frequency = frequency_days / (trend_multiplier * seasonal_multiplier)
    days = max(1, int(random.gauss(adjusted_frequency, adjusted_frequency * 0.2)))
    return days


def should_apply_noise() -> bool:
    return random.random() < NOISE_PROBABILITY


def generate_transactions_for_sku(
    sku_config: dict, start_date: datetime, end_date: datetime, land_prices_df: pd.DataFrame = None
):
    sku = sku_config["sku"]
    sale_frequency_days = sku_config["sale_frequency_days"]
    sale_quantities = sku_config["sale_quantities"]
    initial_stock = sku_config["initial_stock"]
    reorder_point = sku_config["reorder_point"]
    restock_qty = sku_config["restock_qty"]
    base_price = sku_config["base_price"]

    transactions = []

    current_stock = initial_stock
    transactions.append(
        {
            "SKU": sku,
            "DATE": start_date.strftime("%Y-%m-%d"),
            "QUANTITY": initial_stock,
            "SALE/RECEIPT": "RECEIPT",
            "ONLINE_PRICE": base_price,
        }
    )

    current_date = start_date
    trend_mult = get_trend_multiplier(current_date, start_date, end_date)
    seasonal_mult = get_seasonal_multiplier(current_date)
    next_sale_date = current_date + timedelta(
        days=generate_days_until_next_sale(sale_frequency_days, trend_mult, seasonal_mult)
    )

    min_sale_quantity = max(1, min(sale_quantities))

    days_since_last_sale = 0

    while current_date <= end_date:
        trend_mult = get_trend_multiplier(current_date, start_date, end_date)
        seasonal_mult = get_seasonal_multiplier(current_date)
        online_price = get_online_price(base_price, trend_mult)
        land_price_mult = get_land_price_multiplier(current_date, land_prices_df, start_date, end_date)

        force_sale = days_since_last_sale >= MAX_NO_SALE_DAYS
        if current_date >= next_sale_date or force_sale:
            sale_qty = random.choice(sale_quantities)
            
            jitter = random.uniform(1 - DAILY_SALE_JITTER, 1 + DAILY_SALE_JITTER)
            scaled_qty = int(
                sale_qty
                * trend_mult
                * seasonal_mult
                * land_price_mult
                * jitter
            )
            scaled_qty = max(min_sale_quantity, scaled_qty)

            actual_sale = min(scaled_qty, current_stock)
            
            if actual_sale > 0:
                current_stock -= actual_sale
                days_since_last_sale = 0
                transactions.append(
                    {
                        "SKU": sku,
                        "DATE": current_date.strftime("%Y-%m-%d"),
                        "QUANTITY": -actual_sale,
                        "SALE/RECEIPT": "SALE",
                        "ONLINE_PRICE": online_price,
                    }
                )
            
            next_sale_date = current_date + timedelta(
                days=generate_days_until_next_sale(sale_frequency_days, trend_mult, seasonal_mult)
            )
        else:
            days_since_last_sale += 1

        if current_stock < reorder_point:
            if random.random() < RESTOCK_TRIGGER_PROB:
                current_stock += restock_qty
                transactions.append(
                    {
                        "SKU": sku,
                        "DATE": current_date.strftime("%Y-%m-%d"),
                        "QUANTITY": restock_qty,
                        "SALE/RECEIPT": "RECEIPT",
                        "ONLINE_PRICE": online_price,
                    }
                )

        if should_apply_noise():
            noise_qty = random.choice(sale_quantities)
            noise_qty = max(min_sale_quantity, int(noise_qty * random.uniform(0.9, 1.1)))
            if random.random() < 0.5 and current_stock >= noise_qty:
                current_stock -= noise_qty
                transactions.append(
                    {
                        "SKU": sku,
                        "DATE": current_date.strftime("%Y-%m-%d"),
                        "QUANTITY": -noise_qty,
                        "SALE/RECEIPT": "SALE",
                        "ONLINE_PRICE": online_price,
                    }
                )
            else:
                current_stock += noise_qty
                transactions.append(
                    {
                        "SKU": sku,
                        "DATE": current_date.strftime("%Y-%m-%d"),
                        "QUANTITY": noise_qty,
                        "SALE/RECEIPT": "RECEIPT",
                        "ONLINE_PRICE": online_price,
                    }
                )

        current_date += timedelta(days=1)

    return transactions


def get_land_price_multiplier(current_date: datetime, land_prices_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> float:
    if land_prices_df is None or land_prices_df.empty:
        return 1.0
    
    current_month = current_date.strftime("%Y-%m")
    
    matching_prices = land_prices_df[land_prices_df["DATE"] == current_month]
    if matching_prices.empty:
        return 1.0
    
    current_land_price = matching_prices["LAND_PRICE_PER_ACRE"].values[0]
    
    baseline_land_price = land_prices_df["LAND_PRICE_PER_ACRE"].mean()
    
    price_ratio = baseline_land_price / current_land_price
    land_price_multiplier = 0.85 + (price_ratio - 1.0) * 0.2
    land_price_multiplier = max(0.8, min(1.2, land_price_multiplier))
    
    return land_price_multiplier


def generate_all_transactions() -> pd.DataFrame:
    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

    land_prices = generate_land_prices(start_date, end_date)

    all_txns = []

    for sku_config in SKU_CONFIGS:
        sku_txns = generate_transactions_for_sku(sku_config, start_date, end_date, land_prices)
        all_txns.extend(sku_txns)

    df = pd.DataFrame(all_txns)

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["SKU", "DATE"]).reset_index(drop=True)

    df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d")

    return df


def generate_online_prices(start_date: datetime, end_date: datetime, sku_configs: list) -> pd.DataFrame:
    all_prices = []
    date_list = list(daterange(start_date, end_date))
    for sku_cfg in sku_configs:
        base_price = sku_cfg["base_price"]
        sku = sku_cfg["sku"]
        price = base_price * random.uniform(0.95, 1.05)
        for i, date in enumerate(date_list):
            drift = 1.0 + 0.10 * ((date - start_date).days / (end_date - start_date).days)
            
            fluctuation_pct = random.gauss(0, 0.05)
            fluctuation = base_price * fluctuation_pct
            
            if random.random() < 0.01:
                fluctuation += base_price * random.gauss(0, 0.10)
            
            price = base_price * (1.0 + (drift - 1.0) * 0.5) + fluctuation
            price = max(base_price * 0.5, price)
            
            final_price = round(price, 2)
            all_prices.append({
                "DATE": date.strftime("%Y-%m-%d"),
                "SKU": sku,
                "ONLINE_UNIT_PRICE": final_price
            })
    df_prices = pd.DataFrame(all_prices)
    return df_prices


def generate_land_prices(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    all_prices = []
    
    current_date = start_date.replace(day=1)
    price = LAND_PRICE_BASE
    
    total_months = 120
    month_count = 0
    
    while current_date <= end_date:
        linear_drift = 1.0 + 0.12 * (month_count / total_months)
        
        monthly_change = random.gauss(1.0, 0.015)
        
        price = LAND_PRICE_BASE * linear_drift * monthly_change
        
        price = max(LAND_PRICE_BASE * 0.8, min(price, LAND_PRICE_BASE * 1.5))
        
        final_price = round(price, 2)
        all_prices.append({
            "DATE": current_date.strftime("%Y-%m"),
            "LAND_PRICE_PER_ACRE": final_price
        })
        
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
        
        month_count += 1
    
    df_land = pd.DataFrame(all_prices)
    return df_land


def main():
    df = generate_all_transactions()
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Dummy transaction data written to: {OUTPUT_CSV_FILE}")
    print(f"Number of rows: {len(df)}")
    print(f"Date range: {START_DATE_STR} to {END_DATE_STR}")
    print("Preview:")
    print(df.head(10))

    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d")
    
    df_online = generate_online_prices(start_date, end_date, SKU_CONFIGS)
    df_online.to_csv(ONLINE_PRICES_CSV_FILE, index=False)
    print(f"\nDummy online prices written to: {ONLINE_PRICES_CSV_FILE}")
    print("Online price preview:")
    print(df_online.head(10))
    
    df_land = generate_land_prices(start_date, end_date)
    df_land.to_csv(LAND_PRICES_CSV_FILE, index=False)
    print(f"\nDummy land prices written to: {LAND_PRICES_CSV_FILE}")
    print("Land price preview:")
    print(df_land.head(12))


if __name__ == "__main__":
    main()
