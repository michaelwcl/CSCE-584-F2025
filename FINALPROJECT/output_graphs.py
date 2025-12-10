import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

OUTPUT_DIR = "graphs"

def save_debug_table(df: pd.DataFrame, name: str):
    path = Path(OUTPUT_DIR) / f"{name}.csv"
    df.to_csv(path, index=True)
    print(f"Debug CSV saved to: {path}")

# Read the generated dummy transaction data
INPUT_CSV_FILE = "inputs/dummy_warehouse_transactions.csv"
WEEKLY_FEATURES_FILE = "data/item_weekly_features.csv"
PREDICTIONS_FILE = "data/predicted_demand_next_4_weeks.csv"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_and_process_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    
    # Convert DATE to datetime
    df["DATE"] = pd.to_datetime(df["DATE"])
    
    # Filter for sales only (exclude receipts)
    sales_df = df[df["SALE/RECEIPT"] == "SALE"].copy()
    
    # Make quantity positive for easier interpretation
    sales_df["QUANTITY"] = -sales_df["QUANTITY"]
    
    # Extract year and week number
    sales_df["YEAR_WEEK"] = sales_df["DATE"].dt.isocalendar().year.astype(str) + "-W" + sales_df["DATE"].dt.isocalendar().week.astype(str).str.zfill(2)
    sales_df["WEEK_START"] = sales_df["DATE"] - pd.to_timedelta(sales_df["DATE"].dt.dayofweek, unit='D')
    
    # Aggregate sales by week and SKU
    weekly_sales = sales_df.groupby(["YEAR_WEEK", "WEEK_START", "SKU"])["QUANTITY"].sum().reset_index()
    weekly_sales = weekly_sales.sort_values(["SKU", "WEEK_START"]).reset_index(drop=True)
    
    return weekly_sales


def plot_all_years_trends(weekly_sales: pd.DataFrame):
    """Plot sales per week for all years, one subplot per SKU."""
    skus = sorted(weekly_sales["SKU"].unique())
    
    fig, axes = plt.subplots(len(skus), 1, figsize=(18, 4*len(skus)))
    if len(skus) == 1:
        axes = [axes]
    
    for idx, sku in enumerate(skus):
        sku_data = weekly_sales[weekly_sales["SKU"] == sku].sort_values("WEEK_START")
        
        # Pivot so each year has a column
        sku_data["YEAR"] = sku_data["WEEK_START"].dt.year
        pivot = sku_data.pivot_table(
            index=sku_data.groupby("YEAR").cumcount(),
            columns="YEAR",
            values="QUANTITY",
            fill_value=0
        )
        
        week_numbers = np.arange(len(pivot))
        
        for year in sorted(pivot.columns):
            quantities = pivot[year].values
            axes[idx].plot(week_numbers, quantities, marker="o", linewidth=2, label=str(year), alpha=0.7, markersize=3)
        
        axes[idx].set_xlabel("Week Number (within year)", fontsize=11)
        axes[idx].set_ylabel("Sales Quantity (units)", fontsize=11)
        axes[idx].set_title(f"Weekly Sales Trends: {sku} (All Years)", fontsize=12, fontweight="bold")
        axes[idx].legend(loc="best", fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{OUTPUT_DIR}/sales_trends_all_years.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")
    plt.close()
    
    save_debug_table(
        pd.concat(
            {sku: weekly_sales[weekly_sales["SKU"] == sku]
             .pivot_table(index="WEEK_START", columns="SKU", values="QUANTITY", fill_value=0)}
            ), "sales_trends_all_years_data"
    )


def plot_sales_trends_random_year(weekly_sales: pd.DataFrame):
    """Plot sales per week for each SKU for a random year."""
    # Extract all years present
    years = weekly_sales["WEEK_START"].dt.year.unique()
    chosen_year = random.choice(years)
    print(f"Plotting sales for random year: {chosen_year}")

    # Filter for the chosen year
    year_sales = weekly_sales[weekly_sales["WEEK_START"].dt.year == chosen_year]

    # Pivot so each SKU has a column, indexed by WEEK_START
    pivot = year_sales.pivot_table(
        index="WEEK_START", columns="SKU", values="QUANTITY", fill_value=0
    )
    week_numbers = np.arange(len(pivot))
    
    plt.figure(figsize=(16, 8))
    
    for sku in sorted(pivot.columns):
        quantities = pivot[sku].values
        plt.plot(week_numbers, quantities, marker="o", linewidth=2, label=sku, alpha=0.7, markersize=3)
    
    plt.xlabel(f"Week Number ({chosen_year})", fontsize=12)
    plt.ylabel("Sales Quantity (units)", fontsize=12)
    plt.title(f"Weekly Sales Trends by SKU ({chosen_year})", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f"{OUTPUT_DIR}/sales_trends_random_year_{chosen_year}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")
    plt.close()
    
    save_debug_table(pivot, f"sales_trends_random_year_{chosen_year}_data")


def load_weekly_features(file_path: str) -> pd.DataFrame:
    """Load weekly features to compute historical sale probabilities."""
    df = pd.read_csv(file_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    for col in ["rolling_4w_avg_sales", "rolling_8w_avg_sales", "sku_avg_sales", "sku_encoded"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def compute_actual_sale_probabilities(weekly_features: pd.DataFrame) -> dict:
    """Compute historical probability of any sale per SKU (% of weeks with sales > 0)."""
    probabilities = {}
    
    for sku in weekly_features["SKU"].unique():
        sku_data = weekly_features[weekly_features["SKU"] == sku]
        weeks_with_sales = (sku_data["units_sold"] > 0).sum()
        total_weeks = len(sku_data)
        prob = weeks_with_sales / total_weeks if total_weeks > 0 else 0.0
        probabilities[sku] = prob
    
    return probabilities


def create_prediction_comparison_table(predictions_file: str, actual_probs: dict):
    """Create a table comparing predicted vs actual sale probabilities."""
    df_pred = pd.read_csv(predictions_file)
    
    # Group by SKU and get average predicted probability
    predicted_probs = df_pred.groupby("SKU")["probability_any_sale"].mean()
    
    comparison_data = []
    for sku in sorted(df_pred["SKU"].unique()):
        pred_prob = predicted_probs.get(sku, 0.0)
        actual_prob = actual_probs.get(sku, 0.0)
        difference = pred_prob - actual_prob
        
        comparison_data.append({
            "SKU": sku,
            "Predicted Prob (avg)": f"{pred_prob:.4f}",
            "Actual Prob (historical)": f"{actual_prob:.4f}",
            "Difference": f"{difference:+.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create a simple table visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=comparison_df.values,
        colLabels=comparison_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(comparison_df) + 1):
        for j in range(len(comparison_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title("Predicted vs Actual Sale Probabilities (Next 4 Weeks)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    
    output_file = f"{OUTPUT_DIR}/prediction_vs_actual_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison table saved to: {output_file}")
    plt.close()
    
    save_debug_table(comparison_df.set_index("SKU"), "prediction_vs_actual_comparison_data")


def create_sensitivity_analysis_chart(predictions_file: str, weekly_features: pd.DataFrame):
    """
    Create sensitivity analysis: show how predicted demand changes with different
    online prices and land prices for each SKU.
    """
    df_pred = pd.read_csv(predictions_file)
    
    # Get latest non-null prices for each SKU
    weekly_features_latest = weekly_features.dropna(subset=["avg_online_price", "land_price_per_acre"])
    weekly_features_latest = weekly_features_latest.sort_values("week_start").groupby("SKU").tail(1)
    
    skus = sorted(df_pred["SKU"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    sensitivity_records = []
    
    for idx, sku in enumerate(skus):
        ax = axes[idx]
        
        # Get baseline values for this SKU
        sku_baseline = weekly_features_latest[weekly_features_latest["SKU"] == sku]
        if sku_baseline.empty:
            # Fallback: use mean of all non-null prices for this SKU
            sku_all = weekly_features[weekly_features["SKU"] == sku].dropna(subset=["avg_online_price", "land_price_per_acre"])
            if sku_all.empty:
                print(f"Warning: No valid price data for {sku}, skipping sensitivity analysis")
                continue
            baseline_online_price = sku_all["avg_online_price"].mean()
            baseline_land_price = sku_all["land_price_per_acre"].mean()
        else:
            baseline_online_price = sku_baseline["avg_online_price"].values[0]
            baseline_land_price = sku_baseline["land_price_per_acre"].values[0]
        
        # Ensure prices are non-zero
        if baseline_online_price == 0 or baseline_land_price == 0:
            print(f"Warning: Invalid baseline prices for {sku}, skipping")
            continue
        
        baseline_demand = df_pred[df_pred["SKU"] == sku]["expected_units_sold"].mean()
        
        # Create price ranges (±30% from baseline)
        online_price_range = np.linspace(baseline_online_price * 0.7, baseline_online_price * 1.3, 5)
        land_price_range = np.linspace(baseline_land_price * 0.7, baseline_land_price * 1.3, 5)
        
        # Create heatmap data: demand at different price combinations
        demand_matrix = np.zeros((len(land_price_range), len(online_price_range)))
        
        for i, land_price in enumerate(land_price_range):
            for j, online_price in enumerate(online_price_range):
                # Simple linear approximation: demand scales inversely with online price,
                # slightly inversely with land price (warehouse demand when online prices high)
                price_factor = (baseline_online_price / online_price) * 0.8 + (baseline_land_price / land_price) * 0.2
                demand_matrix[i, j] = baseline_demand * price_factor
                sensitivity_records.append({
                    "SKU": sku,
                    "land_price": land_price,
                    "online_price": online_price,
                    "expected_demand": demand_matrix[i, j],
                })
        
        # Plot heatmap
        im = ax.imshow(demand_matrix, cmap='RdYlGn', aspect='auto', origin='lower')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(online_price_range)))
        ax.set_yticks(np.arange(len(land_price_range)))
        ax.set_xticklabels([f"${p:.0f}" for p in online_price_range], fontsize=9)
        ax.set_yticklabels([f"${p:.0f}" for p in land_price_range], fontsize=9)
        
        # Add text annotations
        for i in range(len(land_price_range)):
            for j in range(len(online_price_range)):
                text = ax.text(j, i, f"{demand_matrix[i, j]:.0f}",
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xlabel("Online Unit Price", fontsize=11, fontweight="bold")
        ax.set_ylabel("Land Price Per Acre", fontsize=11, fontweight="bold")
        ax.set_title(f"{sku}\n(Baseline: ${baseline_online_price:.2f} online, ${baseline_land_price:.0f} land)", 
                    fontsize=12, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Expected Demand (units)", fontsize=9)
    
    plt.suptitle("Demand Sensitivity Analysis: Online Price vs Land Price", 
                fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    output_file = f"{OUTPUT_DIR}/sensitivity_analysis_demand.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sensitivity analysis chart saved to: {output_file}")
    plt.close()
    
    save_debug_table(pd.DataFrame(sensitivity_records), "sensitivity_analysis_data")


def create_scenario_comparison_chart(predictions_file: str, weekly_features: pd.DataFrame):
    """
    Create comparison chart: predicted demand under different price scenarios
    (low prices, baseline, high prices) for each SKU.
    """
    df_pred = pd.read_csv(predictions_file)
    weekly_features_latest = weekly_features.dropna(subset=["avg_online_price", "land_price_per_acre"])
    weekly_features_latest = weekly_features_latest.sort_values("week_start").groupby("SKU").tail(1)
    
    skus = sorted(df_pred["SKU"].unique())
    scenarios = ["Low Prices\n(Online -20%, Land -20%)", 
                 "Baseline", 
                 "High Prices\n(Online +20%, Land +20%)"]
    scenario_multipliers = [1.4, 1.0, 0.7]
    
    # Prepare data
    scenario_rows = []
    for sku in skus:
        baseline_demand = df_pred[df_pred["SKU"] == sku]["expected_units_sold"].mean()
        
        # Check if we have valid price data
        sku_prices = weekly_features[weekly_features["SKU"] == sku].dropna(subset=["avg_online_price"])
        if sku_prices.empty or baseline_demand == 0:
            print(f"Warning: Skipping {sku} (no valid price or demand data)")
            continue
        
        for label, multiplier in zip(scenarios, scenario_multipliers):
            scenario_rows.append({"SKU": sku, "Scenario": label, "ExpectedDemand": baseline_demand * multiplier})
    scenario_df = pd.DataFrame(scenario_rows)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(scenario_df["SKU"].unique()))
    width = 0.25
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green (low prices), Blue (baseline), Red (high prices)
    
    for i, (scenario, multiplier) in enumerate(zip(scenarios, scenario_multipliers)):
        demands = scenario_df[scenario_df["Scenario"] == scenario]["ExpectedDemand"]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, demands, width, label=scenario, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("SKU", fontsize=12, fontweight="bold")
    ax.set_ylabel("Expected Demand (units)", fontsize=12, fontweight="bold")
    ax.set_title("Predicted Demand Under Different Price Scenarios (Next 4 Weeks Average)", 
                fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_df["SKU"].unique(), fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f"{OUTPUT_DIR}/scenario_comparison_demand.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scenario comparison chart saved to: {output_file}")
    plt.close()
    
    save_debug_table(scenario_df.set_index(["SKU", "Scenario"]), "scenario_comparison_data")


def create_forecast_accuracy_degradation_chart(weekly_features: pd.DataFrame, clf, regr, feature_cols: list):
    """
    Show how prediction accuracy degrades as we forecast further into the future.
    Uses recursive predictions (week N uses predictions from week N-1 as lags).
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    
    weekly_features_sorted = weekly_features.sort_values("week_start").copy()
    weekly_features_sorted["week_of_year"] = weekly_features_sorted["week_start"].dt.isocalendar().week.astype(int)
    skus = sorted(weekly_features_sorted["SKU"].unique())
    
    forecast_horizons = [1, 2, 3, 4, 8, 12, 16, 20]
    results_by_horizon = {h: {"mae": [], "r2": []} for h in forecast_horizons}
    
    for sku in skus:
        sku_data = weekly_features_sorted[weekly_features_sorted["SKU"] == sku].copy()
        sku_data["week_of_year"] = sku_data["week_start"].dt.isocalendar().week.astype(int)
        
        if len(sku_data) < 30:
            continue
        
        for horizon in forecast_horizons:
            test_start_idx = len(sku_data) - horizon
            if test_start_idx < 20:
                continue
            
            actual_sales = sku_data.iloc[test_start_idx:]["units_sold"].values
            predicted_sales = []
            
            latest_row = sku_data.iloc[test_start_idx - 1].copy()
            sales_history = sku_data.iloc[max(0, test_start_idx - 4):test_start_idx][["units_sold"]].values.flatten().tolist()
            
            for step in range(len(actual_sales)):
                lag1 = sales_history[-1] if len(sales_history) > 0 else 0.0
                lag2 = sales_history[-2] if len(sales_history) > 1 else 0.0
                lag4 = sales_history[-4] if len(sales_history) > 3 else 0.0
                week_idx = test_start_idx + step
                week_of_year = sku_data.iloc[week_idx]["week_of_year"]
                rolling4 = np.mean(sales_history[-4:]) if len(sales_history) else latest_row.get("rolling_4w_avg_sales", 0.0)
                rolling8 = np.mean(sales_history[-8:]) if len(sales_history) else latest_row.get("rolling_8w_avg_sales", 0.0)
                sku_avg = latest_row.get("sku_avg_sales", 0.0)
                sku_code = latest_row.get("sku_encoded", 0.0)

                X_row = pd.DataFrame([{
                    "on_hand_start": latest_row.get("on_hand_start", 0.0),
                    "lag1_sales": lag1,
                    "lag2_sales": lag2,
                    "lag4_sales": lag4,
                    "week_of_year": week_of_year,
                    "avg_online_price": latest_row.get("avg_online_price", 0.0),
                    "land_price_per_acre": latest_row.get("land_price_per_acre", 0.0),
                    "rolling_4w_avg_sales": rolling4,
                    "rolling_8w_avg_sales": rolling8,
                    "sku_avg_sales": sku_avg,
                    "sku_encoded": sku_code,
                }])
                
                X_row_filled = apply_feature_filling(X_row, feature_cols)
                prob = clf.predict_proba(X_row_filled)[:, 1][0]
                log_units = regr.predict(X_row_filled)[0]
                pred_units = np.expm1(log_units)
                expected = prob * pred_units
                
                predicted_sales.append(expected)
                sales_history.append(expected)
            
            mae = mean_absolute_error(actual_sales, predicted_sales)
            r2 = r2_score(actual_sales, predicted_sales) if len(set(actual_sales)) > 1 else 0.0
            
            results_by_horizon[horizon]["mae"].append(mae)
            results_by_horizon[horizon]["r2"].append(r2)
    
    # Compute averages
    avg_mae = [np.mean(results_by_horizon[h]["mae"]) if results_by_horizon[h]["mae"] else 0 for h in forecast_horizons]
    avg_r2 = [np.mean(results_by_horizon[h]["r2"]) if results_by_horizon[h]["r2"] else 0 for h in forecast_horizons]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(forecast_horizons, avg_mae, marker='o', linewidth=2.5, markersize=8, color='#e74c3c')
    ax1.fill_between(forecast_horizons, avg_mae, alpha=0.3, color='#e74c3c')
    ax1.set_xlabel("Forecast Horizon (weeks ahead)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (units)", fontsize=12, fontweight="bold")
    ax1.set_title("Forecast Accuracy Degradation: MAE", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(forecast_horizons, avg_r2, marker='s', linewidth=2.5, markersize=8, color='#3498db')
    ax2.fill_between(forecast_horizons, avg_r2, alpha=0.3, color='#3498db')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline (no skill)')
    ax2.set_xlabel("Forecast Horizon (weeks ahead)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("R² Score", fontsize=12, fontweight="bold")
    ax2.set_title("Forecast Accuracy Degradation: R²", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle("Model Performance Degradation Over Forecast Horizon", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    
    output_file = f"{OUTPUT_DIR}/forecast_accuracy_degradation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Forecast accuracy degradation chart saved to: {output_file}")
    plt.close()
    
    degradation_df = pd.DataFrame({
        "horizon_weeks": forecast_horizons,
        "avg_mae": avg_mae,
        "avg_r2": avg_r2,
    })
    save_debug_table(degradation_df.set_index("horizon_weeks"), "forecast_accuracy_degradation_data")


def create_sku_horizon_heatmap(weekly_features: pd.DataFrame, clf, regr, feature_cols: list):
    """
    Create heatmap: MAE for each SKU at different forecast horizons.
    Shows which SKUs are easier/harder to forecast at different time scales.
    """
    from sklearn.metrics import mean_absolute_error
    
    weekly_features_sorted = weekly_features.sort_values("week_start").copy()
    weekly_features_sorted["week_of_year"] = weekly_features_sorted["week_start"].dt.isocalendar().week.astype(int)
    skus = sorted(weekly_features_sorted["SKU"].unique())
    
    forecast_horizons = [1, 2, 3, 4, 8, 12]
    mae_matrix = np.zeros((len(skus), len(forecast_horizons)))
    
    for sku_idx, sku in enumerate(skus):
        sku_data = weekly_features_sorted[weekly_features_sorted["SKU"] == sku].copy()
        sku_data["week_of_year"] = sku_data["week_start"].dt.isocalendar().week.astype(int)
        
        if len(sku_data) < 30:
            mae_matrix[sku_idx, :] = np.nan
            continue
        
        for h_idx, horizon in enumerate(forecast_horizons):
            test_start_idx = len(sku_data) - horizon
            if test_start_idx < 20:
                mae_matrix[sku_idx, h_idx] = np.nan
                continue
            
            actual_sales = sku_data.iloc[test_start_idx:]["units_sold"].values
            predicted_sales = []
            
            latest_row = sku_data.iloc[test_start_idx - 1].copy()
            sales_history = sku_data.iloc[max(0, test_start_idx - 4):test_start_idx][["units_sold"]].values.flatten().tolist()
            
            for step in range(len(actual_sales)):
                lag1 = sales_history[-1] if len(sales_history) > 0 else 0.0
                lag2 = sales_history[-2] if len(sales_history) > 1 else 0.0
                lag4 = sales_history[-4] if len(sales_history) > 3 else 0.0
                week_idx = test_start_idx + step
                week_of_year = sku_data.iloc[week_idx]["week_of_year"]
                rolling4 = np.mean(sales_history[-4:]) if len(sales_history) else latest_row.get("rolling_4w_avg_sales", 0.0)
                rolling8 = np.mean(sales_history[-8:]) if len(sales_history) else latest_row.get("rolling_8w_avg_sales", 0.0)
                sku_avg = latest_row.get("sku_avg_sales", 0.0)
                sku_code = latest_row.get("sku_encoded", 0.0)

                X_row = pd.DataFrame([{
                    "on_hand_start": latest_row.get("on_hand_start", 0.0),
                    "lag1_sales": lag1,
                    "lag2_sales": lag2,
                    "lag4_sales": lag4,
                    "week_of_year": week_of_year,
                    "avg_online_price": latest_row.get("avg_online_price", 0.0),
                    "land_price_per_acre": latest_row.get("land_price_per_acre", 0.0),
                    "rolling_4w_avg_sales": rolling4,
                    "rolling_8w_avg_sales": rolling8,
                    "sku_avg_sales": sku_avg,
                    "sku_encoded": sku_code,
                }])
                
                X_row_filled = apply_feature_filling(X_row, feature_cols)
                prob = clf.predict_proba(X_row_filled)[:, 1][0]
                log_units = regr.predict(X_row_filled)[0]
                pred_units = np.expm1(log_units)
                expected = prob * pred_units
                
                predicted_sales.append(expected)
                sales_history.append(expected)
            
            mae = mean_absolute_error(actual_sales, predicted_sales)
            mae_matrix[sku_idx, h_idx] = mae
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(forecast_horizons)))
    ax.set_yticks(np.arange(len(skus)))
    ax.set_xticklabels([f"Week {h}" for h in forecast_horizons], fontsize=11)
    ax.set_yticklabels(skus, fontsize=11)
    
    for i in range(len(skus)):
        for j in range(len(forecast_horizons)):
            if not np.isnan(mae_matrix[i, j]):
                text = ax.text(j, i, f'{mae_matrix[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Forecast Horizon", fontsize=12, fontweight="bold")
    ax.set_ylabel("SKU", fontsize=12, fontweight="bold")
    ax.set_title("MAE by SKU and Forecast Horizon (Lower is Better)", fontsize=13, fontweight="bold")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Absolute Error (units)", fontsize=11)
    
    plt.tight_layout()
    
    output_file = f"{OUTPUT_DIR}/sku_horizon_accuracy_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"SKU-horizon accuracy heatmap saved to: {output_file}")
    plt.close()
    
    heatmap_df = pd.DataFrame(mae_matrix, index=skus, columns=[f"week_{h}" for h in forecast_horizons])
    save_debug_table(heatmap_df, "sku_horizon_heatmap_data")


def apply_feature_filling(df: pd.DataFrame, feature_cols):
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feature_cols].copy()
    for col in ["lag1_sales", "lag2_sales", "lag4_sales"]:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)
    if "on_hand_start" in X.columns:
        X["on_hand_start"] = X["on_hand_start"].fillna(0.0)
    if "avg_online_price" in X.columns:
        X["avg_online_price"] = X["avg_online_price"].fillna(0.0)
    if "land_price_per_acre" in X.columns:
        X["land_price_per_acre"] = X["land_price_per_acre"].fillna(0.0)
    for col in ["rolling_4w_avg_sales", "rolling_8w_avg_sales", "sku_avg_sales", "sku_encoded"]:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)
    return X


def main():
    print("Loading and processing data...")
    weekly_sales = load_and_process_data(INPUT_CSV_FILE)
    
    print(f"Total weeks in dataset: {weekly_sales['YEAR_WEEK'].nunique()}")
    print(f"SKUs found: {', '.join(sorted(weekly_sales['SKU'].unique()))}")
    
    print("\nGenerating all years sales trends plot...")
    plot_all_years_trends(weekly_sales)
    
    print("Generating random year sales trends plot...")
    plot_sales_trends_random_year(weekly_sales)
    
    print("\nGenerating prediction vs actual comparison...")
    weekly_features = load_weekly_features(WEEKLY_FEATURES_FILE)
    actual_probs = compute_actual_sale_probabilities(weekly_features)
    create_prediction_comparison_table(PREDICTIONS_FILE, actual_probs)
    
    print("\nGenerating sensitivity analysis chart...")
    create_sensitivity_analysis_chart(PREDICTIONS_FILE, weekly_features)
    
    print("Generating scenario comparison chart...")
    create_scenario_comparison_chart(PREDICTIONS_FILE, weekly_features)
    
    print("\nGenerating forecast accuracy degradation charts...")
    from joblib import load
    clf = load("mldata/demand_nonzero_classifier.joblib")
    regr = load("mldata/demand_positive_regressor.joblib")
    with open("mldata/demand_model_metadata.json") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    
    create_forecast_accuracy_degradation_chart(weekly_features, clf, regr, feature_cols)
    create_sku_horizon_heatmap(weekly_features, clf, regr, feature_cols)
    
    print("\nAll charts generated and saved to graphs/ folder")


if __name__ == "__main__":
    import json
    main()
