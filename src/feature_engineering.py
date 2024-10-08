import pandas as pd
import numpy as np
from src.plot import plot_correlation_matrix
from src.utils import make_dir_if_not_exists
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_overlap(today_high, today_low, yesterday_high, yesterday_low):
    gap = 0
    overlap = 0

    if today_high < yesterday_low:
        gap = yesterday_low - today_high
    elif today_low > yesterday_high:
        gap = today_low - yesterday_high
    elif today_high >= yesterday_high and today_low <= yesterday_low:
        overlap = yesterday_high - yesterday_low
    elif today_high > yesterday_high and today_low > yesterday_low:
        overlap = yesterday_high - today_low
    elif today_high < yesterday_high and today_low < yesterday_low:
        overlap = today_high - yesterday_low
    elif yesterday_high >= today_high and yesterday_low <= today_low:
        overlap = today_high - today_low

    return gap, overlap


def calculate_body_position(row: pd.Series) -> pd.Series:
    try:
        today_open = row["open"]
        today_close = row["close"]

        today_low = min(today_open, today_close)
        today_high = max(today_open, today_close)

        yesterday_open = row["yesterday_open"]
        yesterday_close = row["yesterday_close"]

        yesterday_low = min(yesterday_open, yesterday_close)
        yesterday_high = max(yesterday_open, yesterday_close)

        gap, overlap = calculate_overlap(
            today_high=today_high,
            today_low=today_low,
            yesterday_high=yesterday_high,
            yesterday_low=yesterday_low,
        )
    except Exception as e:
        print(e)

    return gap, overlap


def calculate_range_position(row: pd.Series) -> pd.Series:
    today_low = row["low"]
    today_high = row["high"]

    yesterday_low = row["yesterday_low"]
    yesterday_high = row["yesterday_high"]

    gap, overlap = calculate_overlap(
        today_high=today_high,
        today_low=today_low,
        yesterday_high=yesterday_high,
        yesterday_low=yesterday_low,
    )

    return gap, overlap


def add_features(data: pd.DataFrame):
    data["return"] = data["close"].pct_change()
    data["volatility"] = data["return"].rolling(window=10).std()
    data["momentum"] = data["return"].rolling(window=10).mean()
    data["volume_change"] = data["base_asset_volume"].pct_change()

    data["average_base_asset_per_trade"] = (
        data["base_asset_volume"] / data["num_trades"]
    )

    data["bullish_day_flag"] = (data["close"] > data["open"]).astype(int)

    # gaps
    data["yesterday_open"] = data["open"].shift(1)
    data["yesterday_close"] = data["close"].shift(1)
    # examine the body overlap for sentiment
    data[["body_gap_value", "body_overlap_value"]] = data.apply(
        lambda row: pd.Series(calculate_body_position(row=row)), axis=1
    )
    # examine the tails for volatility and extremes
    data["yesterday_low"] = data["low"].shift(1)
    data["yesterday_high"] = data["high"].shift(1)
    data[["range_gap_value", "range_overlap_value"]] = data.apply(
        lambda row: pd.Series(calculate_range_position(row=row)), axis=1
    )

    candle_body = (data["close"] - data["open"]).abs()
    candle_upper_shadow = data["high"] - data[["open", "close"]].max(axis=1)

    data["candle_range"] = data["high"] - data["low"]
    data["candle_body_over_range"] = candle_body / data["candle_range"]
    data["candle_upper_shadow_over_range"] = candle_upper_shadow / data["candle_range"]

    # "candle_lower_shadow_over_range" has a high correlation with the "candle_upper_shadow_over_range"
    # candle_lower_shadow = data[["open", "close"]].min(axis=1) - data["low"]
    # data["candle_lower_shadow_over_range"] = candle_lower_shadow / data["candle_range"]

    columns_to_drop = [
        "yesterday_open",
        "yesterday_close",
        "yesterday_low",
        "yesterday_high",
    ]
    data.drop(columns=columns_to_drop, inplace=True)
    data.dropna(inplace=True)

    return data


def check_correction(dataset: pd.DataFrame, plot: bool = False):
    path = "data/outputs"
    make_dir_if_not_exists(path)

    correlation_matrix = dataset.corr()
    dataset.to_csv(f"{path}/corr_df.csv", index=False)
    correlation_matrix.to_csv(f"{path}/correlation_matrix.csv", index=True)

    if plot == True:
        plot_correlation_matrix(correlation_matrix)


def vif_calculation(dataset: pd.DataFrame):
    path = "data/outputs"
    make_dir_if_not_exists(path)

    X = dataset.copy()
    features_count = X.shape[1]

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Features"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(features_count)
    ]
    correlation_matrix = X.corr()
    vif_data["Highest Correlation"] = vif_data["Features"].apply(
        lambda x: correlation_matrix[x].sort_values(ascending=False).index[1]
    )
    vif_data.sort_values(by="VIF", ascending=False, inplace=True)

    number_of_rows = 7
    print(f"Top {number_of_rows} features with the most VIF")
    print(vif_data.head(number_of_rows))

    vif_data.to_csv(f"{path}/vif_data.csv", index=False)
