import pandas as pd
from scipy.signal import find_peaks
import numpy as np

from src.data_ingestion import read_btc_olhcv
from src.data_preprocessing import (
    preprocess_data,
    resample_data,
    detect_outliers,
    transform_dataset,
    find_best_transformation_function,
)
from src.plot import visual_inspection, plot_clustered_data
from src.feature_engineering import add_features
from src.kmean_clustering import cluster_assets
from src.feature_engineering import check_correction, vif_calculation


def get_extrema(data: pd.DataFrame, distance: int = 5):
    # Find peaks (local maxima) in the closing price
    peaks, _ = find_peaks(data["close"], distance=distance)

    # Find valleys (local minima) in the closing price
    valleys, _ = find_peaks(-data["close"], distance=distance)

    # Combine peaks and valleys
    # extrema = np.sort(np.concatenate([peaks, valleys]))

    return peaks, valleys


def get_extrema_previous_candles(
    dataset: pd.DataFrame,
    indices: list[int],
    extrema_type: str,
    n_prev_candles: int = 10,
):
    candles_arr = []
    for idx in indices:
        if idx < n_prev_candles:
            continue
        data = dataset.iloc[idx - n_prev_candles : idx].copy()
        data.loc[:, "extrema_idx"] = idx
        # data.loc[:, "extrema_type"] = extrema_type
        candles_arr.append(data)

    candles_df = pd.concat(candles_arr, ignore_index=True)

    return candles_df


def run_data_pipeline():
    df = read_btc_olhcv()

    # The dataset has minute frequency
    dataset = preprocess_data(
        data=df.copy(),
        indices=["symbol", "open_time"],
        datetime_col="open_time",
        ticker_col="symbol",
    )

    # base_asset_volume is the same as volume
    # visual_inspection(
    #     data=dataset.copy(),
    #     column_to_plot="close",
    #     index_col="open_time",
    #     title="BTC/USDT closing price",
    #     comparison_col="base_asset_volume",
    # )

    # dataset = detect_outliers(dataset.copy())

    dataset = add_features(data=dataset)

    # columns for removing has been chosen based on VIF calculation
    columns_to_drop = [
        "missing_partially",
        "missing_ticks",
        "close_time",
        "base_asset_volume",
        "quote_asset_volume",
        "taker_buy_quote_asset_volume",
        "taker_buy_base_asset_volume",
        "open",
        "high",
        "low",
    ]
    dataset.drop(columns=columns_to_drop, inplace=True)

    index = ["symbol", "open_time"]
    dataset.set_index(index, inplace=True)

    # Evaludate the best transformation for each column
    # for col in columns_to_transform:
    #     find_best_transformation_function(data=dataset.copy(), column_to_transform=col)

    dataset = transform_dataset(
        data=dataset.copy(),
        column_to_transform=list(dataset.columns),
        transformation="power",
        plot_hist=False,
    )

    check_correction(dataset=dataset.copy(), plot=False)
    vif_calculation(dataset=dataset.copy())

    # Resample the data to hourly frequency
    # print(f"Number of rows before resampling: {len(dataset)}")
    # dataset = resample_data(data=dataset.copy(), freq="h")
    # print(f"Number of rows after resampling: {len(dataset)}")

    peaks, valleys = get_extrema(data=dataset.copy(), distance=5)

    n_previous_candles = 10
    peaks_df = get_extrema_previous_candles(
        dataset=dataset.copy(),
        indices=peaks,
        n_prev_candles=n_previous_candles,
        extrema_type="peak",
    )
    print(f"Number of peaks dataset: {len(peaks_df)}")

    # valleys_df = get_extrema_previous_candles(
    #     dataset=dataset,
    #     indices=valleys,
    #     n_prev_candles=n_previous_candles,
    #     extrema_type="valley",
    # )
    # print(f"Number of valleys dataset: {len(valleys_df)}")

    peaks_df.set_index(["extrema_idx"], append=True, inplace=True)
    peaks_df.sort_index(inplace=True)
    peaks_df["order"] = peaks_df.groupby(level="extrema_idx").cumcount() + 1

    peaks_df.drop(columns=["close"], inplace=True)

    peaks_df = cluster_assets(dataset=peaks_df.copy())

    plot_clustered_data(peaks_df)
