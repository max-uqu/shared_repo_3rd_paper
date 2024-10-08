import pandas as pd
from scipy import stats
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler, PowerTransformer

from src.plot import plot_histogram


def find_and_fill_missing_dates(data: pd.DataFrame, ticker_col: str):
    ticker = data.name
    freq = "min"

    resampled_data = data.resample(freq).asfreq()

    missing_ticks = resampled_data.isnull().all(axis=1)
    resampled_data["missing_ticks"] = missing_ticks.astype(int)

    number_of_missing_ticks = int(resampled_data["missing_ticks"].sum())
    print(f"Number of missing ticks for {ticker}: {number_of_missing_ticks}")

    missing_rows = missing_ticks[missing_ticks == True]
    if not missing_rows.empty:
        resampled_data.ffill(inplace=True)

    # resampled_data[ticker_col] = ticker
    return resampled_data


def fill_missing_data(data: pd.DataFrame):
    missing_data = data.isnull().any(axis=1)
    data["missing_partially"] = missing_data.astype(int)

    if int(data["missing_partially"].sum()) > 0:
        data.ffill(inplace=True)

    return data


def resample_data(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    data.resample(freq).agg(
        {
            "symbol": "first",
            "close_time": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "taker_buy_quote_asset_volume": "sum",
            "taker_buy_base_asset_volume": "sum",
            "num_trades": "sum",
            "quote_asset_volume": "sum",
            "base_asset_volume": "sum",
        }
    )

    return data


def preprocess_data(
    data: pd.DataFrame, indices: list[str], datetime_col: str, ticker_col: str
):
    data = data[data["is_closed"] == True]
    data.drop(columns=["is_closed"], inplace=True)
    data.drop_duplicates(keep="first", inplace=True)
    data = data[data["open_time"] < "2024-07-15"]

    print(f"Number of rows before removing the duplicates: {len(data)}")
    data = data.drop_duplicates(subset=indices, keep="first")
    print(f"Number of rows after removing the duplicates: {len(data)}")

    data.set_index([datetime_col], inplace=True)

    data = data.groupby(ticker_col, group_keys=False).apply(
        lambda group: find_and_fill_missing_dates(data=group, ticker_col=ticker_col)
    )

    data = data.groupby(ticker_col, group_keys=False).apply(fill_missing_data)
    data.reset_index(inplace=True)
    data = data.sort_values(by=indices)

    print(f"Number of rows after filling the missing dates: {len(data)}")

    return data


def detect_outliers(data: pd.DataFrame):
    # Statistical method: Z-score
    targeted_columns = data["close"]
    threshold: float = 3
    zscore = np.abs(stats.zscore(targeted_columns))
    zscore_outliers = np.where(zscore > threshold, 1, 0)

    # IQR method: Interquartile range
    lower_range = 0.2
    q1 = targeted_columns.quantile(lower_range)
    q3 = targeted_columns.quantile(1 - lower_range)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = np.where(
        (targeted_columns < lower_bound) | (targeted_columns > upper_bound), 1, 0
    )

    data["outliers"] = zscore_outliers | iqr_outliers

    return data


def find_best_transformation_function(data: pd.DataFrame, column_to_transform: str):
    print(f"Column to transform: {column_to_transform}")
    print("---------------------------------------------------------------------")
    # Check for NaN values in the column
    if data[column_to_transform].isnull().any():
        raise ValueError(f"The column '{column_to_transform}' contains NaN values.")

    # Initialize dictionary to hold transformed data and their statistics
    transformations = {}
    stats_summary = []

    # Original data
    original_data = data[column_to_transform]
    transformations["original"] = original_data

    # Shift data if necessary for positive-only transformations
    min_value = original_data.min()
    shift = 0
    if min_value <= 0:
        shift = -min_value + np.finfo(float).eps
    data_positive = original_data + shift

    # Make sure the data is not infinite
    if not np.isfinite(data_positive).all():
        print(f"{column_to_transform} contains infinite values.")
        return data

    # Apply transformations
    transformations["log"] = np.log(data_positive)
    transformations["sqrt"] = np.sqrt(data_positive)
    # transformations["boxcox"], fitted_lambda = stats.boxcox(data_positive)
    transformations["log1p"] = np.log1p(original_data)
    scaler = StandardScaler()
    transformations["standard"] = scaler.fit_transform(
        original_data.values.reshape(-1, 1)
    ).flatten()
    transformations["power"] = (
        PowerTransformer(method="yeo-johnson")
        .fit_transform(original_data.values.reshape(-1, 1))
        .flatten()
    )

    # Compute statistics and add transformed columns to DataFrame
    transformed_data = pd.DataFrame()
    for name, transformed in transformations.items():
        skewness = pd.Series(transformed).skew()
        kurtosis = pd.Series(transformed).kurtosis()
        stats_summary.append(
            {"Transformation": name, "Skewness": skewness, "Kurtosis": kurtosis}
        )
        # if name != "original":
        col_name = f"{name}_{column_to_transform}"
        transformed_data[col_name] = transformed

    # Create a DataFrame of statistics
    stats_df = pd.DataFrame(stats_summary)
    print(stats_df)

    # Choose the transformation with skewness closest to zero
    stats_df["Skewness_Abs"] = stats_df["Skewness"].abs()
    best_transformation = stats_df.loc[
        stats_df["Skewness_Abs"].idxmin(), "Transformation"
    ]

    # Optionally, consider kurtosis or combine metrics
    # stats_df['Combined_Metric'] = stats_df['Skewness'].abs() + stats_df['Kurtosis'].abs()
    # best_transformation = stats_df.loc[stats_df['Combined_Metric'].idxmin(), 'Transformation']
    print(f"Best transformation based on skewness: {best_transformation}")
    print()
    print("---------------------------------------------------------------------")

    # Use the best transformation for further analysis
    best_transformed_column = f"{best_transformation}_{column_to_transform}"
    final_transformed_col = f"t_{column_to_transform}"
    data[final_transformed_col] = transformed_data[best_transformed_column]

    # Plot histograms for the best transformation (assuming plot_histogram is defined)
    plot_histogram(
        data=data,
        main_column=column_to_transform,
        transformed_column=final_transformed_col,
    )

    return data


def apply_transformation(
    data: pd.DataFrame,
    column_to_transform: str,
    transformation: str,
    shift: Optional[float] = None,
    standard_scaler: Optional[StandardScaler] = None,
) -> pd.DataFrame:
    # Check if the column exists in the DataFrame
    if column_to_transform not in data.columns:
        raise ValueError(
            f"The column '{column_to_transform}' does not exist in the DataFrame."
        )

    # Check for NaN values in the column
    if data[column_to_transform].isnull().any():
        raise ValueError(f"The column '{column_to_transform}' contains NaN values.")

    # Get the original data
    original_data = data[column_to_transform]
    transformed_data = None

    # Determine if shifting is necessary
    requires_positive = transformation in ["log", "sqrt", "boxcox"]
    if requires_positive:
        min_value = original_data.min()
        if shift is None:
            shift = -min_value + np.finfo(float).eps if min_value <= 0 else 0
        data_to_transform = original_data + shift
    else:
        data_to_transform = original_data

    if not np.isfinite(data_to_transform).all():
        infinite_mask = np.isinf(data_to_transform)
        print(f"{column_to_transform} contains {len(infinite_mask)} infinite values.")
        data[column_to_transform].replace([np.inf, -np.inf], np.nan, inplace=True)
        mean_value = data[column_to_transform].mean()
        data[column_to_transform].fillna(mean_value, inplace=True)

    # Apply the specified transformation
    if transformation == "log":
        transformed_data = np.log(data_to_transform)
    elif transformation == "sqrt":
        transformed_data = np.sqrt(data_to_transform)
    elif transformation == "boxcox":
        transformed_data, _ = stats.boxcox(data_to_transform)
    elif transformation == "log1p":
        transformed_data = np.log1p(original_data)
    elif transformation == "standard":
        if standard_scaler is None:
            standard_scaler = StandardScaler()
        transformed_data = standard_scaler.fit_transform(
            original_data.values.reshape(-1, 1)
        ).flatten()
    elif transformation == "power":
        transformer = PowerTransformer(method="yeo-johnson")
        transformed_data = transformer.fit_transform(
            original_data.values.reshape(-1, 1)
        ).flatten()
    else:
        raise ValueError(f"Transformation '{transformation}' is not supported.")

    # Add the transformed data to the DataFrame
    transformed_column_name = f"t_{column_to_transform}"
    data[transformed_column_name] = transformed_data

    return data


def transform_dataset(
    data: pd.DataFrame,
    column_to_transform: list[str],
    transformation: str,
    plot_hist: bool = False,
):
    for col in column_to_transform:
        data = apply_transformation(
            data.copy(), column_to_transform=col, transformation=transformation
        )

    if plot_hist:
        # Plot histograms for the transformed columns
        for col in column_to_transform:
            plot_histogram(
                data=data,
                main_column=col,
                transformed_column=f"t_{col}",
            )

    data.drop(columns=column_to_transform, inplace=True)
    data.rename(columns={f"t_{col}": col for col in column_to_transform}, inplace=True)

    return data
