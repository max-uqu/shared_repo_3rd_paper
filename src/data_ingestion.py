import json
import pickle
import os

import dask.dataframe as dd
import pandas as pd


def format_columns(
    data: pd.DataFrame, datetime_cols: list[str] = [], numeric_cols: list[str] = []
):
    for col in datetime_cols:
        data[col] = pd.to_datetime(data[col], unit="ms")

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def read_btc_olhcv():
    pickle_file_path = "data/btc_olhcv.pkl"

    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, "rb") as file:
            df = pickle.load(file)
            print("Data loaded successfully")
            return df

    df = pd.read_csv("data/btc_olhcv.csv")

    df = df[
        [
            "symbol",
            "time",
            # "close_price", # is the same as close
            # "high_price", # is the same as high
            # "low_price", # is the same as low
            # "open_price", # is the same as open
            # "volume", # is the same as base_asset_volume
            # "number_trades", # is the same as num_trades
            "candle",
            # "candle_close", # is the same as is_closed
        ]
    ]

    df["json_candle"] = df["candle"].apply(json.loads)
    # df["datetime"] = pd.to_datetime(df["time"]).dt.floor("s")
    json_candle_df = pd.json_normalize(df["json_candle"])
    print("Data normalized successfully")

    df.drop(columns=["candle", "json_candle", "time"], inplace=True)
    json_candle_df.rename(
        columns={
            "B": "ignore",
            "L": "last_trade_id",
            "Q": "taker_buy_quote_asset_volume",
            "T": "close_time",
            "V": "taker_buy_base_asset_volume",
            "c": "close",
            "f": "first_trade_id",
            "h": "high",
            "i": "interval",
            "l": "low",
            "n": "num_trades",
            "o": "open",
            "q": "quote_asset_volume",
            "s": "symbol",
            "t": "open_time",
            "v": "base_asset_volume",
            "x": "is_closed",
        },
        inplace=True,
    )
    json_candle_df.drop(
        columns=["ignore", "last_trade_id", "first_trade_id", "interval", "symbol"],
        inplace=True,
    )

    datetime_cols = ["open_time", "close_time"]
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "taker_buy_quote_asset_volume",
        "taker_buy_base_asset_volume",
        "base_asset_volume",
        "quote_asset_volume",
    ]
    json_candle_df = format_columns(json_candle_df.copy(), datetime_cols, numeric_cols)

    df = pd.concat([df, json_candle_df], axis=1)

    with open(pickle_file_path, "wb") as file:
        pickle.dump(df, file)
        print("Data saved successfully")

    return df


def read_massive_data():
    df2 = dd.read_csv("data/btc_orderbook.csv", blocksize="100MB")

    # Get the number of partitions
    npartitions = df2.npartitions
    print(f"Number of partitions: {npartitions}")

    # Compute the length of the first partition
    first_partition_len = len(df2.partitions[0])
    print(df2.partitions[0].info())

    # Estimate total number of rows
    estimated_total_rows = first_partition_len * npartitions
    print(f"Estimated total number of rows: {estimated_total_rows}")
