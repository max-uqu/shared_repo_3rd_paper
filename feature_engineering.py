import pandas as pd
import numpy as np


def add_features(data: pd.DataFrame):
    data["return"] = data["close"].pct_change()

    data["log_return"] = np.log(data["return"])
    if not np.isfinite(data["log_return"]).all():
        infinite_mask = np.isinf(data["log_return"])
        print(f"log return contains {len(infinite_mask)} infinite values.")
        data["log_return"].replace([np.inf, -np.inf], np.nan, inplace=True)
        mean_value = data["log_return"].mean()
        data["log_return"].fillna(mean_value, inplace=True)

    data["average_base_asset_per_trade"] = (
        data["base_asset_volume"] / data["num_trades"]
    )
    data["average_quote_asset_per_trade"] = (
        data["quote_asset_volume"] / data["num_trades"]
    )
    data["average_price"] = data["quote_asset_volume"] / data["base_asset_volume"]
    data["takers_average_price"] = (
        data["taker_buy_quote_asset_volume"] / data["taker_buy_base_asset_volume"]
    )

    data.dropna(inplace=True)

    return data
