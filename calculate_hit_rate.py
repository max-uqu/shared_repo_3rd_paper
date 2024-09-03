import timeit
import pandas as pd
import config_file
from itertools import combinations
import numpy as np
import os
from scipy.signal import argrelextrema

pd.options.mode.chained_assignment = None

# tokens_of_interest = ('uniswap', 'aave')

tokens_of_interest = ('chainlink', 'uniswap', 'aave', 'sushi_swap', 'the_graph', 'basic_attention_token')

indicators = ('close', 'open_eth', 'close_eth', 'volume_eth_in_usd', 'open_btc', 'close_btc', 'volume_btc_in_usd', 
              'unrealised_profit_v', 'unrealised_losses_v', 'unrealised_profit_a', 'unrealised_losses_a',
              'amount_usd', 'to')

shift_parameter = -1

def main():
    for token in tokens_of_interest:
        start = timeit.default_timer()
        token_name = str(token)
        # 1 PRE-PROCESSING: Combine all the single data to one dataframe
        df_price = _1_prices()
        df_onchain = _2_onchain(token_name)
        df_merged = _3_merge_dfs(df_price, df_onchain)
        df_merged = _4_delta_calculation(df_merged)
        # 2 HIT RATE: for single indicators


        end = timeit.default_timer()
        print("Time Taken: ", round(end - start, 4))


def _1_prices():
    name_file_price_btc = config_file.folder_database + '1_Bitstamp_BTCUSD_1h.csv'
    df_btc = pd.read_csv(name_file_price_btc, sep=',')
    df_btc['date'] = pd.to_datetime(df_btc['date'])
    df_btc.rename(columns={
        'open': 'open_btc',
        'high': 'high_btc',
        'low': 'low_btc',
        'close': 'close_btc',
        'Volume BTC': 'volume_btc',
        'Volume USD': 'volume_btc_in_usd',
    }, inplace=True)
    df_btc = df_btc.drop('symbol', axis=1)
    df_btc = df_btc.drop('unix', axis=1)

    name_file_price_eth = config_file.folder_database + '1_Bitstamp_ETHUSD_1h.csv'
    df_eth = pd.read_csv(name_file_price_eth, sep=',')
    df_eth['date'] = pd.to_datetime(df_eth['date'])
    df_eth.rename(columns={
        'open': 'open_eth',
        'high': 'high_eth',
        'low': 'low_eth',
        'close': 'close_eth',
        'Volume ETH': 'volume_eth',
        'Volume USD': 'volume_eth_in_usd',
    }, inplace=True)
    df_eth = df_eth.drop('symbol', axis=1)
    df_eth = df_eth.drop('unix', axis=1)

    df_price = pd.merge(df_eth, df_btc, how='inner',
                        left_on='date', right_on='date')
    df_price = _1_1_price_calculations(df_price)
    df_price['date'] = pd.to_datetime(df_price['date'])
    return df_price


def _1_1_price_calculations(df_price):
    # indicators to calculate for the price
    # RSI
    # LOCAL HIGHS and LOWS
    # Convergence/Divergence in Volume (Dominance)
    return df_price


def _2_onchain(token_name):
    # NEW CODE
    name_file_onchain_data = 'df_onchain_complete_' + token_name + '.csv'
    path_onchain_data = 'C:/Users/maxim/python/database/3rd_paper/' + token_name + '/'
    csv_file_onchain_data = path_onchain_data + name_file_onchain_data
    df_onchain = pd.read_csv(csv_file_onchain_data, sep=',')
    return df_onchain


def _3_merge_dfs(df_price, df_onchain):
    # Convert columns to datetime
    end_date = pd.to_datetime('2024-08-25 00:00:00')
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_onchain['timestamp'] = pd.to_datetime(df_onchain['timestamp'])
     # Find the later start date between the two dataframes
    start_date_price = df_price['date'].min()
    start_date_onchain = df_onchain['timestamp'].min()
    start_date = max(start_date_price, start_date_onchain)
    # Merge dataframes on the common date/timestamp
    df_merged = pd.merge(df_onchain, df_price,
                         left_on='timestamp', right_on='date', how='left')
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
    df_merged.drop(columns=['date'], inplace=True)
    df_merged = df_merged[(df_merged['timestamp'] >= start_date) & 
                          (df_merged['timestamp'] <= end_date)]
    return df_merged


def _4_delta_calculation(df_merged):
    for element in indicators:
        # Calculate the delta between each data point
        delta = df_merged[element].diff()
        df_merged[element + '_delta'] = delta
        # Calculate the trend between each data point
        trend = delta.apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'no change'))
        df_merged[element + '_trend'] = trend
    return df_merged