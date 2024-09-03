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
        df_merged = _5_single_indicator_correlation_trend(df_merged, token_name)
        df_merged = _5_single_indicator_correlation_big_delta(df_merged, token_name)
        df_merged = _5_single_indicator_correlation_trend_delayed(df_merged, token_name)
        df_merged = _5_single_indicator_correlation_big_delta_delayed(df_merged, token_name)
        # 3 HIT RATE: for combination indicators
        df_merged = _6_multiple_indicator_correlation_trend(df_merged, token_name)
        df_merged = _6_multiple_indicator_correlation_big_delta(df_merged, token_name)
        df_merged = _6_multiple_indicator_correlation_trend_delayed(df_merged, token_name)
        df_merged = _6_multiple_indicator_correlation_big_delta_delayed(df_merged, token_name)

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


# def _4_delta_calculation(df_input):
#     order = 10
#     for element in indicators:
#         # 1.) start calculation for high and lows
#         data = df_input[element].dropna()
#         local_maxima_indices = argrelextrema(data.values, np.greater, order=order)[0]
#         local_minima_indices = argrelextrema(data.values, np.less, order=order)[0]
#         df_input['local_max_' + element] = False
#         df_input['local_min_' + element] = False
#         df_input.loc[local_maxima_indices, 'local_max_' + element] = True
#         df_input.loc[local_minima_indices, 'local_min_' + element] = True

#         # 2.) calculate trend between highs & lows
#         df_input[element + '_trend'] = None
#         last_index = None
#         last_type = None
#         last_value = None
#         extremum_indices = np.sort(np.concatenate([local_maxima_indices, local_minima_indices]))

#         for idx in extremum_indices:
#             current_value = data.iloc[idx]
#             current_type = 'max' if df_input.loc[idx, 'local_max_' + element] else 'min'

#             if last_index is not None:
#                 # Determine trend direction
#                 if last_type == current_type:
#                     # Same type of extremum: compare values
#                     if (last_type == 'max' and current_value > last_value) or (last_type == 'min' and current_value < last_value):
#                         trend_direction = 'up' if last_type == 'min' else 'down'
#                     else:
#                         trend_direction = 'down' if last_type == 'min' else 'up'
#                 else:
#                     # Different types: normal trend determination
#                     trend_direction = 'up' if last_type == 'min' else 'down'
#                 df_input.loc[last_index:idx, element + '_trend'] = trend_direction
#             # Update last known values
#             last_index = idx
#             last_type = current_type
#             last_value = current_value
#     # Ensure the trend for the last segment is set based on the last extremum
#     if last_index is not None and extremum_indices.size > 0:
#         final_trend_direction = 'up' if last_type == 'min' else 'down'
#         df_input.loc[last_index:, element + 'trend'] = final_trend_direction

#     # 3.) calculate the delta between the values
#     for element in indicators:
#         delta = df_input[element].diff()
#         df_input[element + '_delta'] = delta
#     return df_input


def _4_delta_calculation(df_merged):
    for element in indicators:
        # Calculate the delta between each data point
        delta = df_merged[element].diff()
        df_merged[element + '_delta'] = delta
        # Calculate the trend between each data point
        trend = delta.apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'no change'))
        df_merged[element + '_trend'] = trend
    return df_merged


def _5_single_indicator_correlation_trend(df_input, token_name):
    number_parameter = 1
    number_of_minimum_parameter = number_parameter
    number_of_maximum_parameter = number_parameter

    results_list = []

    for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
        combs = list(combinations(indicators, r))
        for comb in combs:
            trend_columns = [element + '_trend' for element in comb]
            if all(col in df_input.columns for col in trend_columns):
                
                # Determine when all indicators are 'up' or 'down' at time t
                all_up = df_input[trend_columns].apply(lambda row: all(x == 'up' for x in row), axis=1)
                all_down = df_input[trend_columns].apply(lambda row: all(x == 'down' for x in row), axis=1)
                
                # Compare indicators at time t with close_trend at time t+1
                hits_up = df_input[all_up & (df_input['close_trend'].shift(shift_parameter) == 'up')].shape[0]
                hits_down = df_input[all_down & (df_input['close_trend'].shift(shift_parameter) == 'down')].shape[0]

                # Shifted hits
                hits_shifted_up = {}
                hits_shifted_down = {}
                current_up_hits = df_input[all_up & (df_input['close_trend'].shift(shift_parameter) == 'up')].shape[0]
                hits_shifted_up[f'hits_up_shifted_index_1'] = current_up_hits

                current_down_hits = df_input[all_down & (df_input['close_trend'].shift(shift_parameter) == 'down')].shape[0]
                hits_shifted_down[f'hits_down_shifted_index_1'] = current_down_hits

                results_list.append({
                    'combination': ', '.join(comb),
                    'number_of_all_up_signals': all_up.sum(),
                    'number_of_all_down_signals': all_down.sum(),
                    'hits_up_same_index': hits_up,
                    'hits_down_same_index': hits_down,
                    **hits_shifted_up,
                    **hits_shifted_down,
                    'ratio_convergence_up_shifted_index_1': round((hits_shifted_up['hits_up_shifted_index_1'] / all_up.sum()), 4) if all_up.sum() > 0 else 0,
                    'ratio_divergence_up_shifted_index_1': round((1 - (hits_shifted_up['hits_up_shifted_index_1'] / all_up.sum())), 4) if all_up.sum() > 0 else 0,
                    'ratio_convergence_down_shifted_index_1': round((hits_shifted_down['hits_down_shifted_index_1'] / all_down.sum()), 4) if all_down.sum() > 0 else 0,
                    'ratio_divergence_down_shifted_index_1': round((1 - (hits_shifted_down['hits_down_shifted_index_1'] / all_down.sum())), 4) if all_down.sum() > 0 else 0,
                })
    
    results_df = pd.DataFrame(results_list)
    path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/convergence_divergence/'
    results_df.to_csv(path_for_results + 'single_indicator_' + token_name + '.csv', index=False)
    return df_input


def _5_single_indicator_correlation_big_delta(df_input, token_name):
    percentile_threshold = 95
    percentile_value = np.percentile(abs(df_input['close_delta'].dropna()), percentile_threshold)
    df_input['close_big_move_percentile'] = abs(df_input['close_delta']) > percentile_value
    df_input['close_big_move_positive'] = (df_input['close_delta'] > 0) & df_input['close_big_move_percentile']
    df_input['close_big_move_negative'] = (df_input['close_delta'] < 0) & df_input['close_big_move_percentile']

    for element in indicators:
        delta_column = element + '_delta'
        if delta_column in df_input.columns and not df_input[delta_column].dropna().empty:
            percentile_value = np.percentile(abs(df_input[delta_column].dropna()), percentile_threshold)
            df_input[element + '_big_move_percentile'] = abs(df_input[delta_column]) > percentile_value
            df_input[element + '_big_move_positive'] = (df_input[delta_column] > 0) & df_input[element + '_big_move_percentile']
            df_input[element + '_big_move_negative'] = (df_input[delta_column] < 0) & df_input[element + '_big_move_percentile']
        else:
            df_input[element + '_big_move_percentile'] = False
            df_input[element + '_big_move_positive'] = False
            df_input[element + '_big_move_negative'] = False
    
    number_parameter = 1
    number_of_minimum_parameter = number_parameter
    number_of_maximum_parameter = number_parameter

    results_list = []
    distribution_list = []  # List to store distribution results

    for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
        combs = list(combinations(indicators, r))
        for comb in combs:
            big_delta_columns = [element + '_big_move_percentile' for element in comb]
            positive_columns = [element + '_big_move_positive' for element in comb]
            negative_columns = [element + '_big_move_negative' for element in comb]

            if all(col in df_input.columns for col in big_delta_columns):
                
                big_delta = df_input[big_delta_columns].apply(lambda row: all(x == True for x in row), axis=1)
                big_positive_delta = df_input[positive_columns].apply(lambda row: all(x == True for x in row), axis=1)
                big_negative_delta = df_input[negative_columns].apply(lambda row: all(x == True for x in row), axis=1)
                
                hits_big_delta_same_index = df_input[big_delta & (df_input['close_big_move_percentile'] == True)].shape[0]
                hits_positive_delta_same_index = df_input[big_positive_delta & (df_input['close_big_move_positive'] == True)].shape[0]
                hits_negative_delta_same_index = df_input[big_negative_delta & (df_input['close_big_move_negative'] == True)].shape[0]

                hits_shifted = {}
                hits_positive_shifted = {}
                hits_negative_shifted = {}
                for shift in range(1, 2):
                    shifted_hits = df_input[big_delta & df_input['close_big_move_percentile'].shift(-shift) == True].index
                    shifted_hits_positive = df_input[big_positive_delta & df_input['close_big_move_positive'].shift(-shift) == True].index
                    shifted_hits_negative = df_input[big_negative_delta & df_input['close_big_move_negative'].shift(-shift) == True].index

                    hits_shifted[f'hits_big_delta_shift_index_{shift}'] = len(shifted_hits)
                    hits_positive_shifted[f'hits_positive_delta_shift_index_{shift}'] = len(shifted_hits_positive)
                    hits_negative_shifted[f'hits_negative_delta_shift_index_{shift}'] = len(shifted_hits_negative)

                total_big_delta = big_delta.sum()
                total_positive_delta = big_positive_delta.sum()
                total_negative_delta = big_negative_delta.sum()

                # Distribution of signals by month
                df_input['month'] = df_input['timestamp'].dt.to_period('M')

                positive_distribution = df_input[big_positive_delta].groupby('month').size().rename('positive_delta_count')
                negative_distribution = df_input[big_negative_delta].groupby('month').size().rename('negative_delta_count')

                # Combine the distributions and add to the list
                monthly_distribution = pd.concat([positive_distribution, negative_distribution], axis=1).reset_index()
                monthly_distribution['combination'] = ', '.join(comb)
                distribution_list.append(monthly_distribution)

                # Append the results
                results_list.append({
                    'combination': ', '.join(comb),
                    'total_rows': df_input.shape[0],
                    'number_of_big_delta_signals': total_big_delta,
                    'number_of_positive_delta_signals': total_positive_delta,
                    'number_of_negative_delta_signals': total_negative_delta,
                    'ratio_number_of_signals': round((total_big_delta / df_input.shape[0]) * 100, 4),
                    'hits_big_delta_same_index': hits_big_delta_same_index,
                    'hits_positive_delta_same_index': hits_positive_delta_same_index,
                    'hits_negative_delta_same_index': hits_negative_delta_same_index,
                    **hits_shifted,
                    **hits_positive_shifted,
                    **hits_negative_shifted,
                    'ratio_big_delta_shifted_index_1': round((hits_shifted['hits_big_delta_shift_index_1'] / total_big_delta), 4) if total_big_delta > 0 else 0,
                    'ratio_positive_delta_shifted_index_1': round((hits_positive_shifted['hits_positive_delta_shift_index_1'] / total_positive_delta), 4) if total_positive_delta > 0 else 0,
                    'ratio_negative_delta_shifted_index_1': round((hits_negative_shifted['hits_negative_delta_shift_index_1'] / total_negative_delta), 4) if total_negative_delta > 0 else 0,
                })
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results_list)
    path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/big_delta/'
    results_df.to_csv(path_for_results + 'single_indicator_' + token_name + '.csv', index=False)

    # Convert distribution list to DataFrame and save
    distribution_df = pd.concat(distribution_list, ignore_index=True)
    distribution_df.to_csv(path_for_results + 'distribution_of_big_delta_signals_' + token_name + '.csv', index=False)

    return df_input


def _5_single_indicator_correlation_trend_delayed(df_input, token_name):
    number_parameter = 1
    number_of_minimum_parameter = number_parameter
    number_of_maximum_parameter = number_parameter

    results_list = []
    distribution_list = []  # List to store distribution results

    for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
        combs = list(combinations(indicators, r))
        for comb in combs:
            trend_columns = [element + '_trend' for element in comb]
            if all(col in df_input.columns for col in trend_columns):
                
                # Determine when all indicators are 'up' or 'down' at time t
                all_up = df_input[trend_columns].apply(lambda row: all(x == 'up' for x in row), axis=1)
                all_down = df_input[trend_columns].apply(lambda row: all(x == 'down' for x in row), axis=1)
                
                # Step 1: Check hits at index 0 (same index)
                hits_up_0 = df_input[all_up & (df_input['close_trend'] == 'up')].shape[0]
                hits_down_0 = df_input[all_down & (df_input['close_trend'] == 'down')].shape[0]

                # Mark rows where the hit was incorrect at index 0
                incorrect_up_0 = df_input[all_up & (df_input['close_trend'] != 'up')]
                incorrect_down_0 = df_input[all_down & (df_input['close_trend'] != 'down')]

                # Step 2: Check hits at index +1 for the previously incorrect rows
                hits_up_1 = incorrect_up_0[incorrect_up_0['close_trend'].shift(-1) == 'up'].shape[0]
                hits_down_1 = incorrect_down_0[incorrect_down_0['close_trend'].shift(-1) == 'down'].shape[0]

                # Total hits considering both index 0 and index +1
                total_hits_up = hits_up_0 + hits_up_1
                total_hits_down = hits_down_0 + hits_down_1

                # Calculate the ratios
                total_signals_up = all_up.sum()
                total_signals_down = all_down.sum()
                ratio_convergence_up = round(total_hits_up / total_signals_up, 4) if total_signals_up > 0 else 0
                ratio_divergence_up = round(1 - ratio_convergence_up, 4)
                ratio_convergence_down = round(total_hits_down / total_signals_down, 4) if total_signals_down > 0 else 0
                ratio_divergence_down = round(1 - ratio_convergence_down, 4)

                # Distribution of potential signals by month for incorrect up and down signals
                incorrect_up_0['month'] = incorrect_up_0['timestamp'].dt.to_period('M')
                incorrect_down_0['month'] = incorrect_down_0['timestamp'].dt.to_period('M')

                up_distribution = incorrect_up_0.groupby('month').size().rename('up_count')
                down_distribution = incorrect_down_0.groupby('month').size().rename('down_count')

                # Combine the distributions and add to the list
                monthly_distribution = pd.concat([up_distribution, down_distribution], axis=1).reset_index()
                monthly_distribution['combination'] = ', '.join(comb)
                distribution_list.append(monthly_distribution)

                # Append the results
                results_list.append({
                    'combination': ', '.join(comb),
                    'number_of_all_up_signals': round(incorrect_up_0.shape[0], 0),
                    'number_of_all_down_signals': round(incorrect_down_0.shape[0], 0),
                    'correct_hits_up_convergence': round(incorrect_up_0.shape[0] * ratio_convergence_up, 0),
                    'correct_hits_down_convergence': round(incorrect_down_0.shape[0] * ratio_convergence_down, 0),
                    'correct_hits_up_divergence': round(incorrect_up_0.shape[0] * ratio_divergence_up, 0),
                    'correct_hits_down_divergence': round(incorrect_down_0.shape[0] * ratio_divergence_down, 0),
                    'ratio_convergence_up': ratio_convergence_up,
                    'ratio_divergence_up': ratio_divergence_up,
                    'ratio_convergence_down': ratio_convergence_down,
                    'ratio_divergence_down': ratio_divergence_down,
                })
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results_list)
    path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/convergence_divergence/'
    results_df.to_csv(path_for_results + 'single_indicator_delayed_' + token_name + '.csv', index=False)

    # Convert distribution list to DataFrame and save
    distribution_df = pd.concat(distribution_list, ignore_index=True)
    distribution_df.to_csv(path_for_results + 'distribution_single_indicator_' + token_name + '.csv', index=False)

    return df_input


def _5_single_indicator_correlation_big_delta_delayed(df_input, token_name):
    percentile_threshold = 95
    percentile_value = np.percentile(abs(df_input['close_delta'].dropna()), percentile_threshold)
    df_input['close_big_move_percentile'] = abs(df_input['close_delta']) > percentile_value
    df_input['close_big_move_positive'] = (df_input['close_delta'] > 0) & df_input['close_big_move_percentile']
    df_input['close_big_move_negative'] = (df_input['close_delta'] < 0) & df_input['close_big_move_percentile']

    for element in indicators:
        delta_column = element + '_delta'
        if delta_column in df_input.columns and not df_input[delta_column].dropna().empty:
            percentile_value = np.percentile(abs(df_input[delta_column].dropna()), percentile_threshold)
            df_input[element + '_big_move_percentile'] = abs(df_input[delta_column]) > percentile_value
            df_input[element + '_big_move_positive'] = (df_input[delta_column] > 0) & df_input[element + '_big_move_percentile']
            df_input[element + '_big_move_negative'] = (df_input[delta_column] < 0) & df_input[element + '_big_move_percentile']
        else:
            df_input[element + '_big_move_percentile'] = False
            df_input[element + '_big_move_positive'] = False
            df_input[element + '_big_move_negative'] = False
    
    number_parameter = 1
    number_of_minimum_parameter = number_parameter
    number_of_maximum_parameter = number_parameter

    results_list = []

    for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
        combs = list(combinations(indicators, r))
        for comb in combs:
            big_delta_columns = [element + '_big_move_percentile' for element in comb]
            positive_columns = [element + '_big_move_positive' for element in comb]
            negative_columns = [element + '_big_move_negative' for element in comb]

            if all(col in df_input.columns for col in big_delta_columns):
                
                big_delta = df_input[big_delta_columns].apply(lambda row: all(x == True for x in row), axis=1)
                big_positive_delta = df_input[positive_columns].apply(lambda row: all(x == True for x in row), axis=1)
                big_negative_delta = df_input[negative_columns].apply(lambda row: all(x == True for x in row), axis=1)
                
                hits_big_delta_same_index = df_input[big_delta & (df_input['close_big_move_percentile'] == True)].shape[0]
                hits_positive_delta_same_index = df_input[big_positive_delta & (df_input['close_big_move_positive'] == True)].shape[0]
                hits_negative_delta_same_index = df_input[big_negative_delta & (df_input['close_big_move_negative'] == True)].shape[0]

                # Checking for hits at shifted index (+1) where the same index was wrong
                delayed_hits_shifted = df_input[big_delta & ~(df_input['close_big_move_percentile']) & 
                                                (df_input['close_big_move_percentile'].shift(-1) == True)].shape[0]
                delayed_hits_positive_shifted = df_input[big_positive_delta & ~(df_input['close_big_move_positive']) & 
                                                         (df_input['close_big_move_positive'].shift(-1) == True)].shape[0]
                delayed_hits_negative_shifted = df_input[big_negative_delta & ~(df_input['close_big_move_negative']) & 
                                                         (df_input['close_big_move_negative'].shift(-1) == True)].shape[0]

                total_big_delta = big_delta.sum()
                total_positive_delta = big_positive_delta.sum()
                total_negative_delta = big_negative_delta.sum()

                results_list.append({
                    'combination': ', '.join(comb),
                    'total_rows': df_input.shape[0],
                    'number_of_big_delta_signals': total_big_delta,
                    'number_of_positive_delta_signals': total_positive_delta,
                    'number_of_negative_delta_signals': total_negative_delta,
                    'ratio_number_of_signals': round((total_big_delta / df_input.shape[0]) * 100, 4),
                    'hits_big_delta_same_index': hits_big_delta_same_index,
                    'hits_positive_delta_same_index': hits_positive_delta_same_index,
                    'hits_negative_delta_same_index': hits_negative_delta_same_index,
                    'delayed_hits_shifted_index_1': delayed_hits_shifted,
                    'delayed_hits_positive_shifted_index_1': delayed_hits_positive_shifted,
                    'delayed_hits_negative_shifted_index_1': delayed_hits_negative_shifted,
                    'ratio_big_delta_shifted_index_1': round((delayed_hits_shifted / total_big_delta), 4) if total_big_delta > 0 else 0,
                    'ratio_positive_delta_shifted_index_1': round((delayed_hits_positive_shifted / total_positive_delta), 4) if total_positive_delta > 0 else 0,
                    'ratio_negative_delta_shifted_index_1': round((delayed_hits_negative_shifted / total_negative_delta), 4) if total_negative_delta > 0 else 0,
                })
    
    results_df = pd.DataFrame(results_list)
    path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/big_delta/'
    results_df.to_csv(path_for_results + 'single_indicator_delayed_' + token_name + '.csv', index=False)
    return df_input


def _6_multiple_indicator_correlation_trend(df_input, token_name):
    numbers = [2]
    for number_parameter in numbers:
        number_of_minimum_parameter = number_parameter
        number_of_maximum_parameter = number_parameter

        results_list = []

        for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
            combs = list(combinations(indicators, r))
            for comb in combs:
                trend_columns = [element + '_trend' for element in comb]
                if all(col in df_input.columns for col in trend_columns):
                    
                    # Determine when all indicators are 'up' or 'down' at time t
                    all_up = df_input[trend_columns].apply(lambda row: all(x == 'up' for x in row), axis=1)
                    all_down = df_input[trend_columns].apply(lambda row: all(x == 'down' for x in row), axis=1)
                    
                    # Compare indicators at time t with close_trend at time t+1
                    hits_up = df_input[all_up & (df_input['close_trend'].shift(shift_parameter) == 'up')].shape[0]
                    hits_down = df_input[all_down & (df_input['close_trend'].shift(shift_parameter) == 'down')].shape[0]

                    # Shifted hits
                    hits_shifted_up = {}
                    hits_shifted_down = {}
                    current_up_hits = df_input[all_up & (df_input['close_trend'].shift(shift_parameter) == 'up')].shape[0]
                    hits_shifted_up[f'hits_up_shifted_index_1'] = current_up_hits

                    current_down_hits = df_input[all_down & (df_input['close_trend'].shift(shift_parameter) == 'down')].shape[0]
                    hits_shifted_down[f'hits_down_shifted_index_1'] = current_down_hits

                    results_list.append({
                        'combination': ', '.join(comb),
                        'number_of_all_up_signals': all_up.sum(),
                        'number_of_all_down_signals': all_down.sum(),
                        'hits_up_same_index': hits_up,
                        'hits_down_same_index': hits_down,
                        **hits_shifted_up,
                        **hits_shifted_down,
                        'ratio_convergence_up_shifted_index_1': round((hits_shifted_up['hits_up_shifted_index_1'] / all_up.sum()), 4) if all_up.sum() > 0 else 0,
                        'ratio_divergence_up_shifted_index_1': round((1 - (hits_shifted_up['hits_up_shifted_index_1'] / all_up.sum())), 4) if all_up.sum() > 0 else 0,
                        'ratio_convergence_down_shifted_index_1': round((hits_shifted_down['hits_down_shifted_index_1'] / all_down.sum()), 4) if all_down.sum() > 0 else 0,
                        'ratio_divergence_down_shifted_index_1': round((1 - (hits_shifted_down['hits_down_shifted_index_1'] / all_down.sum())), 4) if all_down.sum() > 0 else 0,
                    })
        
        results_df = pd.DataFrame(results_list)
        path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/convergence_divergence/'
        results_df.to_csv(path_for_results + 'multiple_indicator_' + str(number_parameter) + '_' + token_name + '.csv', index=False)
    return df_input


def _6_multiple_indicator_correlation_big_delta(df_input, token_name):
    percentile_threshold = 95
    percentile_value = np.percentile(abs(df_input['close_delta'].dropna()), percentile_threshold)
    df_input['close_big_move_percentile'] = abs(df_input['close_delta']) > percentile_value
    df_input['close_big_move_positive'] = (df_input['close_delta'] > 0) & df_input['close_big_move_percentile']
    df_input['close_big_move_negative'] = (df_input['close_delta'] < 0) & df_input['close_big_move_percentile']

    for element in indicators:
        delta_column = element + '_delta'
        if delta_column in df_input.columns and not df_input[delta_column].dropna().empty:
            percentile_value = np.percentile(abs(df_input[delta_column].dropna()), percentile_threshold)
            df_input[element + '_big_move_percentile'] = abs(df_input[delta_column]) > percentile_value
            df_input[element + '_big_move_positive'] = (df_input[delta_column] > 0) & df_input[element + '_big_move_percentile']
            df_input[element + '_big_move_negative'] = (df_input[delta_column] < 0) & df_input[element + '_big_move_percentile']
        else:
            df_input[element + '_big_move_percentile'] = False
            df_input[element + '_big_move_positive'] = False
            df_input[element + '_big_move_negative'] = False
    
    numbers = [2]
    results_list = []
    distribution_list = []  # List to store distribution results

    for number_parameter in numbers:
        number_of_minimum_parameter = number_parameter
        number_of_maximum_parameter = number_parameter

        for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
            combs = list(combinations(indicators, r))
            for comb in combs:
                big_delta_columns = [element + '_big_move_percentile' for element in comb]
                positive_columns = [element + '_big_move_positive' for element in comb]
                negative_columns = [element + '_big_move_negative' for element in comb]

                if all(col in df_input.columns for col in big_delta_columns):
                    
                    big_delta = df_input[big_delta_columns].apply(lambda row: all(x == True for x in row), axis=1)
                    big_positive_delta = df_input[positive_columns].apply(lambda row: all(x == True for x in row), axis=1)
                    big_negative_delta = df_input[negative_columns].apply(lambda row: all(x == True for x in row), axis=1)
                    
                    hits_big_delta_same_index = df_input[big_delta & (df_input['close_big_move_percentile'] == True)].shape[0]
                    hits_positive_delta_same_index = df_input[big_positive_delta & (df_input['close_big_move_positive'] == True)].shape[0]
                    hits_negative_delta_same_index = df_input[big_negative_delta & (df_input['close_big_move_negative'] == True)].shape[0]

                    hits_shifted = {}
                    hits_positive_shifted = {}
                    hits_negative_shifted = {}
                    for shift in range(1, 2):
                        shifted_hits = df_input[big_delta & df_input['close_big_move_percentile'].shift(-shift) == True].index
                        shifted_hits_positive = df_input[big_positive_delta & df_input['close_big_move_positive'].shift(-shift) == True].index
                        shifted_hits_negative = df_input[big_negative_delta & df_input['close_big_move_negative'].shift(-shift) == True].index

                        hits_shifted[f'hits_big_delta_shift_index_{shift}'] = len(shifted_hits)
                        hits_positive_shifted[f'hits_positive_delta_shift_index_{shift}'] = len(shifted_hits_positive)
                        hits_negative_shifted[f'hits_negative_delta_shift_index_{shift}'] = len(shifted_hits_negative)

                    total_big_delta = big_delta.sum()
                    total_positive_delta = big_positive_delta.sum()
                    total_negative_delta = big_negative_delta.sum()

                    # Distribution of signals by month
                    df_input['month'] = df_input['timestamp'].dt.to_period('M')

                    positive_distribution = df_input[big_positive_delta].groupby('month').size().rename('positive_delta_count')
                    negative_distribution = df_input[big_negative_delta].groupby('month').size().rename('negative_delta_count')

                    # Combine the distributions and add to the list
                    monthly_distribution = pd.concat([positive_distribution, negative_distribution], axis=1).reset_index()
                    monthly_distribution['combination'] = ', '.join(comb)
                    distribution_list.append(monthly_distribution)

                    # Append the results
                    results_list.append({
                        'combination': ', '.join(comb),
                        'total_rows': df_input.shape[0],
                        'number_of_big_delta_signals': total_big_delta,
                        'number_of_positive_delta_signals': total_positive_delta,
                        'number_of_negative_delta_signals': total_negative_delta,
                        'ratio_number_of_signals': round((total_big_delta / df_input.shape[0]) * 100, 4),
                        'hits_big_delta_same_index': hits_big_delta_same_index,
                        'hits_positive_delta_same_index': hits_positive_delta_same_index,
                        'hits_negative_delta_same_index': hits_negative_delta_same_index,
                        **hits_shifted,
                        **hits_positive_shifted,
                        **hits_negative_shifted,
                        'ratio_big_delta_shifted_index_1': round((hits_shifted['hits_big_delta_shift_index_1'] / total_big_delta), 4) if total_big_delta > 0 else 0,
                        'ratio_positive_delta_shifted_index_1': round((hits_positive_shifted['hits_positive_delta_shift_index_1'] / total_positive_delta), 4) if total_positive_delta > 0 else 0,
                        'ratio_negative_delta_shifted_index_1': round((hits_negative_shifted['hits_negative_delta_shift_index_1'] / total_negative_delta), 4) if total_negative_delta > 0 else 0,
                    })
        
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results_list)
        path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/big_delta/'
        results_df.to_csv(path_for_results + 'multiple_indicator_'+ str(number_parameter) + '_' + token_name + '.csv', index=False)

        # Convert distribution list to DataFrame and save
        distribution_df = pd.concat(distribution_list, ignore_index=True)
        distribution_df.to_csv(path_for_results + 'distribution_multiple_indicator_' + token_name + '.csv', index=False)

    return df_input


def _6_multiple_indicator_correlation_trend_delayed(df_input, token_name):
    numbers = [2]
    results_list = []
    distribution_list = []  # List to store distribution results

    for number_parameter in numbers:
        number_of_minimum_parameter = number_parameter
        number_of_maximum_parameter = number_parameter

        for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
            combs = list(combinations(indicators, r))
            for comb in combs:
                trend_columns = [element + '_trend' for element in comb]
                if all(col in df_input.columns for col in trend_columns):
                    
                    # Determine when all indicators are 'up' or 'down' at time t
                    all_up = df_input[trend_columns].apply(lambda row: all(x == 'up' for x in row), axis=1)
                    all_down = df_input[trend_columns].apply(lambda row: all(x == 'down' for x in row), axis=1)
                    
                    # Step 1: Check hits at index 0 (same index)
                    hits_up_0 = df_input[all_up & (df_input['close_trend'] == 'up')].shape[0]
                    hits_down_0 = df_input[all_down & (df_input['close_trend'] == 'down')].shape[0]

                    # Mark rows where the hit was incorrect at index 0
                    incorrect_up_0 = df_input[all_up & (df_input['close_trend'] != 'up')]
                    incorrect_down_0 = df_input[all_down & (df_input['close_trend'] != 'down')]

                    # Step 2: Check hits at index +1 for the previously incorrect rows
                    hits_up_1 = incorrect_up_0[incorrect_up_0['close_trend'].shift(-1) == 'up'].shape[0]
                    hits_down_1 = incorrect_down_0[incorrect_down_0['close_trend'].shift(-1) == 'down'].shape[0]

                    # Total hits considering both index 0 and index +1
                    total_hits_up = hits_up_0 + hits_up_1
                    total_hits_down = hits_down_0 + hits_down_1

                    # Calculate the ratios
                    total_signals_up = all_up.sum()
                    total_signals_down = all_down.sum()
                    ratio_convergence_up = round(total_hits_up / total_signals_up, 4) if total_signals_up > 0 else 0
                    ratio_divergence_up = round(1 - ratio_convergence_up, 4)
                    ratio_convergence_down = round(total_hits_down / total_signals_down, 4) if total_signals_down > 0 else 0
                    ratio_divergence_down = round(1 - ratio_convergence_down, 4)

                    # Distribution of potential signals by month for incorrect up and down signals
                    incorrect_up_0['month'] = incorrect_up_0['timestamp'].dt.to_period('M')
                    incorrect_down_0['month'] = incorrect_down_0['timestamp'].dt.to_period('M')

                    up_distribution = incorrect_up_0.groupby('month').size().rename('up_count')
                    down_distribution = incorrect_down_0.groupby('month').size().rename('down_count')

                    # Combine the distributions and add to the list
                    monthly_distribution = pd.concat([up_distribution, down_distribution], axis=1).reset_index()
                    monthly_distribution['combination'] = ', '.join(comb)
                    distribution_list.append(monthly_distribution)

                    # Append the results
                    results_list.append({
                        'combination': ', '.join(comb),
                        'number_of_all_up_signals': round(incorrect_up_0.shape[0], 0),
                        'number_of_all_down_signals': round(incorrect_down_0.shape[0], 0),
                        'correct_hits_up_convergence': round(incorrect_up_0.shape[0] * ratio_convergence_up, 0),
                        'correct_hits_down_convergence': round(incorrect_down_0.shape[0] * ratio_convergence_down, 0),
                        'correct_hits_up_divergence': round(incorrect_up_0.shape[0] * ratio_divergence_up, 0),
                        'correct_hits_down_divergence': round(incorrect_down_0.shape[0] * ratio_divergence_down, 0),
                        'ratio_convergence_up': ratio_convergence_up,
                        'ratio_divergence_up': ratio_divergence_up,
                        'ratio_convergence_down': ratio_convergence_down,
                        'ratio_divergence_down': ratio_divergence_down,
                    })
        
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results_list)
        path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/convergence_divergence/'
        results_df.to_csv(path_for_results + 'multiple_indicator_delayed_' + str(number_parameter) + '_' + token_name + '.csv', index=False)

        # Convert distribution list to DataFrame and save
        distribution_df = pd.concat(distribution_list, ignore_index=True)
        distribution_df.to_csv(path_for_results + 'distribution_multiple_indicator_' + token_name + '.csv', index=False)

    return df_input


def _6_multiple_indicator_correlation_big_delta_delayed(df_input, token_name):
    percentile_threshold = 95
    percentile_value = np.percentile(abs(df_input['close_delta'].dropna()), percentile_threshold)
    df_input['close_big_move_percentile'] = abs(df_input['close_delta']) > percentile_value
    df_input['close_big_move_positive'] = (df_input['close_delta'] > 0) & df_input['close_big_move_percentile']
    df_input['close_big_move_negative'] = (df_input['close_delta'] < 0) & df_input['close_big_move_percentile']

    for element in indicators:
        delta_column = element + '_delta'
        if delta_column in df_input.columns and not df_input[delta_column].dropna().empty:
            percentile_value = np.percentile(abs(df_input[delta_column].dropna()), percentile_threshold)
            df_input[element + '_big_move_percentile'] = abs(df_input[delta_column]) > percentile_value
            df_input[element + '_big_move_positive'] = (df_input[delta_column] > 0) & df_input[element + '_big_move_percentile']
            df_input[element + '_big_move_negative'] = (df_input[delta_column] < 0) & df_input[element + '_big_move_percentile']
        else:
            df_input[element + '_big_move_percentile'] = False
            df_input[element + '_big_move_positive'] = False
            df_input[element + '_big_move_negative'] = False
    
    numbers = [2]
    for number_parameter in numbers:
        number_of_minimum_parameter = number_parameter
        number_of_maximum_parameter = number_parameter

        results_list = []

        for r in range(number_of_minimum_parameter, number_of_maximum_parameter + 1):
            combs = list(combinations(indicators, r))
            for comb in combs:
                big_delta_columns = [element + '_big_move_percentile' for element in comb]
                positive_columns = [element + '_big_move_positive' for element in comb]
                negative_columns = [element + '_big_move_negative' for element in comb]

                if all(col in df_input.columns for col in big_delta_columns):
                    
                    big_delta = df_input[big_delta_columns].apply(lambda row: all(x == True for x in row), axis=1)
                    big_positive_delta = df_input[positive_columns].apply(lambda row: all(x == True for x in row), axis=1)
                    big_negative_delta = df_input[negative_columns].apply(lambda row: all(x == True for x in row), axis=1)
                    
                    hits_big_delta_same_index = df_input[big_delta & (df_input['close_big_move_percentile'] == True)].shape[0]
                    hits_positive_delta_same_index = df_input[big_positive_delta & (df_input['close_big_move_positive'] == True)].shape[0]
                    hits_negative_delta_same_index = df_input[big_negative_delta & (df_input['close_big_move_negative'] == True)].shape[0]

                    # Checking for hits at shifted index (+1) where the same index was wrong
                    delayed_hits_shifted = df_input[big_delta & ~(df_input['close_big_move_percentile']) & 
                                                    (df_input['close_big_move_percentile'].shift(-1) == True)].shape[0]
                    delayed_hits_positive_shifted = df_input[big_positive_delta & ~(df_input['close_big_move_positive']) & 
                                                             (df_input['close_big_move_positive'].shift(-1) == True)].shape[0]
                    delayed_hits_negative_shifted = df_input[big_negative_delta & ~(df_input['close_big_move_negative']) & 
                                                             (df_input['close_big_move_negative'].shift(-1) == True)].shape[0]

                    total_big_delta = big_delta.sum()
                    total_positive_delta = big_positive_delta.sum()
                    total_negative_delta = big_negative_delta.sum()

                    results_list.append({
                        'combination': ', '.join(comb),
                        'total_rows': df_input.shape[0],
                        'number_of_big_delta_signals': total_big_delta,
                        'number_of_positive_delta_signals': total_positive_delta,
                        'number_of_negative_delta_signals': total_negative_delta,
                        'ratio_number_of_signals': round((total_big_delta / df_input.shape[0]) * 100, 4),
                        'hits_big_delta_same_index': hits_big_delta_same_index,
                        'hits_positive_delta_same_index': hits_positive_delta_same_index,
                        'hits_negative_delta_same_index': hits_negative_delta_same_index,
                        'delayed_hits_shifted_index_1': delayed_hits_shifted,
                        'delayed_hits_positive_shifted_index_1': delayed_hits_positive_shifted,
                        'delayed_hits_negative_shifted_index_1': delayed_hits_negative_shifted,
                        'ratio_big_delta_shifted_index_1': round((delayed_hits_shifted / total_big_delta), 4) if total_big_delta > 0 else 0,
                        'ratio_positive_delta_shifted_index_1': round((delayed_hits_positive_shifted / total_positive_delta), 4) if total_positive_delta > 0 else 0,
                        'ratio_negative_delta_shifted_index_1': round((delayed_hits_negative_shifted / total_negative_delta), 4) if total_negative_delta > 0 else 0,
                    })
        
        results_df = pd.DataFrame(results_list)
        path_for_results = f'C:/Users/maxim/python/database/3rd_paper/{token_name}/results/big_delta/'
        results_df.to_csv(path_for_results + 'multiple_indicator_delayed_'+ str(number_parameter) + '_' + token_name + '.csv', index=False)
    return df_input


if __name__ == "__main__":
    main()