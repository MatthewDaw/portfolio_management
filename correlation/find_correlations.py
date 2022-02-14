from misc import getContinuousBlocks
from misc import getContinuousBlocksFullDataset
import numpy as np
import pickle

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def find_correlations(args, pandas_input):

    # general variables for iteration
    days_in_dataset = pandas_input.shape[0]
    pandas_without_dates = pandas_input.drop(['TICKER'], axis=1)
    daily_returns = pandas_without_dates.pct_change()
    start = 0
    end = args.days_for_correlation_relavency

    flattened_correlations = []
    correlation_maps = []
    set_indexes = []

    # iterating each mock trading day
    while end <= days_in_dataset:
        # get only stocks that are all non-nan, we extend outside of bounds we are correlating that we know y values will be defined
        active_stock = daily_returns[start:end+args.max_hold_period].dropna(axis='columns')

        # stock correlations
        correlations_scores = active_stock.corr()

        # flattened_corr_sorted, index is pairs of stock (eg, ('MFD', 'NXN') ) and entries are (not-absolute) correlation value
            # list is ordered from highest correlated to least correlated
        #other variables are just intermediate values
        labels_to_drop = get_redundant_pairs(active_stock)
        flattened_corr = correlations_scores.unstack().drop(labels=labels_to_drop)
        flattened_corr_abs = correlations_scores.abs().unstack()
        flattened_corr_abs_sorted = flattened_corr_abs.drop(labels=labels_to_drop).sort_values(ascending=False)
        flattened_corr_sorted = flattened_corr.loc[flattened_corr_abs_sorted.index]

        # correlation_map is map where the keys are stock tickers and values is a list where indexes are tickers and values are the (not absolute) correlation score
        correlation_map = {}
        absolute_correlations = correlations_scores.abs()

        for col in absolute_correlations.columns:
            row = absolute_correlations[col]
            row = row.drop([col])
            row = row.sort_values(ascending=False)
            correlation_map[col] = correlations_scores[col].loc[row.index]

        flattened_correlations.append(flattened_corr_sorted)
        correlation_maps.append(correlation_map)
        set_indexes.append((start, end))
        start += 1
        end += 1

    with open(rf"{args.data_directory}flattened_correlations.pkl", "wb") as output_file:
        pickle.dump(flattened_correlations, output_file)

    with open(rf"{args.data_directory}correlation_maps.pkl", "wb") as output_file:
        pickle.dump(correlation_maps, output_file)

    with open(rf"{args.data_directory}set_indexes.pkl", "wb") as output_file:
        pickle.dump(set_indexes, output_file)

    return flattened_correlations, correlation_maps, set_indexes

