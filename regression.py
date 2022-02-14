import numpy as np

from correlation.find_correlations import find_correlations

import pickle

import time

def get_y_data(args, pandas_input):
    """
    :return: list (int) 0 hit stop loss, 1 no sell, 2 hit profit goal
    """
    # pandas_input = pandas_input.dropna()
    X = []
    y = []
    for col in pandas_input.columns:
        if not col in args.columns_not_to_process:
            block_indexes = getContinuousBlocks(pandas_input[col], acceptableGapsSize=0)
            for block in block_indexes:
                continuousDataBlock = pandas_input[col][block[0]:block[1]].to_numpy()
                if not(np.isnan(continuousDataBlock)).any():
                    for i in range(args.days_to_think, len(continuousDataBlock) - args.max_hold_period - 1):
                        X.append( (col, int(i-args.days_to_think),int(i)) )
                        buy_value = continuousDataBlock[i]
                        positionSold = False
                        j = 0
                        while j < args.max_hold_period and not(positionSold):
                            j += 1
                            if continuousDataBlock[i+j+1] < buy_value * (1-args.stop_loss):
                                y.append(0)
                                positionSold = True
                            elif continuousDataBlock[i+j+1] > buy_value * (1+args.minimum_return):
                                y.append(2)
                                positionSold = True
                        if not(positionSold):
                            y.append(1)
                    if len(X) > 200:
                        X = np.array(X)
                        y = np.array(y)
                        return X, y
    X = np.array(X)
    y = np.array(y)
    return X, y

def generate_y_values(args, pandas_dataset, columns, x_slices):
    pass

def run_regression(args, pandas_input, smoothedPandasInput):
    # find_correlations(args, pandas_input)

    # with open(rf"{args.data_directory}flattened_correlations.pkl", "rb") as input_file:
    #     flattened_correlations = pickle.load(input_file)
    #
    # with open(rf"{args.data_directory}correlation_maps.pkl", "rb") as input_file:
    #     correlation_maps = pickle.load(input_file)

    with open(rf"{args.data_directory}set_indexes.pkl", "rb") as input_file:
        set_indexes = pickle.load(input_file)[:50]

    with open(rf"{args.data_directory}flattened_correlations_minimized.pkl", "rb") as input_file:
        flattened_correlations = pickle.load(input_file)

    with open(rf"{args.data_directory}correlation_maps_minimized.pkl", "rb") as input_file:
        correlation_maps = pickle.load(input_file)


    for i in range(len(flattened_correlations)):
        set_indexes_selected = set_indexes[i]
        for col in correlation_maps[i].keys():
            correlations_for_col = correlation_maps[i][col]
            mask = (correlations_for_col > .95) | (correlations_for_col < -.95)
            highly_correlated_pairs = correlations_for_col[mask]

            generate_y_values(args, pandas_input, [col] + list(highly_correlated_pairs.index), set_indexes_selected)



        mask = flattened_correlations[i] > .95
        significantValues_selected = flattened_correlations[i][mask]
        set_indexes_selected = set_indexes[i]

        pass
