
import argparse

import pandas as pd

from data_preper.clean_data import prep_data

from data_labeler.label_data import regression

from correlation.find_correlations import find_correlations

from regression2 import run_regression

from transformer_test import run_test

def data_arguments():
    parser = argparse.ArgumentParser('model', add_help=False)

    parser.add_argument('--fourier_window_size', type=int,
                        help='(greater than 30) size of window to use in calculating fourier transform',
                        default=100)

    # parser.add_argument('--fourier_window_margin', type=float,
    #                     help='(percentage of window) how much of window to exclude in summery, percentage is split between left and right of window',
    #                     default=.2)

    parser.add_argument('--percent_of_frequencies_to_keep', type=float,
                        help='(percentage of window size) number of terms to cut off in fourier transform (larger means more smoothing)',
                        default=.5)

    # parser.add_argument('--path_to_returns_file', type=str, default="../data/returns.csv")
    parser.add_argument('--path_to_returns_file', type=str, default="../../data/stock_data/returns.csv")
    # parser.add_argument('--path_to_stock_info_file', type=str, default="../data/stock_info.csv")
    parser.add_argument('--path_to_stock_info_file', type=str, default="../../data/stock_data/stock_info.csv")

    parser.add_argument('--data_directory', type=str, default="../../data/stock_data/")

    parser.add_argument('--columns_not_to_process', default=['TICKER'])
    parser.add_argument('--largest_holes_to_fill', default=10)
    parser.add_argument('--patience', default=5)
    parser.add_argument('--max_branch', default=10)
    parser.add_argument('--min_points_needed_to_interpolate', default=8)
    parser.add_argument('--days_to_think', default=15)

    parser.add_argument('--days_for_correlation_relavency', default=50)
    return parser


def get_model_arguments():
    parser = argparse.ArgumentParser('model', add_help=False)
    parser.add_argument('--max_hold_period', default=4, type=str, help="This help isn't actually helpful, sorry :(")
    parser.add_argument('--minimum_return', default=.05, type=str, help="This help isn't actually helpful, sorry :(")
    parser.add_argument('--stop_loss', default=.05, type=str, help="This help isn't actually helpful, sorry :(")
    return parser


if __name__ == '__main__':
    dataset_arguemnts = argparse.ArgumentParser('DINO', parents=[data_arguments(), get_model_arguments()])
    args = dataset_arguemnts.parse_args()

    data = pd.read_csv(args.data_directory+"cleaned_data.csv")
    smoothedPandasInput = pd.read_csv(args.data_directory+"smoothed_data.csv")

    # run_test(args, data, smoothedPandasInput)

    print(run_regression(args, data, smoothedPandasInput))
