from data_preper.quick_hole_filler import fillQuickHoles
from data_preper.fourier_smoother import fourier_smoother
import pandas as pd
from data_preper.data_standardizer import standardizer_data

def prep_data(args):
    returns = pd.read_csv(args.data_directory + 'returns.csv')
    pandasInput, updatedColumnIndexes, pointsFilled = fillQuickHoles(args, returns)
    pandasInput = standardizer_data(args, pandasInput)
    smoothedPandasInput = fourier_smoother(args, pandasInput)
    return pandasInput, smoothedPandasInput