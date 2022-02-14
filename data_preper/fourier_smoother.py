import numpy as np
import pandas as pd

from misc import getContinuousBlocks

import matplotlib.pyplot as plt

# fourier_window_size
# fourier_cut_off

def smooth_continuous_block(args, data):
    data = data.to_numpy()
    window_size = args.fourier_window_size
    window_shift_size = window_size // 2
    start = 0
    len_of_data = len(data)
    end = min(window_size, len_of_data)

    rescaling = np.zeros(len_of_data)
    smoothed_data = np.zeros(len_of_data)

    while rescaling[-1] == 0:
        data_segment = data[start:end]
        n = len(data_segment)
        tiltingConstant = data_segment[0] + (np.linspace(0, 1, n) * (data_segment[-1] - data_segment[0]))
        tilted_data_segment = data_segment - tiltingConstant
        fhat = np.fft.fft(tilted_data_segment, n)
        psd_idxs = np.arange(n) < args.percent_of_frequencies_to_keep*n
        fhat_clean = psd_idxs * fhat
        signal_filtered = np.fft.ifft(fhat_clean)
        rescaling[start:end] += 1
        smoothed_data[start:end] += (signal_filtered.real + tiltingConstant)
        start += window_shift_size
        end += window_shift_size

    processedData = smoothed_data * (1/rescaling)
    # plot for debugging, remove in final product
    # plt.plot(data)
    # plt.plot(processedData)
    # plt.show()
    return processedData


def fourier_smoother(args, pandas_input):
    for col in pandas_input.columns:
        if not col in args.columns_not_to_process:
            # small test, remove in final product
            # testData = [2,2,3,float('nan'),3,4,5,float('nan')
            # bill = pd.DataFrame()
            # bill['a'] = testData
            # block_indexes_pre = getContinuousBlocks(bill['a'], acceptableGapsSize=0)

            block_indexes = getContinuousBlocks(pandas_input[col], acceptableGapsSize=0)
            for block in block_indexes:
                continuousDataBlock = pandas_input[col][block[0]:block[1]]
                pandas_input.loc[block[0]:block[1], col] = continuousDataBlock

    return pandas_input