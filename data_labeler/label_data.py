
from misc import getContinuousBlocks

# max_hold_period minimum_return stop_loss

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

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


def convert_x_data(args, smoothedPandasInput, keys, y):
    X = []
    y_out = []
    for i, key in enumerate(keys):
        newXData = smoothedPandasInput[(key[0])][int(key[1]):int(key[2])]
        if len(newXData == args.days_to_think) and not(np.isnan(newXData)).any():
            X.append(newXData)
            y_out.append(y[i])
    return np.array(X), np.array(y_out)

def regression(args, data, smoothedPandasInput):
    X, y = label_data(args, data)

    X, y = convert_x_data(args, smoothedPandasInput, X, y)

    # y = y[:200]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    clf = LogisticRegression(random_state=0, multi_class='multinomial').fit(X_train, y_train)

    predictions = clf.predict(X_test)

    print("f1_score", f1_score(y_test, predictions, average='macro'))

    print("accuracy", accuracy_score(y_test, predictions))

