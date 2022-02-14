import numpy as np
from correlation.find_correlations import find_correlations
import pickle
import time
from misc import getContinuousBlocks
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cvxopt import matrix, solvers
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def make_y_values(args, pandas_input):
    # pandas_input = pandas_input.dropna()
    X = []
    y = []
    actual_return_next_time_step = []
    actual_return = []
    for col in pandas_input.columns:
        if not col in args.columns_not_to_process:
            block_indexes = getContinuousBlocks(pandas_input[col], acceptableGapsSize=0)
            for block in block_indexes:
                continuousDataBlock = pandas_input[col][block[0]:block[1]].to_numpy()
                if not(np.isnan(continuousDataBlock)).any():
                    for i in range(args.days_to_think, len(continuousDataBlock) - args.max_hold_period - 1):
                        assert not np.isnan(pandas_input[col][int(i-args.days_to_think + block[0]):int(i) + block[0]]).any()
                        # X.append( continuousDataBlock[int(i-args.days_to_think):int(i)] )
                        X.append((col, int(i-args.days_to_think + block[0]),int(i + block[0])))
                        future_data = continuousDataBlock[i:i+args.max_hold_period]
                        slope = np.polyfit(np.arange(len(future_data)), future_data, 1)[0]
                        y.append(slope)
                        actual_return_next_time_step.append((continuousDataBlock[i] - continuousDataBlock[i-1]))
                        actual_return.append((continuousDataBlock[i+args.max_hold_period-1] - continuousDataBlock[i-1]))

                    if len(X) > 200:
                        X = np.array(X)
                        y = np.array(y)
                        actual_return_next_time_step = np.array(actual_return_next_time_step)
                        actual_return = np.array(actual_return)
                        return X, np.array([y, actual_return_next_time_step, actual_return]).T
    X = np.array(X)
    y = np.array(y)
    actual_return_next_time_step = np.array(actual_return_next_time_step)
    actual_return = np.array(actual_return)
    return X, np.array([y, actual_return_next_time_step, actual_return]).T


def convert_x_data(smoothedPandasInput, keys):
    X = []
    for i, key in enumerate(keys):
        newXData = smoothedPandasInput[(key[0])][int(key[1]):int(key[2])]
        assert not np.isnan(newXData).any()
        X.append(newXData)
    return np.array(X)

def portfolioManagement(expected_returns, variance, R):

    B = [1, R]

    P = np.diag(variance)

    m = len(expected_returns)

    A = np.array([np.ones(m), np.array(expected_returns)])

    q = np.zeros(m)

    distributedX = np.array(solvers.qp(matrix(P), matrix(q), A=matrix(A), b=matrix(B))['x']).T

    return np.prod(distributedX @ expected_returns), distributedX @ variance, distributedX

def run_animation(posterior_mean, distributed_xs, rewards, risks, actual_return_next_time_step, actual_return):
    fig, ax = plt.subplots(2)
    min_y = min([ min(abs(step)) for step in distributed_xs ])
    max_y = max([max(step) for step in distributed_xs])
    min_x = min(posterior_mean)
    max_x = max(posterior_mean)
    ax[0].set_xlim([min_x, max_x])
    ax[0].set_xlabel('Expected Return')
    ax[0].set_ylim([min_y, max_y])
    ax[0].set_ylabel('Amount To Invest')
    # graph = ax.scatter(posterior_mean,  abs(distributed_xs[0]), color='0.8')
    titles = ["reward", "variance", "short return", "longer return"]
    plt.tight_layout()
    # ax[0].scatter(posterior_mean, abs(distributed_xs[0]))
    # num = 0
    # ax[1].bar(titles, [rewards[-num], risks[-num]])
    # plt.show()
    def update(num):
        ax[0].cla()
        ax[1].cla()
        ax[0].scatter(posterior_mean, abs(distributed_xs[num]))
        ax[1].bar(titles, [rewards[num], risks[num], actual_return_next_time_step[num], actual_return[num]])
        ax[0].set_xlabel('Expected Return')
        ax[0].set_ylabel('Amount To Invest')
        # graph.set_array(abs(distributed_xs[num]))
    N = len(distributed_xs) - 1
    # N = 60

    num = np.argmax(actual_return_next_time_step)

    update(num)



    # ani = animation.FuncAnimation(fig, update, N)
    # f = r"animation.gif"
    # writergif = animation.PillowWriter(fps=30)
    # ani.save(f, writer=writergif)

def run_regression(args, pandas_input, smoothedPandasInput):
    X, y = make_y_values(args, pandas_input)

    X = convert_x_data(smoothedPandasInput, X)

    X_pand = pd.DataFrame(X)

    X_pand.to_csv("X_stock.csv")

    y_pand = pd.DataFrame(y)

    y_pand.to_csv("y_stock.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    y_train_slope = y_train[:,0]
    y_train_actual_return = y_train[:, 1]

    y_test_slope = y_test[:, 0]
    y_test_actual_next_step = y_test[:, 1]
    y_test_actual_return = y_test[:, 2]
    


    param = {'max_depth': [2], 'eta': [1], 'objective': ['multi:softprob'], 'alpha': [.1], 'gamma': [.3],
             'lambda': [.1]}
    num_round = 2

    lambList = [.1, .8]

    gammaLlist = [.1, .8, 2]

    alphaList = [.1, .8]

    best_accuracy = 0
    best_hyperparameters = []


    for l in lambList:
        for g in gammaLlist:
            for a in alphaList:
                param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softprob', 'alpha': a, 'gamma': g, 'lambda': l}
                model = XGBClassifier(param)
                model.fit(X_train, y_train_slope)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test_slope, y_pred)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = [l, g, a]

    print(best_accuracy)

    print(best_hyperparameters)


    lambList = np.arange(-10, 6, dtype=float)

    error = []

    for lamb in lambList:
        lamb = 10 ** lamb
        clf = linear_model.Lasso(alpha=lamb, normalize=True, tol=1e-2, max_iter=100000)
        clf = clf.fit(X_train, y_train_slope)
        predictions = clf.predict(X_test)
        error.append(mean_squared_error(predictions, y_test_slope))

    bestLambda = 10**lambList[np.argmin(error)]

    clf = linear_model.Lasso(alpha=bestLambda, normalize=True, tol=1e-2, max_iter=100000)
    clf = clf.fit(X_train, y_train_slope)
    predictions = clf.predict(X_test)

    prior_variance = np.var(predictions - y_test_slope)

    mean_squared_error(predictions, y_test_slope)

    combined = np.array([y_test_slope, predictions, y_test_actual_next_step, y_test_actual_return]).T
    combined = np.array(sorted(combined, key=lambda x: x[0])).T

    y_test_slope, predictions, y_test_actual_next_step, y_test_actual_return = combined

    variance_box_size = 20

    variance_points = []

    for i in range(len(y_test) - variance_box_size):
        variance_points.append( np.var(y_test_slope[i:i+variance_box_size] - predictions[i:i+variance_box_size]) )


    posterior_variance = np.sqrt(np.array(prior_variance)**2 + np.array(variance_points)**2)
    y_test_slope = y_test_slope[int(variance_box_size/2):-int(variance_box_size/2)]
    predictions = predictions[int(variance_box_size/2):-int(variance_box_size/2)]
    y_test_actual_next_step = y_test_actual_next_step[int(variance_box_size/2):-int(variance_box_size/2)]
    y_test_actual_return = y_test_actual_return[int(variance_box_size/2):-int(variance_box_size/2)]

    # posterior_mean = (y_test_slope + predictions) / 2

    posterior_mean = predictions

    r_range = np.linspace(0.0001, max(np.abs(posterior_mean))*1.1, 200)

    rewards = []
    risks = []
    distributed_xs = []
    actual_return_next_time_step = []
    actual_return = []

    for r in r_range:
        reward, risk, distributed_x = portfolioManagement(posterior_mean, posterior_variance, r)
        rewards.append(r)
        risks.append(risk)
        distributed_xs.append(distributed_x[0])
        actual_return_next_time_step.append( distributed_x[0] @ y_test_actual_next_step )
        actual_return.append( distributed_x[0] @ y_test_actual_return )

    distributed_xs = np.array(distributed_xs)

    run_animation(posterior_mean, distributed_xs, rewards, risks, actual_return_next_time_step, actual_return)

    plt.plot(rewards, risks)
    # plt.show()
    # plt.plot(rewards, risks)
    plt.xlabel("Expected Return")
    plt.ylabel("Optimal Variance")
    plt.title("Optimal Versus Constrained Expected Return")
    plt.tight_layout()
    plt.show()

