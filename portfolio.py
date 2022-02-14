# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags

from matplotlib import pyplot as plt
import importlib
# importlib.import_module('mpl_toolkits.mplot3d').Axes3D

from mpl_toolkits.mplot3d import axes3d

# import cvxpy as cp

from cvxopt import matrix, solvers

from scipy.linalg import lu_factor, lu_solve

def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """

    x, y, u = startingPoint( Q, c, A, b, guess )

    m, n = A.shape

    sigma = .1

    def F(x,y,u):
        row1 = Q@x - A.T@u + c
        row2 = A@x - y - b
        row3 = np.diag(y) @ np.diag(u) @ np.ones(m)
        return np.concatenate((row1, row2, row3), axis=0)

    def DF(x,y, u):
        row1 = np.concatenate((Q,np.zeros((n,m)), -1*A.T ), axis=1)
        row2 = np.concatenate((A, -1*np.eye(m), np.zeros((m,m))), axis=1)
        row3 = np.concatenate((np.zeros((m,n)), np.diag(u), np.diag(y)), axis=1)
        return np.concatenate((row1,row2,row3))

    for i in range(niter+1):


        v = (y.T @ u) / n

        if v < tol:
            return x, c.T@x
        # calculate next step
        newStep = np.zeros(m * 2 + n)
        newStep[-1 * n:] = sigma * v

        fHat = -1 * F(x,y,u)
        fStep = fHat + newStep

        dfStep = DF(x,y, u)
        lu, piv = lu_factor(dfStep)
        changeInValues = lu_solve((lu, piv), fStep)

        # prepare dirction
        changeInX = changeInValues[:n]
        changeInY = changeInValues[n:m+n]
        changeInU = changeInValues[m+n:]

        # we are going to choose the minimum of (1, -ui/dui | dui < 0)
        possibleBetas= [1]
        for i in range(len(changeInU)):
            if changeInU[i] < 0:
                possibleBetas.append(-.95 * u[i] / changeInU[i])
        beta = min(possibleBetas)

        # choose delta in a similar way
        possibleDeltas = [1]
        for i in range(len(changeInY)):
            if changeInY[i] < 0:
                possibleDeltas.append(-.95 * y[i] / changeInY[i])
        delta = min(possibleDeltas)

        alpha = min(beta, delta)

        # adjust weights and directions
        x = x + changeInX * alpha
        y = y + changeInY * alpha
        u = u + changeInU * alpha

    return x, c.T@x

# if __name__ == '__main__':
#     Q = np.array([[1,-1], [-1,2]])
#     c = np.array([-2,-6])
#     A = np.array([[-1,1,-2,1,0], [-1,-2,-1,0,1] ]).T
#     b = np.array([-2,-2,-3,0,0])
#
#     x = np.array([.5,.5])
#
#     print(qInteriorPoint(Q, c, A, b, (x,np.ones(5),np.ones(5)), niter=30 ))

def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()


# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    H = laplacian(n)

    # Create the tent pole configuration.
    L = np.zeros((n, n))
    L[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1] = .5
    m = [n // 6 - 1, n // 6, int(5 * (n / 6.)) - 1, int(5 * (n / 6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()
    # Set initial guesses.
    x = np.ones((n, n)).ravel()
    y = np.ones(n ** 2)
    mu = np.ones(n ** 2)

    c = np.array([-1 * ((n - 1) ** -2) for i in range(n**2)])

    A = np.eye(225,225)

    z = qInteriorPoint(H, c, A, L, (x, y, mu))[0].reshape((n,n))

    # return L

    # Plot the solution.
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.show()

# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    inputData = np.loadtxt(filename)

    actualReturns = inputData[:,1:]

    R = 1.13

    Q = np.cov(actualReturns.T)

    # m is number of assets in given data
    m = len(Q)

    u = np.mean(actualReturns, axis=0)

    b = [1,1.13]

    A = np.array([np.ones(m), u])

    p = np.zeros(m)

    h = np.zeros(m)

    G = -1*np.eye(m)

    P = matrix(Q)
    b = matrix(b)
    A = matrix(A)
    q = matrix(p)
    G = matrix(G)
    h = matrix(h)

    return np.ravel(solvers.qp(P, q, A=A, b=b)['x']), np.ravel(solvers.qp(P, q, G, h, A, b)['x'])
