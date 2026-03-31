# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import struct
from scipy.optimize import least_squares
import time
import pdb
import numpy as np
from numpy.linalg import svd, norm, inv
import time
from scipy.linalg import svd, solve
from scipy.sparse.linalg import eigs
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs, svds


def GS2_gen(V1, V2, A, B, r):
    """
    Generalized Gram-Schmidt process with a specialized inner product defined by <.,.>_diag(AA', BB').

    Inputs:
        V1: ndarray
            Matrix whose columns are to be orthogonalized (shape: n1 x k).
        V2: ndarray
            Matrix whose columns are to be orthogonalized (shape: n2 x k).
        A: ndarray
            Matrix used in defining the inner product (shape: n1 x n1).
        B: ndarray
            Matrix used in defining the inner product (shape: n2 x n2).
        r: float
            Regularization parameter.

    Outputs:
        U1: ndarray
            Orthonormalized matrix corresponding to V1 (shape: n1 x k).
        U2: ndarray
            Orthonormalized matrix corresponding to V2 (shape: n2 x k).

    The function ensures that the combined matrix [U1; U2] has orthonormal columns with respect to the inner product
    defined by diag(AA', BB'), meaning:
        U1.T @ A @ A.T @ U1 + U2.T @ B @ B.T @ U2 = I
    """
    # Number of vectors to orthonormalize
    k = V1.shape[1]
    
    # Initialize U1 and U2 as zero matrices of the same shape as V1 and V2
    U1 = np.zeros_like(V1)
    U2 = np.zeros_like(V2)
    
    # Precompute M1 and M2 to simplify inner products and norms
    M1 = A @ A.T + r * np.eye(A.shape[0])
    M2 = B @ B.T + r * np.eye(B.shape[0])
    
    # Orthonormalize the first vector
    tmp = np.sqrt(
        V1[:, 0].T @ M1 @ V1[:, 0] +
        V2[:, 0].T @ M2 @ V2[:, 0]
    )
    U1[:, 0] = V1[:, 0] / tmp
    U2[:, 0] = V2[:, 0] / tmp
    
    # Iterate over the remaining vectors
    for i in range(1, k):
        U1[:, i] = V1[:, i]
        U2[:, i] = V2[:, i]
        for j in range(i):
            # Compute the inner product with respect to the specialized inner product
            tmp = (
                U1[:, i].T @ M1 @ U1[:, j] +
                U2[:, i].T @ M2 @ U2[:, j]
            )
            # Subtract the projection onto the previous vectors
            U1[:, i] -= tmp * U1[:, j]
            U2[:, i] -= tmp * U2[:, j]
        # Normalize the vector
        tmp = np.sqrt(
            U1[:, i].T @ M1 @ U1[:, i] +
            U2[:, i].T @ M2 @ U2[:, i]
        )
        U1[:, i] /= tmp
        U2[:, i] /= tmp
    
    return U1, U2
    
def GS_gen(V, B, r):
    """
    Generalized Gram-Schmidt process with inner product defined by <.,.>_{BB'}
    
    Parameters:
        V : ndarray
            Matrix whose columns are to be orthogonalized (shape: n x k)
        B : ndarray
            Matrix used in defining the inner product (shape: n x n)
        r : float
            Regularization parameter

    Returns:
        U : ndarray
            Matrix with orthonormal columns with respect to BB', satisfying U.T @ B @ B.T @ U = I
    """
    n, k = V.shape
    U = np.zeros((n, k))
    
    # Precompute M = B @ B.T + r * I for efficiency
    M = B @ B.T + r * np.eye(n)
    
    # Orthonormalize the first vector
    norm_factor = np.sqrt((V[:, 0].T @ M @ V[:, 0]))
    U[:, 0] = V[:, 0] / norm_factor
    
    # Orthonormalize the remaining vectors
    for i in range(1, k):
        U[:, i] = V[:, i]
        for j in range(i):
            # Compute the inner product with respect to the modified inner product
            inner_prod = U[:, i].T @ M @ U[:, j]
            # Subtract the projection onto the j-th vector
            U[:, i] -= inner_prod * U[:, j]
        # Normalize the vector
        norm_factor = np.sqrt(U[:, i].T @ M @ U[:, i])
        U[:, i] /= norm_factor

    return U
   
def SVRG_k(U_j, V, X, Y, r_x, M, m, eta):
    """
    SVRG to solve min_{U_j} tr(1/2 U_j' X X' U_j / N - U_j' X Y' V / N)
    For more details, see SVRG (Stochastic Variance Reduced Gradient) algorithm.

    Parameters:
    - U_j: ndarray
        Current estimate of U_j (shape: n_features x k)
    - V: ndarray
        Matrix V (shape: n_features x k)
    - X: ndarray
        Data matrix X (shape: n_features x N)
    - Y: ndarray
        Data matrix Y (shape: n_features x N)
    - r_x: float
        Regularization parameter
    - M: int
        Number of outer iterations
    - m: int
        Number of inner iterations
    - eta: float
        Step size (learning rate)

    Returns:
    - U_j: ndarray
        Updated estimate of U_j after M outer iterations
    """
    n_features, N = X.shape  # Get the number of features and samples

    for _ in range(M):
        W_0 = U_j.copy()
        W_t = W_0.copy()

        # Compute the full gradient at W_0 (batch gradient)
        batch_grad = (X @ (X.T @ W_0 - Y.T @ V)) / N + r_x * W_0

        for _ in range(m):
            # Randomly select an index i_t from 0 to N-1
            i_t = np.random.randint(N)
            x_i_t = X[:, i_t]  # Column vector of shape (n_features,)
            #pdb.set_trace()
            # Compute the inner product scalar_vector (shape: k,)
            scalar_vector = x_i_t.T @ (W_t - W_0)  # x_i_t' * (W_t - W_0)

            # Compute the outer product term1 (shape: n_features x k)
            term1 = np.outer(x_i_t, scalar_vector)

            # Update W_t
            W_t -= eta * (term1 + r_x * (W_t - W_0) + batch_grad)
            #print(W_t)

        # Update U_j after inner loop
        U_j = W_t

        # For testing purposes (optional):
        # Uncomment the following lines to check convergence diagnostics
        # print(r_x * np.linalg.norm(U_j)**2 + np.linalg.norm(U_j.T @ X / N - V.T @ Y / N)**2)
        # C_x = X @ X.T / N + r_x * np.eye(n_features)
        # print(np.linalg.norm(U_j - np.linalg.solve(C_x, X @ (Y.T @ V) / N)))
        # print(np.trace(0.5 * (U_j.T @ X) @ (X.T @ U_j) / N - (U_j.T @ X) @ (Y.T @ V) / N))

    return U_j



'''    
def SVRG_Factorization(U_j, V, X, Y, r_x, M):
    N = X.shape[1]
    A = X @ X.T /N /r_x
    norm_A = np.linalg.norm(A, ord='fro')
    print("norm A is ", norm_A)
    U_1 = X @ Y.T /N @ V
    U = X @ Y.T /N @ V
    for i in range(M):
        U=A @ U
        U_1 = U_1+(-1)**(i+1) * U
    return U_1/r_x
'''    
    
def SVRG_Factorization(U_j, V, X, Y, r_x, m):
    N = X.shape[1]
    
    U_0 = Y.T @ V   #Y pass 1
    U_1 = X @ U_0 /N #X pass 1
    U_2 = U_1
    
    for i in range(m):
        U_1 = X.T @ U_1  #X pass 1
        U_1 = X @ U_1/N / r_x  #X pass 1
        U_2 = U_2+(-1)**(i+1) * U_1
    return U_2/r_x

    