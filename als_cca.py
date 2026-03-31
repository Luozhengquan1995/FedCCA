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
from utils import GS2_gen, GS_gen, SVRG_k, SVRG_Factorization

def als_cca(X, Y, Ux, Uy, innerIter, outerIter, r, evalPoints,
            U_truth, Ux_truth, Uy_truth, D_truth, fmax, factorization, BannerIter):
    """
    Alternating Least-Squares CCA implementation.

    Parameters:
    - X: ndarray
        Data matrix X (features x samples)
    - Y: ndarray
        Data matrix Y (features x samples)
    - Ux: ndarray
        Initial estimate of Ux (features x k)
    - Uy: ndarray
        Initial estimate of Uy (features x k)
    - innerIter: int
        Number of inner iterations for SVRG
    - outerIter: int
        Number of outer iterations
    - r: float
        Regularization parameter
    - evalPoints: list or array
        Iterations at which to evaluate and record metrics
    - U_truth: ndarray
        Ground truth U (for evaluation)
    - Ux_truth: ndarray
        Ground truth Ux (for evaluation)
    - Uy_truth: ndarray
        Ground truth Uy (for evaluation)
    - D_truth: ndarray
        Ground truth D (for evaluation)
    - fmax: float
        Maximum objective function value (for normalization)

    Returns:
    - U: ndarray
        Final U matrix
    - Ux: ndarray
        Final Ux matrix
    - Uy: ndarray
        Final Uy matrix
    - D: ndarray
        Final D matrix
    - fea, res, val_err, vec_err, obj_err: ndarray
        Arrays recording evaluation metrics over iterations
    - fea_x, res_x, val_err_x, vec_err_x, obj_err_x: ndarray
        Evaluation metrics for Ux
    - fea_y, res_y, val_err_y, vec_err_y, obj_err_y: ndarray
        Evaluation metrics for Uy
    - time_elapsed: ndarray
        Cumulative time elapsed at each evaluation point
    - num_pass: ndarray
        Number of passes over data at each evaluation point
    """
    # Save and set random seed for reproducibility
    rng_state = np.random.get_state()
    np.random.seed(2018)

    # Initialize arrays to store metrics
    num_eval_points = len(evalPoints)
    fea = np.zeros(num_eval_points)
    res = np.zeros(num_eval_points)
    val_err = np.zeros(num_eval_points)
    vec_err = np.zeros(num_eval_points)
    obj_err = np.zeros(num_eval_points)

    fea_x = np.zeros(num_eval_points)
    res_x = np.zeros(num_eval_points)
    val_err_x = np.zeros(num_eval_points)
    vec_err_x = np.zeros(num_eval_points)
    obj_err_x = np.zeros(num_eval_points)

    fea_y = np.zeros(num_eval_points)
    res_y = np.zeros(num_eval_points)
    val_err_y = np.zeros(num_eval_points)
    vec_err_y = np.zeros(num_eval_points)
    obj_err_y = np.zeros(num_eval_points)

    time_elapsed = np.zeros(num_eval_points)
    num_pass = np.zeros(num_eval_points)

    # Get dimensions
    k = Ux.shape[1]
    N = X.shape[1]

    # For evaluation purposes, compute covariance matrices
    Cxy = X @ Y.T
    Cxx = X @ X.T + r * np.eye(X.shape[0])
    Cyy = Y @ Y.T + r * np.eye(Y.shape[0])
    A = np.block([
        [np.zeros((X.shape[0], X.shape[0])), Cxy],
        [Cxy.T, np.zeros((Y.shape[0], Y.shape[0]))]
    ])
    B = np.block([
        [Cxx, np.zeros((Cxx.shape[0], Cyy.shape[1]))],
        [np.zeros((Cyy.shape[0], Cxx.shape[1])), Cyy]
    ])

    # Initialize parameters for SVRG
    M = innerIter
    m = N
    eta = 1
    eta_x = eta / np.max(np.sum((X * np.sqrt(N)) ** 2, axis=0))
    eta_y = eta / np.max(np.sum((Y * np.sqrt(N)) ** 2, axis=0))

    # Initialize counters
    t = 0
    numPass = 1
    cumulativeTime = 0

    # Initial evaluation if iteration 0 is in evalPoints
    i = 0
    if i in evalPoints:
        # Evaluate metrics
        Uxx, Uyy = GS2_gen(Ux, Uy, X, Y, r)
        U = np.vstack((Uxx, Uyy))
        sv = np.linalg.svd(U_truth.T @ B @ U, compute_uv=False)
        D = np.linalg.solve(U.T @ B @ U, U.T @ A @ U)
        D = np.diag(np.diag(D))
        fea[t] = np.linalg.norm(np.eye(U.shape[1]) - U.T @ B @ U, ord='fro')
        res[t] = np.linalg.norm(A @ U - B @ U @ D, ord=1) / U.shape[0]
        val_err[t] = np.linalg.norm(np.diag(D_truth) - np.sort(np.diag(D))[::-1], ord=np.inf)
        vec_err[t] = abs(1 - min(sv) ** 2)
        obj_err[t] = abs(np.trace(np.abs(U.T @ A @ U)) / fmax - 1)

        svx = np.linalg.svd(Ux_truth.T @ Cxx @ Ux, compute_uv=False)
        Dx = np.linalg.solve(Ux.T @ Cxx @ Ux, Ux.T @ Cxy @ Uy)
        Dx = np.diag(np.diag(Dx))
        fea_x[t] = np.linalg.norm(np.eye(Ux.shape[1]) - Ux.T @ Cxx @ Ux, ord='fro')
        res_x[t] = np.linalg.norm(Cxy @ Uy - Cxx @ Ux @ Dx, ord=1) / Ux.shape[0]
        val_err_x[t] = np.linalg.norm(np.diag(D_truth) - np.sort(np.diag(Dx))[::-1], ord=np.inf)
        vec_err_x[t] = abs(1 - min(svx) ** 2)
        obj_err_x[t] = abs(np.trace(np.abs(Ux.T @ Cxy @ Uy)) / fmax - 1)

        svy = np.linalg.svd(Uy_truth.T @ Cyy @ Uy, compute_uv=False)
        Dy = np.linalg.solve(Uy.T @ Cyy @ Uy, Uy.T @ Cxy.T @ Ux)
        Dy = np.diag(np.diag(Dy))
        fea_y[t] = np.linalg.norm(np.eye(Uy.shape[1]) - Uy.T @ Cyy @ Uy, ord='fro')
        res_y[t] = np.linalg.norm(Cxy.T @ Ux - Cyy @ Uy @ Dy, ord=1) / Uy.shape[0]
        val_err_y[t] = np.linalg.norm(np.diag(D_truth) - np.sort(np.diag(Dy))[::-1], ord=np.inf)
        vec_err_y[t] = abs(1 - min(svy) ** 2)
        obj_err_y[t] = abs(np.trace(np.abs(Uy.T @ Cxy.T @ Ux)) / fmax - 1)

        time_elapsed[t] = cumulativeTime
        num_pass[t] = numPass
        t += 1
        print(f'Finish {i}th iteration')

    # Main optimization loop
    for i in range(1, outerIter + 1):
        tStart = time.time()

        # Update Ux and Uy using SVRG_k (assumed to be implemented)
        if not factorization:
            Ux_tilde = SVRG_k(Ux, Uy, X * np.sqrt(N), Y * np.sqrt(N), r, M, m, eta_x)   # 1+M*(2+m/N) passes over X and 1+M passes over Y
            Uy_tilde = SVRG_k(Uy, Ux, Y * np.sqrt(N), X * np.sqrt(N), r, M, m, eta_y)
        else:
            Ux_tilde = SVRG_Factorization(Ux, Uy, X * np.sqrt(N), Y * np.sqrt(N), r, BannerIter)
            Uy_tilde = SVRG_Factorization(Uy, Ux, Y * np.sqrt(N), X * np.sqrt(N), r, BannerIter)

        # Orthonormalize Ux and Uy
        Ux = GS_gen(Ux_tilde, X, r) # k^2 passes over X
        Uy = GS_gen(Uy_tilde, Y, r)

        # Update time and number of passes
        cumulativeTime += time.time() - tStart
        if not factorization:
            numPass += 2 + M * (3 + m / N) + k ** 2
        else:
            numPass += 1+2*BannerIter+k ** 2

        if i in evalPoints:
            # Evaluate metrics
            Uxx, Uyy = GS2_gen(Ux, Uy, X, Y, r)
            U = np.vstack((Uxx, Uyy))
            sv = np.linalg.svd(U_truth.T @ B @ U, compute_uv=False)
            D = np.linalg.solve(U.T @ B @ U, U.T @ A @ U)
            D = np.diag(np.diag(D))
            fea[t] = np.linalg.norm(np.eye(U.shape[1]) - U.T @ B @ U, ord='fro')
            res[t] = np.linalg.norm(A @ U - B @ U @ D, ord=1) / U.shape[0]
            val_err[t] = np.linalg.norm(np.diag(D_truth) - np.sort(np.diag(D))[::-1], ord=np.inf)
            vec_err[t] = abs(1 - min(sv) ** 2)
            obj_err[t] = abs(np.trace(np.abs(U.T @ A @ U)) / fmax - 1)

            svx = np.linalg.svd(Ux_truth.T @ Cxx @ Ux, compute_uv=False)
            Dx = np.linalg.solve(Ux.T @ Cxx @ Ux, Ux.T @ Cxy @ Uy)
            Dx = np.diag(np.diag(Dx))
            fea_x[t] = np.linalg.norm(np.eye(Ux.shape[1]) - Ux.T @ Cxx @ Ux, ord='fro')
            res_x[t] = np.linalg.norm(Cxy @ Uy - Cxx @ Ux @ Dx, ord=1) / Ux.shape[0]
            val_err_x[t] = np.linalg.norm(np.diag(D_truth) - np.sort(np.diag(Dx))[::-1], ord=np.inf)
            vec_err_x[t] = abs(1 - min(svx) ** 2)
            obj_err_x[t] = abs(np.trace(np.abs(Ux.T @ Cxy @ Uy)) / fmax - 1)

            svy = np.linalg.svd(Uy_truth.T @ Cyy @ Uy, compute_uv=False)
            Dy = np.linalg.solve(Uy.T @ Cyy @ Uy, Uy.T @ Cxy.T @ Ux)
            Dy = np.diag(np.diag(Dy))
            fea_y[t] = np.linalg.norm(np.eye(Uy.shape[1]) - Uy.T @ Cyy @ Uy, ord='fro')
            res_y[t] = np.linalg.norm(Cxy.T @ Ux - Cyy @ Uy @ Dy, ord=1) / Uy.shape[0]
            val_err_y[t] = np.linalg.norm(np.diag(D_truth) - np.sort(np.diag(Dy))[::-1], ord=np.inf)
            vec_err_y[t] = abs(1 - min(svy) ** 2)
            obj_err_y[t] = abs(np.trace(np.abs(Uy.T @ Cxy.T @ Ux)) / fmax - 1)

            time_elapsed[t] = cumulativeTime
            num_pass[t] = numPass
            t += 1
            print(f'Finish {i}th iteration')

    # Final computation of U and D
    U1, U2 = GS2_gen(Ux, Uy, X, Y, r)
    U = np.vstack((U1, U2))
    D = np.linalg.solve(U.T @ B @ U, U.T @ A @ U)
    D = np.diag(np.diag(D))

    # Restore the original random state
    np.random.set_state(rng_state)

    return (fea, res, val_err, vec_err, obj_err,
            fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
            fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
            time_elapsed, num_pass)