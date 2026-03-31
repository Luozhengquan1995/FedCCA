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
from scipy.linalg import  solve
import scipy.io as sio

from scipy.io import loadmat, savemat
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs, svds
import os
import sys
from client import Client
from server import Server

# 获取当前目录
current_directory = os.getcwd()

# 将当前目录添加到系统路径中
if current_directory not in sys.path:
    sys.path.append(current_directory)
from utils import GS2_gen, GS_gen, SVRG_k, SVRG_Factorization

#from als_cca import als_cca
from tals2_cca_fl import tals2_cca_fl
from tals2_cca import tals2_cca
#from asi_cca3 import asi_cca3
#from asi_cca4_minbeta import asi_cca4_minbeta
from utils_fl import GS2_gen_fl, GS_gen_fl, SVRG_Factorization_fl_x, SVRG_Factorization_fl_y


def main(): 
    # List of dataset names
    data_names = [
        'mmill',
        'JW11',
        'mnist_training',
    ]
    
    # Initialize k values
    kk = np.ones(len(data_names), dtype=int) * 10
    kk[0] = 4  # Set the first k value to 4
    
    # Loop over the first three datasets
    for idx in range(3):
        k = kk[idx]
        data_name = data_names[idx]
        print("Data name: ",data_name)
        # Load the data
        data = loadmat(f'./data/{data_name}.mat')
        X = data['X']
        Y = data['Y']
        #pdb.set_trace()
        # Ensure that X and Y have dimensions (features, samples)
        if X.shape[0] > X.shape[1]:
            X = X.T
        if Y.shape[0] > Y.shape[1]:
            Y = Y.T
    
        # Plotting parameters
        fontSize_label = 20
        fontSize_title = 20
        fontSize_legend = 18
        lineWidth = 2
        markSize = 6
        
        
        # Initialize figures
        fig_hObj, ax_hObj = plt.subplots()
        ax_hObj.set_xlabel('time (seconds)', fontsize=fontSize_label)
        ax_hObj.set_ylabel(r'$(f^{*}-f)/f^{*}$', fontsize=fontSize_label)
        ax_hObj.set_yscale('log')
        ax_hObj.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hObj.grid(True)
        
        # Repeat for other figures
        fig_hAngle, ax_hAngle = plt.subplots()
        ax_hAngle.set_xlabel('time (seconds)', fontsize=fontSize_label)
        ax_hAngle.set_ylabel(r'$\sin^{2}\theta$', fontsize=fontSize_label)
        ax_hAngle.set_yscale('log')
        ax_hAngle.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hAngle.grid(True)
        
        fig_hxAngle, ax_hxAngle = plt.subplots()
        ax_hxAngle.set_xlabel('time (seconds)', fontsize=fontSize_label)
        ax_hxAngle.set_ylabel(r'$\sin^{2}\theta_{u}$', fontsize=fontSize_label)
        ax_hxAngle.set_yscale('log')
        ax_hxAngle.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hxAngle.grid(True)
        
        fig_hyAngle, ax_hyAngle = plt.subplots()
        ax_hyAngle.set_xlabel('time (seconds)', fontsize=fontSize_label)
        ax_hyAngle.set_ylabel(r'$\sin^{2}\theta_{v}$', fontsize=fontSize_label)
        ax_hyAngle.set_yscale('log')
        ax_hyAngle.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hyAngle.grid(True)
        
        fig_hObj2, ax_hObj2 = plt.subplots()
        ax_hObj2.set_xlabel('# Passes', fontsize=fontSize_label)
        ax_hObj2.set_ylabel(r'$(f^{*}-f)/f^{*}$', fontsize=fontSize_label)
        ax_hObj2.set_yscale('log')
        ax_hObj2.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hObj2.grid(True)
        
        fig_hAngle2, ax_hAngle2 = plt.subplots()
        ax_hAngle2.set_xlabel('# Passes', fontsize=fontSize_label)
        ax_hAngle2.set_ylabel(r'$\sin^{2}\theta$', fontsize=fontSize_label)
        ax_hAngle2.set_yscale('log')
        ax_hAngle2.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hAngle2.grid(True)
        
        fig_hxAngle2, ax_hxAngle2 = plt.subplots()
        ax_hxAngle2.set_xlabel('# Passes', fontsize=fontSize_label)
        ax_hxAngle2.set_ylabel(r'$\sin^{2}\theta_{u}$', fontsize=fontSize_label)
        ax_hxAngle2.set_yscale('log')
        ax_hxAngle2.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hxAngle2.grid(True)
        
        fig_hyAngle2, ax_hyAngle2 = plt.subplots()
        ax_hyAngle2.set_xlabel('# Passes', fontsize=fontSize_label)
        ax_hyAngle2.set_ylabel(r'$\sin^{2}\theta_{v}$', fontsize=fontSize_label)
        ax_hyAngle2.set_yscale('log')
        ax_hyAngle2.text(1, 1, f'{data_name}: k={k}', fontsize=fontSize_title)
        ax_hyAngle2.grid(True)
        
        
        
        #pdb.set_trace()
        # Initialize legend strings and curves
        lgStr = []
        hObjCurve = []
        hAngleCurve = []
        hxAngleCurve = []
        hyAngleCurve = []
        hObj2Curve = []
        hAngle2Curve = []
        hxAngle2Curve = []
        hyAngle2Curve = []
        
        marker = ['*', 's', 'd', 'h', '^', 'v', '>', '<', '+']
        color = ['r', 'g', 'b', 'c', 'm', 'k']
        c = 1  # Counter for marker and color selection
        
        # Data centering
        X = X - np.mean(X, axis=1, keepdims=True)
        Y = Y - np.mean(Y, axis=1, keepdims=True)
        X = X / np.sqrt(X.shape[1])
        Y = Y / np.sqrt(Y.shape[1])
        r = 0.1  # Regularization parameter
        
        if 1 or 'Ux_truth' not in data or data['Ux_truth'].shape[1] < k:
            # Define covariance functions
            def AutoCov(Z):
                return Z @ Z.T
            
            def CrossCov(Z1, Z2):
                return Z1 @ Z2.T
            
            Cxy = CrossCov(X, Y)
            Cxx = AutoCov(X)
            Cyy = AutoCov(Y)
            
            # Build matrices A and B for generalized eigenvalue problem
            A = np.block([
                [np.zeros((X.shape[0], X.shape[0])), Cxy],
                [Cxy.T, np.zeros((Y.shape[0], Y.shape[0]))]
            ])
            B = np.block([
                [Cxx + r * np.eye(Cxx.shape[0]), np.zeros((X.shape[0], Y.shape[0]))],
                [np.zeros((Y.shape[0], X.shape[0])), Cyy + r * np.eye(Cyy.shape[0])]
            ])
            
            # 求解广义特征值问题
            D_truth, U_truth = eigs(A, k=k, M=B, which='LR')
            
            # 对特征值和特征向量进行排序（降序）
            idx = np.argsort(-np.real(D_truth))
            D_truth = D_truth[idx]
            U_truth = U_truth[:, idx]
            
            # 将特征值转换为对角矩阵
            D_truth = np.diag(D_truth)
            
            # 对正则化协方差矩阵进行特征分解
            SX, UX = eigh(Cxx + r * np.eye(Cxx.shape[0]))
            SY, UY = eigh(Cyy + r * np.eye(Cyy.shape[0]))
            
            # **移除**反转操作，以保持与 MATLAB 一致
            # SX = SX[::-1]
            # UX = UX[:, ::-1]
            # SY = SY[::-1]
            # UY = UY[:, ::-1]
            
            # 计算归一化的逆平方根矩阵
            Cxx_nsqrt = UX @ np.diag(SX ** (-0.5)) @ UX.T
            Cyy_nsqrt = UY @ np.diag(SY ** (-0.5)) @ UY.T
            
            # 计算矩阵 T
            T = Cxx_nsqrt @ Cxy @ Cyy_nsqrt
            
            # 计算前 k 个奇异值和奇异向量
            Ux_truth, S_values, Uy_truth_T = svds(T, k=k)
            Uy_truth = Uy_truth_T.T  # 将右奇异向量转置
            
            # 对奇异值和奇异向量进行排序（降序）
            idx = np.argsort(-S_values)
            S_values = S_values[idx]
            Ux_truth = Ux_truth[:, idx]
            Uy_truth = Uy_truth[:, idx]
            
            # 调整奇异向量
            Ux_truth = Cxx_nsqrt @ Ux_truth
            Uy_truth = Cyy_nsqrt @ Uy_truth
            
            # 计算用于验证的奇异值
            sv_matrix = U_truth.T @ B @ np.vstack([Ux_truth, Uy_truth]) / np.sqrt(2)
            sv = np.linalg.svd(sv_matrix, compute_uv=False)
            
            # 将奇异值转换为对角矩阵，以与 D_truth 一致
            S = np.diag(S_values)
            
            # 输出结果
            print(np.array([np.linalg.norm(D_truth - S, ord=np.inf), abs(1 - min(sv) ** 2)]))
            
            # Final objective function value
            fmax = np.sum(S_values)
            
            # Save computed ground truth back to the .mat file
            data['U_truth'] = U_truth
            data['D_truth'] = D_truth
            data['Ux_truth'] = Ux_truth
            data['Uy_truth'] = Uy_truth
            data['S'] = S_values
            savemat(f'./data/{data_name}.mat', data)
        else:
            U_truth = data['U_truth'][:, :k]
            D_truth = data['D_truth'][:, :k]
            Ux_truth = data['Ux_truth'][:, :k]
            Uy_truth = data['Uy_truth'][:, :k]
            S_values = data['S'][:k]
            fmax = np.sum(S_values)
        
        column_Xnorms = np.linalg.norm(X, axis=0)
        column_Ynorms = np.linalg.norm(Y, axis=0)
        
        rawalpha=0.1
        alpha = 1/(np.sqrt(0.1*rawalpha))
        max_norm = alpha*max(np.sqrt(X.shape[1])*np.max(column_Xnorms), np.sqrt(Y.shape[1])*np.max(column_Ynorms))
        X1=X/max_norm
        Y1=Y/max_norm
        #pdb.set_trace()
        
        
        
        # Check if ground truth exists; if not, compute it
        if 1 or 'Ux_truth' not in data or data['Ux_truth'].shape[1] < k:
            # Define covariance functions
            def AutoCov(Z):
                return Z @ Z.T
            
            def CrossCov(Z1, Z2):
                return Z1 @ Z2.T
            
            Cxy1 = CrossCov(X1, Y1)
            Cxx1 = AutoCov(X1)
            Cyy1 = AutoCov(Y1)
            
            # Build matrices A and B for generalized eigenvalue problem
            A1 = np.block([
                [np.zeros((X1.shape[0], X1.shape[0])), Cxy1],
                [Cxy1.T, np.zeros((Y1.shape[0], Y1.shape[0]))]
            ])
            B1 = np.block([
                [Cxx1 + r * np.eye(Cxx1.shape[0]), np.zeros((X1.shape[0], Y1.shape[0]))],
                [np.zeros((Y1.shape[0], X1.shape[0])), Cyy1 + r * np.eye(Cyy1.shape[0])]
            ])
            
            # 求解广义特征值问题
            D_truth1, U_truth1 = eigs(A1, k=k, M=B1, which='LR')
            
            # 对特征值和特征向量进行排序（降序）
            idx = np.argsort(-np.real(D_truth1))
            D_truth1 = D_truth1[idx]
            U_truth1 = U_truth1[:, idx]
            
            # 将特征值转换为对角矩阵
            D_truth1 = np.diag(D_truth1)
            
            # 对正则化协方差矩阵进行特征分解
            SX1, UX1 = eigh(Cxx1 + r * np.eye(Cxx1.shape[0]))
            SY1, UY1 = eigh(Cyy1 + r * np.eye(Cyy1.shape[0]))
            
            # **移除**反转操作，以保持与 MATLAB 一致
            # SX = SX[::-1]
            # UX = UX[:, ::-1]
            # SY = SY[::-1]
            # UY = UY[:, ::-1]
            
            # 计算归一化的逆平方根矩阵
            Cxx_nsqrt1 = UX1 @ np.diag(SX1 ** (-0.5)) @ UX1.T
            Cyy_nsqrt1 = UY1 @ np.diag(SY1 ** (-0.5)) @ UY1.T
            
            # 计算矩阵 T
            T1 = Cxx_nsqrt1 @ Cxy1 @ Cyy_nsqrt1
            
            # 计算前 k 个奇异值和奇异向量
            Ux_truth1, S_values1, Uy_truth_T1 = svds(T1, k=k)
            Uy_truth1 = Uy_truth_T1.T  # 将右奇异向量转置
            
            # 对奇异值和奇异向量进行排序（降序）
            idx1 = np.argsort(-S_values1)
            S_values1 = S_values1[idx1]
            Ux_truth1 = Ux_truth1[:, idx1]
            Uy_truth1 = Uy_truth1[:, idx1]
            
            # 调整奇异向量
            Ux_truth1 = Cxx_nsqrt1 @ Ux_truth1
            Uy_truth1 = Cyy_nsqrt1 @ Uy_truth1
            
            # 计算用于验证的奇异值
            sv_matrix1 = U_truth1.T @ B1 @ np.vstack([Ux_truth1, Uy_truth1]) / np.sqrt(2)
            sv1 = np.linalg.svd(sv_matrix1, compute_uv=False)
            
            # 将奇异值转换为对角矩阵，以与 D_truth 一致
            S1 = np.diag(S_values1)
            
            # 输出结果
            print(np.array([np.linalg.norm(D_truth1 - S1, ord=np.inf), abs(1 - min(sv1) ** 2)]))
            
            
            # Final objective function value
            fmax1 = np.sum(S_values1)
        
        
        
        
        print("Ground Truth is finished")
        # Set iterations and evaluation points
        Banner=1
        innerIter = 2
        outerIter = 60
        evalPoints = np.arange(0, outerIter + 1, 2)
        
        '''
        # Initialization
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X1, Y1, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        results = als_cca(
            X1, Y1, GS_gen(Ux, X1, r), GS_gen(Uy, Y1, r), innerIter, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax, Banner, BannerIter
        )
        
        # Unpack results
        (fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = results
        
        print("Als_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_ALSBanner.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('ALS-k')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='ALS-k-Banner')
                         
        # Set iterations and evaluation points
        Banner=0
        innerIter = 2
        outerIter = 60
        evalPoints = np.arange(0, outerIter + 1, 2)
        
        
        # Initialization
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X, Y, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        results = als_cca(
            X, Y, GS_gen(Ux, X, r), GS_gen(Uy, Y, r), innerIter, outerIter, r, evalPoints,
            U_truth, Ux_truth, Uy_truth, D_truth, fmax, Banner, BannerIter
        )
        
        # Unpack results
        (fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = results
        
        print("Als_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_ALS.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('ALS-k')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, label='ALS-k')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, label='ALS-k')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='ALS-k')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='ALS-k')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, label='ALS-k')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='ALS-k')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='ALS-k')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='ALS-k')
                         
        '''  
        # Initialization
        num_clients = 5
        client_list = []
        sigma = 0.0
        # Split X1 and Y1 IID into K parts for K clients
        X_splits = np.array_split(X1, num_clients, axis=1)
        Y_splits = np.array_split(Y1, num_clients, axis=1)
    
        for i in range(num_clients):
            XX = X_splits[i]
            YY = Y_splits[i]
            client = Client(XX, YY, idx=i, sigma=sigma)
            client_list.append(client)
    
        server = Server(client_list)
        
        BannerIter=3
        dp=False
        client_ratio=1.0
        Banner=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        Cxy2 = X1 @ Y1.T
        Cxx2 = X1 @ X1.T + r * np.eye(X1.shape[0])
        Cyy2 = Y1 @ Y1.T + r * np.eye(Y1.shape[0])
        A2 = np.block([
            [np.zeros((X1.shape[0], X1.shape[0])), Cxy2],
            [Cxy2.T, np.zeros((Y1.shape[0], Y1.shape[0]))]
        ])
        B2 = np.block([
            [Cxx2, np.zeros((Cxx2.shape[0], Cyy2.shape[1]))],
            [np.zeros((Cyy2.shape[0], Cxx2.shape[1])), Cyy2]
        ])
        eta = 1
        N = X.shape[1]
        eta_x = eta / np.max(np.sum(np.abs(X * np.sqrt(N)) ** 2, axis=0))
        eta_y = eta / np.max(np.sum(np.abs(Y * np.sqrt(N)) ** 2, axis=0))
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen_fl(Ux, Uy, r, server, dp, client_ratio=1.0)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = tals2_cca_fl(
            Cxy2, Cxx2,Cyy2,A2,B2, eta_x,eta_y, GS_gen_fl(Ux, 'X', r, server, dp, client_ratio), GS_gen_fl(Uy, 'Y', r, server, dp, client_ratio), innerIter, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, Banner, BannerIter, server, dp, client_ratio
        )
        
        print("Tals_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_TALSBanner_.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append(f'FedTALS_power_Clientnum{num_clients}')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth,alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        
        
            
        # Initialization
        num_clients = 10
        client_list = []
        sigma = 0.0
        # Split X1 and Y1 IID into K parts for K clients
        X_splits = np.array_split(X1, num_clients, axis=1)
        Y_splits = np.array_split(Y1, num_clients, axis=1)
    
        for i in range(num_clients):
            XX = X_splits[i]
            YY = Y_splits[i]
            client = Client(XX, YY, idx=i, sigma=sigma)
            client_list.append(client)
    
        server = Server(client_list)
        
        BannerIter=3
        dp=False
        client_ratio=1.0
        Banner=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        Cxy2 = X1 @ Y1.T
        Cxx2 = X1 @ X1.T + r * np.eye(X1.shape[0])
        Cyy2 = Y1 @ Y1.T + r * np.eye(Y1.shape[0])
        A2 = np.block([
            [np.zeros((X1.shape[0], X1.shape[0])), Cxy2],
            [Cxy2.T, np.zeros((Y1.shape[0], Y1.shape[0]))]
        ])
        B2 = np.block([
            [Cxx2, np.zeros((Cxx2.shape[0], Cyy2.shape[1]))],
            [np.zeros((Cyy2.shape[0], Cxx2.shape[1])), Cyy2]
        ])
        eta = 1
        N = X.shape[1]
        eta_x = eta / np.max(np.sum(np.abs(X * np.sqrt(N)) ** 2, axis=0))
        eta_y = eta / np.max(np.sum(np.abs(Y * np.sqrt(N)) ** 2, axis=0))
        # Orthonormalize initial Ux and Uy using GS2_gen function
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen_fl(Ux, Uy, r, server, dp, client_ratio=1.0)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = tals2_cca_fl(
            Cxy2, Cxx2,Cyy2,A2,B2, eta_x,eta_y, GS_gen_fl(Ux, 'X', r, server, dp, client_ratio), GS_gen_fl(Uy, 'Y', r, server, dp, client_ratio), innerIter, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, Banner, BannerIter, server, dp, client_ratio
        )
        
        print("Tals_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_TALSBanner_.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append(f'FedTALS_power_Clientnum{num_clients}')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth,alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
                              
        
        
                                      
        # Initialization
        num_clients = 20
        client_list = []
        sigma = 0.0
        # Split X1 and Y1 IID into K parts for K clients
        X_splits = np.array_split(X1, num_clients, axis=1)
        Y_splits = np.array_split(Y1, num_clients, axis=1)
    
        for i in range(num_clients):
            XX = X_splits[i]
            YY = Y_splits[i]
            client = Client(XX, YY, idx=i, sigma=sigma)
            client_list.append(client)
    
        server = Server(client_list)
        
        BannerIter=3
        dp=False
        client_ratio=1.0
        Banner=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        Cxy2 = X1 @ Y1.T
        Cxx2 = X1 @ X1.T + r * np.eye(X1.shape[0])
        Cyy2 = Y1 @ Y1.T + r * np.eye(Y1.shape[0])
        A2 = np.block([
            [np.zeros((X1.shape[0], X1.shape[0])), Cxy2],
            [Cxy2.T, np.zeros((Y1.shape[0], Y1.shape[0]))]
        ])
        B2 = np.block([
            [Cxx2, np.zeros((Cxx2.shape[0], Cyy2.shape[1]))],
            [np.zeros((Cyy2.shape[0], Cxx2.shape[1])), Cyy2]
        ])
        eta = 1
        N = X.shape[1]
        eta_x = eta / np.max(np.sum(np.abs(X * np.sqrt(N)) ** 2, axis=0))
        eta_y = eta / np.max(np.sum(np.abs(Y * np.sqrt(N)) ** 2, axis=0))
        # Orthonormalize initial Ux and Uy using GS2_gen function
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen_fl(Ux, Uy, r, server, dp, client_ratio=1.0)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = tals2_cca_fl(
           Cxy2, Cxx2,Cyy2,A2,B2, eta_x,eta_y, GS_gen_fl(Ux, 'X', r, server, dp, client_ratio), GS_gen_fl(Uy, 'Y', r, server, dp, client_ratio), innerIter, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, Banner, BannerIter, server, dp, client_ratio
        )
        
        print("Tals_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_TALSBanner_.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append(f'FedTALS_power_Clientnum{num_clients}')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth,alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
         

            
        # Initialization
        num_clients = 50
        client_list = []
        sigma = 0.0
        # Split X1 and Y1 IID into K parts for K clients
        X_splits = np.array_split(X1, num_clients, axis=1)
        Y_splits = np.array_split(Y1, num_clients, axis=1)
    
        for i in range(num_clients):
            XX = X_splits[i]
            YY = Y_splits[i]
            client = Client(XX, YY, idx=i, sigma=sigma)
            client_list.append(client)
    
        server = Server(client_list)
        
        BannerIter=3
        dp=False
        client_ratio=1.0
        Banner=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        Cxy2 = X1 @ Y1.T
        Cxx2 = X1 @ X1.T + r * np.eye(X1.shape[0])
        Cyy2 = Y1 @ Y1.T + r * np.eye(Y1.shape[0])
        A2 = np.block([
            [np.zeros((X1.shape[0], X1.shape[0])), Cxy2],
            [Cxy2.T, np.zeros((Y1.shape[0], Y1.shape[0]))]
        ])
        B2 = np.block([
            [Cxx2, np.zeros((Cxx2.shape[0], Cyy2.shape[1]))],
            [np.zeros((Cyy2.shape[0], Cxx2.shape[1])), Cyy2]
        ])
        eta = 1
        N = X.shape[1]
        eta_x = eta / np.max(np.sum(np.abs(X * np.sqrt(N)) ** 2, axis=0))
        eta_y = eta / np.max(np.sum(np.abs(Y * np.sqrt(N)) ** 2, axis=0))
        # Orthonormalize initial Ux and Uy using GS2_gen function
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen_fl(Ux, Uy, r, server, dp, client_ratio=1.0)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = tals2_cca_fl(
           Cxy2, Cxx2,Cyy2,A2,B2, eta_x,eta_y, GS_gen_fl(Ux, 'X', r, server, dp, client_ratio), GS_gen_fl(Uy, 'Y', r, server, dp, client_ratio), innerIter, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, Banner, BannerIter, server, dp, client_ratio
        )
        
        print("Tals_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_TALSBanner_.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append(f'FedTALS_power_Clientnum{num_clients}')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth,alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
                   
        # Initialization
        num_clients = 100
        client_list = []
        sigma = 0.0
        # Split X1 and Y1 IID into K parts for K clients
        X_splits = np.array_split(X1, num_clients, axis=1)
        Y_splits = np.array_split(Y1, num_clients, axis=1)
    
        for i in range(num_clients):
            XX = X_splits[i]
            YY = Y_splits[i]
            client = Client(XX, YY, idx=i, sigma=sigma)
            client_list.append(client)
    
        server = Server(client_list)
        
        BannerIter=3
        dp=False
        client_ratio=1.0
        Banner=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        Cxy2 = X1 @ Y1.T
        Cxx2 = X1 @ X1.T + r * np.eye(X1.shape[0])
        Cyy2 = Y1 @ Y1.T + r * np.eye(Y1.shape[0])
        A2 = np.block([
            [np.zeros((X1.shape[0], X1.shape[0])), Cxy2],
            [Cxy2.T, np.zeros((Y1.shape[0], Y1.shape[0]))]
        ])
        B2 = np.block([
            [Cxx2, np.zeros((Cxx2.shape[0], Cyy2.shape[1]))],
            [np.zeros((Cyy2.shape[0], Cxx2.shape[1])), Cyy2]
        ])
        eta = 1
        N = X.shape[1]
        eta_x = eta / np.max(np.sum(np.abs(X * np.sqrt(N)) ** 2, axis=0))
        eta_y = eta / np.max(np.sum(np.abs(Y * np.sqrt(N)) ** 2, axis=0))
        # Orthonormalize initial Ux and Uy using GS2_gen function
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen_fl(Ux, Uy, r, server, dp, client_ratio=1.0)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = tals2_cca_fl(
           Cxy2, Cxx2,Cyy2,A2,B2, eta_x,eta_y, GS_gen_fl(Ux, 'X', r, server, dp, client_ratio), GS_gen_fl(Uy, 'Y', r, server, dp, client_ratio), innerIter, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, Banner, BannerIter, server, dp, client_ratio
        )
        
        print("Tals_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_TALSBanner_.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append(f'FedTALS_power_Clientnum{num_clients}')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth,alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label=f'FedTALS_power_Clientnum{num_clients}')
                               
        # Initialization
        Banner=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X1, Y1, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = tals2_cca(
            X1, Y1, GS_gen(Ux, X1, r), GS_gen(Uy, Y1, r), innerIter, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, Banner, BannerIter
        )
        
        print("Tals_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_TALSBanner_.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('TALSBanner')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, alpha=0.5,label='TALS_power')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS_power')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS_power')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS_power')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS_power')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS_power')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS_power')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS_power')
                   
        # Initialization
        Banner=0
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X, Y, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = tals2_cca(
            X, Y, GS_gen(Ux, X, r), GS_gen(Uy, Y, r), innerIter, outerIter, r, evalPoints,
            U_truth, Ux_truth, Uy_truth, D_truth, fmax, Banner, BannerIter
        )
        
        print("Tals_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_TALS.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('TALS')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, alpha=0.5, label='TALS')
        
        '''
        # Initialization
        Banner=1
        burnings=4
        coef=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X1, Y1, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = asi_cca3(
            X1, Y1, GS_gen(Ux, X1, r), GS_gen(Uy, Y1, r), innerIter, burnings, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, coef, Banner, BannerIter
        )
        
        print("FastTALS_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_FastTALSBanner.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('FastTALS_T_04Banner')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='FastTALSBanner')
        
        # Initialization
        Banner=0
        burnings=4
        coef=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X, Y, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = asi_cca3(
            X, Y, GS_gen(Ux, X, r), GS_gen(Uy, Y, r), innerIter, burnings, outerIter, r, evalPoints,
            U_truth, Ux_truth, Uy_truth, D_truth, fmax, coef, Banner, BannerIter
        )
        
        print("FastTALS_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_FastTALS.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('FastTALS_T_04')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, label='FastTALS')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, label='FastTALS')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='FastTALS')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='FastTALS')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, label='FastTALS')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='FastTALS')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='FastTALS')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='FastTALS')
        
        # Initialization
        Banner=1
        burnings=0
        coef=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X1, Y1, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = asi_cca4_minbeta(
            X1, Y1, GS_gen(Ux, X1, r), GS_gen(Uy, Y1, r), innerIter, burnings, outerIter, r, evalPoints,
            U_truth1, Ux_truth1, Uy_truth1, D_truth1, fmax1, coef, Banner, BannerIter
        )
        
        print("AdaFastTALS_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_AdaFastTALSBanner.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('AdaFastTALSBanner')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='AdaFastTALSBanner')
                         
        # Initialization
        Banner=0
        burnings=0
        coef=1
        rng_state = np.random.get_state()
        np.random.seed(2018)
        Ux = np.random.randn(X.shape[0], k)
        Uy = np.random.randn(Y.shape[0], k)
        np.random.set_state(rng_state)
        
        # Orthonormalize initial Ux and Uy using GS2_gen function
        Ux, Uy = GS2_gen(Ux, Uy, X, Y, r)
        
        # Call the alternating least-squares CCA algorithm
        # Assumed to return the necessary outputs
        (_, _, _, _,
         fea, res, val_err, vec_err, obj_err,
         fea_x, res_x, val_err_x, vec_err_x, obj_err_x,
         fea_y, res_y, val_err_y, vec_err_y, obj_err_y,
         time_elapsed, num_pass) = asi_cca4_minbeta(
            X, Y, GS_gen(Ux, X, r), GS_gen(Uy, Y, r), innerIter, burnings, outerIter, r, evalPoints,
            U_truth, Ux_truth, Uy_truth, D_truth, fmax, coef, Banner, BannerIter
        )
        
        print("AdaFastTALS_CCA is finished")
        # Save results to a .mat file
        savemat(f'./result_mat/{data_name}_k{k}_AdaFastTALS.mat', {
            'fea': fea,
            'res': res,
            'val_err': val_err,
            'vec_err': vec_err,
            'obj_err': obj_err,
            'fea_x': fea_x,
            'res_x': res_x,
            'val_err_x': val_err_x,
            'vec_err_x': vec_err_x,
            'obj_err_x': obj_err_x,
            'fea_y': fea_y,
            'res_y': res_y,
            'val_err_y': val_err_y,
            'vec_err_y': vec_err_y,
            'obj_err_y': obj_err_y,
            'time_elapsed': time_elapsed,
            'num_pass': num_pass,
        }, do_compression=True)
        
        
        # Update legend strings and plotting variables
        lgStr.append('AdaFastTALS')
        markerStr = '-' + marker[(c - 1) % len(marker)] + color[(c - 1) % len(color)]
        c += 1
        
        # Plot results on the figures
        ax_hObj.plot(time_elapsed, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                     markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        ax_hAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                       markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        ax_hxAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        ax_hyAngle.plot(time_elapsed, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        ax_hObj2.plot(num_pass, np.maximum(np.finfo(float).eps, obj_err), markerStr,
                      markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        ax_hAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err), markerStr,
                        markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        ax_hxAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_x), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        ax_hyAngle2.plot(num_pass, np.maximum(np.finfo(float).eps, vec_err_y), markerStr,
                         markersize=markSize, linewidth=lineWidth, label='AdaFastTALS')
        '''
        # Add legends to the plots
        ax_hObj.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontSize_legend)
        ax_hAngle.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontSize_legend)
        ax_hxAngle.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontSize_legend)
        ax_hyAngle.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontSize_legend)
        ax_hObj2.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontSize_legend)
        ax_hAngle2.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontSize_legend)
        ax_hxAngle2.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontSize_legend)
        ax_hyAngle2.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontSize_legend)
        
        # Repeat similar steps for other algorithms as in the MATLAB code
        # (e.g., si_cca_uai, tals2_cca, asi_cca3, etc.)
        # Each algorithm would be called with appropriate parameters
        # Results would be saved and plotted similarly
        
        # Save the figures
        fig_hObj.savefig(f'./output/image_clientnum/{data_name}k{k}_obj.png', dpi=300, bbox_inches='tight')
        fig_hAngle.savefig(f'./output/image_clientnum/{data_name}k{k}_angle.png', dpi=300, bbox_inches='tight')
        fig_hxAngle.savefig(f'./output/image_clientnum/{data_name}k{k}_angle_x.png', dpi=300, bbox_inches='tight')
        fig_hyAngle.savefig(f'./output/image_clientnum/{data_name}k{k}_angle_y.png', dpi=300, bbox_inches='tight')
        fig_hObj2.savefig(f'./output/image_clientnum/{data_name}k{k}_obj_passes.png', dpi=300, bbox_inches='tight')
        fig_hAngle2.savefig(f'./output/image_clientnum/{data_name}k{k}_angle_passes.png', dpi=300, bbox_inches='tight')
        fig_hxAngle2.savefig(f'./output/image_clientnum/{data_name}k{k}_angle_x_passes.png', dpi=300, bbox_inches='tight')
        fig_hyAngle2.savefig(f'./output/image_clientnum/{data_name}k{k}_angle_y_passes.png', dpi=300, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig_hObj)
        plt.close(fig_hAngle)
        plt.close(fig_hxAngle)
        plt.close(fig_hyAngle)
        plt.close(fig_hObj2)
        plt.close(fig_hAngle2)
        plt.close(fig_hxAngle2)
        plt.close(fig_hyAngle2)
        
        print("Image and mat saving is finished")

if __name__ == '__main__':
    main()