import os
os.environ["OMP_NUM_THREADS"] = "5"
import logging

import numpy as np
import argparse
import tensorly as tl
import pandas as pd
import logging
import time

from alg_approx_varma import BCD
from help_func import get_L
from tensorOp import tensor_op

import multiprocessing as mp

def one_step_forecast(t,y_normed,args,N):
    start_time = time.time()
    y = y_normed[:,:(t+1)]
    y = y[:,:-1]
    x = np.reshape(np.flip(y,axis=1),(-1,1),order='F') # vectorized y, yt to y1
    
    r = 1
    s = 0
    lmbd = [0.5]
    gamma = []
    theta = []
    L = get_L(lmbd, gamma, theta, r, s,  t, 0)
    G_init = np.zeros((N,N,1))
    A = tl.tenalg.mode_dot(G_init, L, 2)  # N x N x T tensor
    lmbd_est, gamma_est, theta_est, G_est, Loss, flag_maxiter, iter_no,residual = \
                                BCD(y, 0, r, s, 0.003, 5,
                                    args.lr, y[:,:t-2], np.array(lmbd), np.array(gamma),
                                    np.array(theta), A, G_init, L,
                                    stop_thres = 1e-2, stop_method = 'SepEst')
    L_est = get_L(lmbd_est, gamma_est, theta_est, r, s, t, 0)
    A_est = tl.tenalg.mode_dot(G_est, L_est, 2)  # N x N x T tensor
    y_forecast = tensor_op.unfold(A_est,1).numpy() @ x
    end_time = time.time()
    residual = residual.T
    residual_centered = residual - np.mean(residual, axis=0)
    cov_matrix = np.dot(residual_centered.T, residual_centered) / (t - 1)
    return np.squeeze(y_forecast),cov_matrix,end_time-start_time

def main():
    # model setting
    step_size = 1e-3

    # hyper parameters default values
    stop_thres = 1e-4

    parser = argparse.ArgumentParser()
    # ours
    parser.add_argument('--a',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--b',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--lmda',type=float,default=1e-1,help='hyperparameter in gradient descent')
    parser.add_argument('--thresholding_option',type=str,default='hard',choices=['hard','soft','none'],
                        help='choose hard thresholding or soft thresholding or no thresholding')
    parser.add_argument('--A_init',type=str,choices=['spec','zero'],default='zero')
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--max_iter',type=int,default=5000)
    # ours & wd
    parser.add_argument('--r1',type=int,default=3,help='rank at mode 1')
    parser.add_argument('--r2',type=int,default=3,help='rank at mode 2')
    # ar & var & di & bvar & wd
    parser.add_argument('--ar_p',type=int,default=1,help='ar order')
    # dfm & di
    parser.add_argument('--k_factor',type=int,default=1,help='number of factors')
    parser.add_argument('--factor_order',type=int,default=1,help='ar order for factors')
    # basu
    parser.add_argument('--VARpen',type=str,choices=['HLag','L1'])
    parser.add_argument('--VARMApen',type=str,choices=['HLag','L1'])
    # bvar
    parser.add_argument('--prior',type=str,choices=['minn','ncp'])
    parser.add_argument('--data_path',type=str,help='path to data file')
    parser.add_argument('--ttratio',type=float,default=0.8,help='ratio of train set')
    parser.add_argument('--exp_name',type=str,required=True,help='folder name to store the result')

    args = parser.parse_args()

    np.random.seed(0)

    
    df = pd.read_csv(args.data_path,header=None)
    raw = df.to_numpy().astype('float64').T
    T,N = raw.shape
    y_normed = raw.T
    ttratio = args.ttratio
    trainsz = int(ttratio * T)
    testsz = T - trainsz
    start_ind = trainsz

    # rolling forecast
    forecast_err = np.zeros((T-start_ind,2))
    y_hat = np.zeros((T-start_ind,N))
    y_true = np.zeros((T-start_ind,N))
    # true_A = pre_G = None
    tar_norm = 0
    Loss_list = np.zeros((T-start_ind))
    log_likelihood = np.zeros((T-start_ind))
    runtime = np.zeros((T-start_ind))

    param_list = []
    for t in range(start_ind,T):
        param_list.append((t,y_normed,args,N))
    num_workers = 5
    with mp.Pool(processes=num_workers) as pool:
        # Apply the compute function to each element in the data list
        results = pool.starmap(one_step_forecast, param_list)

    for t in range(start_ind,T):
        y = y_normed[:,:(t+1)].T
        y_target = y[-1,:]
        y_forecast = results[t-start_ind][0]
        y_hat[t-start_ind,:] = y_forecast
        y_true[t-start_ind,:] = y_target
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        cov_matrix = results[t-start_ind][1]
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])
        runtime[t-start_ind] = results[t-start_ind][2]
    
    np.save(f'result/{args.exp_name}/Y_hat_approx_varma_p0_r1_s0.npy',y_hat)


if __name__ == "__main__":
   main()
