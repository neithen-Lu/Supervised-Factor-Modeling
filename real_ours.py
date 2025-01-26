import os
os.environ["OMP_NUM_THREADS"] = "5"
import logging

import numpy as np
import argparse
import tensorly as tl
import pandas as pd
import logging
import time
import scipy

from alg_real import train_epoch

import multiprocessing as mp


def one_step_forecast(t,y_normed,args,N):
    # print(t,y_normed[:,t])
    start_time = time.time()
    y = y_normed[:,:(t+1)].T
    X = np.zeros((t+1-args.T0,N,args.T0))
    for i in range(args.T0):
        X[:,:,i] = y[(args.T0-1-i):t-i,:]
    y = y[args.T0:,:]
    y = y[:-1,:]
    x = X[-1,:,:]
    X = X[:-1,:,:]
    
    if args.A_init == 'zero':
        A_init = np.zeros((N,N,args.T0))
    else:
        A_init = None
    A_est,U1,U2,norm_0_idx,_,_ = train_epoch(y=y,X=X,P=args.T0,r1=args.r1,r2=args.r2,a=args.a,b=args.b,s=args.s,lmda=args.lmda,thresholding_option=args.thresholding_option,max_iter=args.max_iter,step_size=args.lr,A_init=A_init,min_loss=1e-4)
    nonzero_idx = list(np.arange(0,args.T0))
    for i in norm_0_idx:
        nonzero_idx.remove(i)
    y_forecast = np.einsum('NP,iNP->i',x,A_est)
    end_time = time.time()
    return y_forecast,nonzero_idx,U1,U2,end_time-start_time

def main():
    # model setting
    step_size = 1e-3

    # hyper parameters default values
    stop_thres = 1e-4

    parser = argparse.ArgumentParser()
    # ours
    parser.add_argument('--a',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--b',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--T0', type=int,default=10, help='T0')
    parser.add_argument('--s',type=int,default=4,help='sparsity level')
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
    trainsz = int(T*ttratio)
    testsz = T - trainsz
    start_ind = trainsz

    # rolling forecast
    forecast_err = np.zeros((T-start_ind,2))
    y_hat = np.zeros((T-start_ind,N))
    nonzero_idx = np.zeros((T-start_ind,args.s))
    y_true = np.zeros((T-start_ind,N))
    # true_A = pre_G = None
    tar_norm = 0
    time = np.zeros((T-start_ind))

    param_list = []
    for t in range(start_ind,T):
        param_list.append((t,y_normed,args,N))
    num_workers = 5
    with mp.Pool(processes=num_workers) as pool:
        # Apply the compute function to each element in the data list
        results = pool.starmap(one_step_forecast, param_list)

    for t in range(start_ind,T):
        y = y_normed[:,:(t+1)].T
        X = np.zeros((t+1-args.T0,N,args.T0))
        for i in range(args.T0):
            X[:,:,i] = y[(args.T0-1-i):t-i,:]
        y = y[args.T0:,:]
        y_target = y[-1,:]
        y = y[:-1,:]
        x = X[-1,:,:]
        X = X[:-1,:,:]
        y_forecast = results[t-start_ind][0]
        nonzero_idx[t-start_ind,:] = results[t-start_ind][1]
        U1 = results[t-start_ind][2]
        U2 = results[t-start_ind][3]
        time[t-start_ind] =results[t-start_ind][4]
        y_hat[t-start_ind,:] = y_forecast
        y_true[t-start_ind,:] = y_target
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])


    np.save(f'result/{args.exp_name}/Y_hat_ours_T0{args.T0}_s{args.s}_r1{args.r1}_r2{args.r2}_lr{args.lr}_iter{args.max_iter}.npy',y_hat)
    np.save(f'result/{args.exp_name}/U1_ours_T0{args.T0}_s{args.s}_r1{args.r1}_r2{args.r2}.npy',U1)
    np.save(f'result/{args.exp_name}/U2_ours_T0{args.T0}_s{args.s}_r1{args.r1}_r2{args.r2}.npy',U2)


if __name__ == "__main__":
   main()
