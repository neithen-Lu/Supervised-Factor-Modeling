import os
os.environ["OMP_NUM_THREADS"] = "4"
import logging

import numpy as np
import argparse
import tensorly as tl
import pandas as pd
import logging
import scipy
import time


import statsmodels.api as sm
from statsmodels.tsa.api import VAR


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--ar_p',type=int,default=1)
    parser.add_argument('--random_walk',action='store_true')
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
    tar_norm = 0
    runtime = np.zeros((T-start_ind))

    for t in range(start_ind,T-1):
        start_time = time.time()
        y = y_normed[:,:(t+1)].T
        y_target = y[-1,:]
        y = y[:-1,:]
        
        if args.random_walk:
            y_forecast = y[-1,:]
            end_time = time.time()
        else:
            var_model = VAR(endog=y)
            result = var_model.fit(args.ar_p)
            y_forecast = np.squeeze(result.forecast(y, 1))
            end_time = time.time()
            # log likelihood
            residual = result.resid
            residual_centered = residual - np.mean(residual, axis=0)
            if t >= T-17:
                cov_matrix = np.dot(residual_centered.T, residual_centered) / (t - 1)
        
        y_hat[t-start_ind,:] = y_forecast
        y_true[t-start_ind,:] = y_target
        runtime[t-start_ind] = end_time-start_time
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])
    
        

    if args.random_walk:
        np.save(f'result/{args.exp_name}/Y_hat_rw.npy',y_hat)
        np.save(f'result/{args.exp_name}/Y_true.npy',y_hat)
    else:
        np.save(f'result/{args.exp_name}/Covmat_var{args.ar_p}.npy',cov_matrix)
        np.save(f'result/{args.exp_name}/Y_hat_var{args.ar_p}.npy',y_hat)
        np.save(f'result/{args.exp_name}/Time_var{args.ar_p}.npy',runtime)
        



if __name__ == "__main__":
   main()
