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
from statsmodels.tsa.ar_model import AutoReg


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--ar_p',type=int,default=1)
    parser.add_argument('--k_factor',type=int,default=1)
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

    for t in range(start_ind,T):
        start_time = time.time()
        y = y_normed[:,:(t+1)].T
        y_target = y[-1,:]
        y = y[:-1,:]

        # obtain factors
        # estimate covariance 
        dfR = args.k_factor
        Y = y.T
        Ycov = Y @ Y.T / (t - 1)
        Yeigfl, Yufl = np.linalg.eig(Ycov)
        Ys, Yu = np.diag(Yeigfl[:dfR]), Yufl[:, :dfR]
        dfLambda = Yu @ Ys ** (1 / 2)
        F = np.linalg.inv(Ys) @ dfLambda.T @ Y
        
        y_forecast = np.zeros(N)
        residual = np.zeros((t-args.ar_p-1,N))
        for n in range(N):
            ar_model = AutoReg(endog=y[1:,n],lags=args.ar_p,exog=F[:,:-1].T)
            result = ar_model.fit()
            Ft = F[:,-1]
            y_forecast[n] = result.forecast(steps=1,exog=Ft[np.newaxis,:])[0]
            residual[:,n] = result.resid
        end_time = time.time()

        y_hat[t-start_ind,:] = y_forecast
        y_true[t-start_ind,:] = y_target
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])
        # log likelihood
        residual_centered = residual - np.mean(residual, axis=0)
        cov_matrix = np.dot(residual_centered.T, residual_centered) / (t - 1)
        runtime[t-start_ind] = end_time-start_time
    for t in range(start_ind,T):
        # calculate predictive log-likelihood
        dist = scipy.stats.multivariate_normal(mean=y_hat[t-start_ind,:], cov=cov_matrix)
        log_likelihood[t-start_ind] = dist.logpdf(y_true[t-start_ind,:])

    np.save(f'result/{args.exp_name}/Y_hat_factoraug_{args.ar_p}_{args.k_factor}.npy',y_hat)

if __name__ == "__main__":
   main()
