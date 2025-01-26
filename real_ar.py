import os
os.environ["OMP_NUM_THREADS"] = "4"
import logging

import numpy as np
import argparse
import pandas as pd



import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--ar_p',type=int,default=1)
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

    for t in range(start_ind,T):
        y = y_normed[:,:(t+1)].T
        y_target = y[-1,:]
        y = y[:-1,:]
        
        y_forecast = np.zeros(N)
        for n in range(N):
            ar_model = AutoReg(endog=y[:,n],lags=args.ar_p)
            result = ar_model.fit()
            y_forecast[n] = result.forecast(steps=1)[0]


        y_hat[t-start_ind,:] = y_forecast
        y_true[t-start_ind,:] = y_target
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])


    np.save(f'result/{args.exp_name}/Y_hat_ar{args.ar_p}.npy',y_hat)


if __name__ == "__main__":
   main()
