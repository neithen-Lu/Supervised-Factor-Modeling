import os
os.environ["OMP_NUM_THREADS"] = "5"
import numpy as np
import pandas as pd
from alg_Basu import BasuEstimate
import time
import argparse
import multiprocessing as mp
import scipy


def one_step_forecast(t,y_normed,args):
    y = y_normed[:,:t]
    y_forecast = BasuEstimate(y,args.VARpen,args.VARMApen)

    return y_forecast


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--VARpen',type=str,choices=['HLag','L1'])
    parser.add_argument('--VARMApen',type=str,choices=['HLag','L1'])
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
    time = np.zeros((T-start_ind))

    param_list = []
    for t in range(start_ind,T):
        param_list.append((t,y_normed,args))
    num_workers = 1
    with mp.Pool(processes=num_workers) as pool:
        # Apply the compute function to each element in the data list
        results = pool.starmap(one_step_forecast, param_list)

    for t in range(start_ind,T):
        y = y_normed[:,:(t+1)].T
        y_target = y[-1,:]
        y = y[:-1,:]
        y_forecast = results[t-start_ind][0]
        y_hat[t-start_ind,:] = y_forecast
        y_true[t-start_ind,:] = y_target
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])
    
    np.save(f'result/{args.exp_name}/Y_hat_basu_{args.VARpen}_{args.VARMApen}.npy',y_hat)


             
if __name__ == "__main__":
    main()
