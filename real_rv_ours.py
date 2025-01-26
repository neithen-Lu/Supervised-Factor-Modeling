import os
os.environ["OMP_NUM_THREADS"] = "4"
import logging

import numpy as np
import argparse
import tensorly as tl
import pandas as pd
import logging

from dgp import *
from alg_real import train_epoch_shared_subspace

def main():
    # model setting
    step_size = 1e-3

    # hyper parameters default values
    stop_thres = 1e-4

    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--T0', type=int,default=10, help='T0')
    parser.add_argument('--s',type=int,default=4,help='sparsity level')
    parser.add_argument('--data_path',type=str,help='path to data file')
    parser.add_argument('--r1',type=int,default=3,help='rank at mode 1')
    
    parser.add_argument('--G_init',type=str,choices=['spec','zero'],default='zero')
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--max_iter',type=int,default=5000)
    args = parser.parse_args()

    np.random.seed(0)

    # logging
    wkdir = f'result/{args.data}'
    if not os.path.exists(wkdir):
        os.makedirs('{}/log'.format(wkdir))
        os.makedirs('{}/csv'.format(wkdir))
    logging.basicConfig(filename='{}/log/{}tune.log'.format(wkdir,args.T0), encoding='utf-8', level=logging.CRITICAL,
                            format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logging.critical(f"Experiment setting:\n --T0 {args.T0} --s {args.s} --r1 {args.r1} --lr {args.lr} --G_init {args.G_init} --max_iter {args.max_iter}")


    ############### read and process data #############
    df = pd.read_csv(args.data_path,header=None)
    RVraw = df.to_numpy().astype('float64')
    N,T = RVraw.shape
    print(N,T)
    y_mean = np.mean(RVraw, 1)
    y_sd = np.std(RVraw, 1)
    y_normed = (RVraw - y_mean[:, np.newaxis])/ y_sd[:,np.newaxis]
    ttratio = 0.9
    trainsz = int(ttratio * y_normed.shape[1])
    testsz = y_normed.shape[1] - int(ttratio * y_normed.shape[1])
    # print("We use " + str(trainsz) + " for training, and " + str(testsz) + " for testing.")
    start_ind = trainsz

    # rolling forecast
    forecast_err = np.zeros((T-start_ind,2))
    # true_A = pre_G = None
    tar_norm = 0
    Loss_list = np.zeros((T-start_ind))

    for t in range(start_ind,T):
        print("Forecast target: ", t)
        y = y_normed[:,:(t+1)].T
        Ycov = y[:-1,:].T @ y[:-1,:] / t
        Yeigfl, Yufl = np.linalg.eig(Ycov)
        Sy, Uy = np.diag(Yeigfl[:args.r1]), Yufl[:, :args.r1]

        dfLambda = Uy @ Sy ** (1 / 2)
        loading1 = dfLambda.T
        loading2 = np.linalg.inv(Sy)

        f = y @ Uy # [time, var]
        F = np.zeros((t+1-args.T0,args.r1,args.T0))
        for i in range(args.T0):
            F[:,:,i] = f[(args.T0-1-i):t-i,:]
        y = y[args.T0:,:]
        y_target = y[-1,:]
        yf = f[args.T0:-1,:] # delete target
        xf_target = F[-1,:,:]
        Xf = F[:-1,:,:]
        
        if args.G_init == 'zero':
            G_init = np.zeros((args.r1,args.r1,args.T0))
        else:
            G_init = None
        G_est,norm_0_idx,_ = train_epoch_shared_subspace(y=yf,X=Xf,P=args.T0,r1=args.r1,s=args.s,max_iter=args.max_iter,min_loss=stop_thres,step_size=args.lr,G_init=args.G_init)
        nonzero_idx = list(np.arange(0,args.T0))
        for i in norm_0_idx:
            nonzero_idx.remove(i)
        yf_forecast = np.einsum('rP,irP->i',xf_target,G_est)
        y_forecast = Uy @ yf_forecast
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])
        print("err: ",forecast_err[t-start_ind,:])

        f=open(f'result/{args.data}/csv/T0{args.T0}rank_'+str(args.r1)+"s"+str(args.s)+'Lambda.csv','a')
        np.savetxt(f, loading1, fmt='%10.7f',delimiter=',')
        f.write("\n")
        f.close()
        
    logging.critical(f"Mean forecast error: {np.mean(forecast_err,0)}",)


if __name__ == "__main__":
   main()
