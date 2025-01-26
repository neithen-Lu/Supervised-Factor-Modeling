import os
os.environ["OMP_NUM_THREADS"] = "5"
import logging

import numpy as np
import argparse
import tensorly as tl
import pandas as pd
import logging
import glob

from alg_approx_varma import BCD
from help_func import get_L
from tensorOp import tensor_op

import multiprocessing as mp



def one_step_forecast(t,y,args,N):
    y=y.T
    x = np.reshape(np.flip(y,axis=1),(-1,1),order='F') # vectorized y, yt to y1
    
    r = 1
    s = 0
    lmbd = [0.5]
    gamma = []
    theta = []
    L = get_L(lmbd, gamma, theta, r, s,  t, 0)
    G_init = np.zeros((N,N,1))
    A = tl.tenalg.mode_dot(G_init, L, 2)  # N x N x T tensor
    lmbd_est, gamma_est, theta_est, G_est, Loss, flag_maxiter, iter_no,residual  = \
                                BCD(y, 0, r, s, 0.003, 5,
                                    args.lr, y[:,:-2], np.array(lmbd), np.array(gamma),
                                    np.array(theta), A, G_init, L,
                                    stop_thres = 1e-2, stop_method = 'SepEst')
    L_est = get_L(lmbd_est, gamma_est, theta_est, r, s, t, 0)
    A_est = tl.tenalg.mode_dot(G_est, L_est, 2)  # N x N x T tensor
    y_forecast = tensor_op.unfold(A_est,1).numpy() @ x
    return np.squeeze(y_forecast)

def main():
    # model setting
    step_size = 1e-3

    # hyper parameters default values
    stop_thres = 1e-4

    parser = argparse.ArgumentParser()
    # dgp
    parser.add_argument('--dgp',type=str,choices=['ar','arma','season_ar','season_arma','mix','season_ar_diffrank'],default='arma')
    parser.add_argument('--season',type=int)
    parser.add_argument('--dgp_p',type=int,default=1)
    parser.add_argument('--dgp_r',type=int,default=2)
    parser.add_argument('--dgp_r1',type=int)
    parser.add_argument('--dgp_r2',type=int)
    parser.add_argument('--rho',type=float,default=0.7)

    # experiment setting
    parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--T', type=int,default=200, help='sample size for dgp')
    parser.add_argument('--T0', type=int,default=10, help='T0')
    parser.add_argument('--N', type=int, default=5, help='diemnsion for dgp')
    parser.add_argument('--seed', type=int,default=0)

    # parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--ar_p',type=int,default=1)
    parser.add_argument('--lr',type=float,default=1e-3,help='step size')
    parser.add_argument('--data',type=str,default='money',choices=['money','RV','MEDIUM','HUGE'],help='name of the dataset')
    parser.add_argument('--random_walk',action='store_true')
    parser.add_argument('--data_path',type=str,help='path to data file')
    parser.add_argument('--ttratio',type=float,default=0.8,help='ratio of train set')
    parser.add_argument('--exp_name',type=str,required=True,help='folder name to store the result')


    args = parser.parse_args()

 # check and set other parameters based on args
    r1 = r2 = args.dgp_r
    if args.dgp_r1 is not None:
        r1 = args.dgp_r1 
    if args.dgp_r2 is not None:
        r2 = args.dgp_r2
    forecast_err = np.zeros((args.n_rep,2))
    from dgp import DGP_VAR,DGP_VARMA,DGP_SEASON_VARMA,DGP_SEASON_VAR,DGP_MIX,DGP_SEASON_VAR_BIC,DGP_MIX_BIC,DGP_VARMA_BIC,DGP_SEASON_VAR_DIFFRANK
    for i in range(args.n_rep):
        np.random.seed(i+args.seed)
        if args.dgp == 'ar':
            y,true_A = DGP_VAR(args.N,args.T,burn=200,P=args.dgp_p,r=args.dgp_r,rho=args.rho)
            if args.T0 > args.dgp_p:
                true_A = np.concatenate((true_A, np.zeros((true_A.shape[0], true_A.shape[1], args.T0-args.dgp_p)).astype(true_A.dtype)),axis=-1)
        elif args.dgp == 'arma':
            y,true_A,eps = DGP_VARMA(args.N,args.T,burn=200,p=args.dgp_p,r=args.dgp_r,rho=args.rho)
        elif args.dgp == 'season_arma':
            y,true_A = DGP_SEASON_VARMA(args.N,args.T,burn=200,p=args.dgp_p,r=args.dgp_r,rho=args.rho,season=args.season)
        elif args.dgp == 'season_ar':
            y,true_A = DGP_SEASON_VAR(args.N,args.T,burn=200,r=args.dgp_r,rho=args.rho)
        elif args.dgp == 'mix':
            y,true_A = DGP_MIX(args.N,args.T,P=args.T,burn=200,r=args.dgp_r,rho=args.rho)
        elif args.dgp == 'season_ar_diffrank':
            y,true_A = DGP_SEASON_VAR_DIFFRANK(args.N,args.T,burn=200,r1=args.dgp_r1,r2=args.dgp_r2,rho=args.rho)
        y=y.T
        y_target = y[-1,:]
        y = y[:-1,:]
        N = y.shape[1]

        y_forecast = one_step_forecast(y.shape[0],y,args,N)

        forecast_err[i,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[i,1] = np.linalg.norm(y_forecast-y_target,ord=1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    wkdir = f'{dir_path}/result/rate/{args.exp_name}'
    with open(f'{wkdir}/forecast_approx_varma_p0_r1_s0_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}.csv','a') as f:
        f.write(f'{args.seed},{args.n_rep},T{args.T}_N{args.N},{np.average(forecast_err[:,0])},{np.average(forecast_err[:,1])}\n')
    np.savetxt(f'{wkdir}/csv/forecast_approx_varma_p0_r1_s0_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}_seed{args.seed}.csv',forecast_err)

if __name__ == "__main__":
   main()
