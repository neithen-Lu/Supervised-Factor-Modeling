import os
os.environ["OMP_NUM_THREADS"] = "4"
import logging

import numpy as np
import argparse
import tensorly as tl
import pandas as pd
import logging
import glob


import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


def main():

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
    parser.add_argument('--k_factor',type=int,default=1)
    parser.add_argument('--data_path',type=str,help='path to data file')
    parser.add_argument('--ttratio',type=float,default=0.8,help='ratio of train set')
    parser.add_argument('--exp_name',type=str,required=True,help='folder name to store the result')

    args = parser.parse_args()
    r1 = r2 = args.dgp_r
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
        T,N = y.shape

        # obtain factors
        # estimate covariance 
        dfR = args.k_factor
        Y = y.T
        Ycov = Y @ Y.T / (T - 1)
        Yeigfl, Yufl = np.linalg.eig(Ycov)
        Ys, Yu = np.diag(Yeigfl[:dfR]), Yufl[:, :dfR]
        dfLambda = Yu @ Ys ** (1 / 2)
        F = np.linalg.inv(Ys) @ dfLambda.T @ Y
        
        y_forecast = np.zeros(N)
        for n in range(N):
            ar_model = AutoReg(endog=y[1:,n],lags=args.ar_p,exog=F[:,:-1].T)
            result = ar_model.fit()
            Ft = F[:,-1]
            y_forecast[n] = result.forecast(steps=1,exog=Ft[np.newaxis,:])[0]

        forecast_err[i,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[i,1] = np.linalg.norm(y_forecast-y_target,ord=1)
    r1 = r2 = args.dgp_r
    dir_path = os.path.dirname(os.path.realpath(__file__))
    wkdir = f'{dir_path}/result/rate/{args.exp_name}'
    with open(f'{wkdir}/forecast_factoraug_ar{args.ar_p}_kfactor{args.k_factor}_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}.csv','a') as f:
        f.write(f'{args.seed},{args.n_rep},T{args.T}_N{args.N}_di_ar{args.ar_p}_kfactor{args.k_factor},{np.average(forecast_err[:,0])},{np.average(forecast_err[:,1])}\n')
    np.savetxt(f'{wkdir}/csv/forecast_factoraug_ar{args.ar_p}_kfactor{args.k_factor}_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}_seed{args.seed}.csv',forecast_err)

if __name__ == "__main__":
   main()
