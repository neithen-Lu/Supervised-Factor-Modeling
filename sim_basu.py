import os
os.environ["OMP_NUM_THREADS"] = "5"
import numpy as np
import pandas as pd
from alg_Basu import BasuEstimate
import timeit
import csv
import logging
import argparse
import multiprocessing as mp
import glob

import sys, getopt


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
    parser.add_argument('--VARpen',type=str,choices=['HLag','L1'])
    parser.add_argument('--VARMApen',type=str,choices=['HLag','L1'])
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

        y_forecast = BasuEstimate(y,args.VARpen,args.VARMApen)

        forecast_err[i,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[i,1] = np.linalg.norm(y_forecast-y_target,ord=1)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    wkdir = f'{dir_path}/result/rate/{args.exp_name}'
    with open(f'{wkdir}/forecast_basu_{args.VARpen}_{args.VARMApen}_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}.csv','a') as f:
        f.write(f'{args.seed},{args.n_rep},T{args.T}_N{args.N}_A_init_{args.A_init},{np.average(forecast_err[:,0])},{np.average(forecast_err[:,1])}\n')
    np.savetxt(f'{wkdir}/csv/forecast_basu_{args.VARpen}_{args.VARMApen}_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}_seed{args.seed}.csv',forecast_err)

             
if __name__ == "__main__":
    main()
