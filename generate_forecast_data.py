import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import argparse
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from dgp import DGP_VAR,DGP_VARMA,DGP_SEASON_VARMA,DGP_SEASON_VAR,DGP_MIX,DGP_SEASON_VAR_BIC,DGP_MIX_BIC,DGP_VARMA_BIC,DGP_SEASON_VAR_DIFFRANK
from alg import train, train_epoch
from utils import unfold,get_acf

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',type=str,default='test')
    
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

    # hyperparameter for training
    parser.add_argument('--a',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--b',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--s',type=int,default=3,help='hyperparameter in gradient descent')
    parser.add_argument('--lmda',type=float,default=1e-1,help='hyperparameter in gradient descent')
    parser.add_argument('--lr',type=float,default=1e-3,help='step size')
    parser.add_argument('--stop_thres',type=float,default=1e-4,help='stopping threshold for the different of A in each iteration')
    parser.add_argument('--thresholding_option',type=str,default='hard',choices=['hard','soft','none'],
                        help='choose hard thresholding or soft thresholding or no thresholding')
    parser.add_argument('--thresholding_interval',type=int,default=10,help='how many steps to do a thresholding')
    parser.add_argument('--max_iter',type=int,default=10000)
    parser.add_argument('--stop_method',type=str,choices=['none','loss','Adiff'],default='loss',
                        help='none: no early stop; loss: loss does not decrease for some iterations; Adiff: diff of |A_t - A_{t-1}|_F less than some threshold')
    parser.add_argument('--schedule',type=str,choices=['none','half'])

    # task related
    parser.add_argument('--iterative',action='store_true',help='whether to use an iterative method to select T0')
    parser.add_argument('--T0_init',type=int,default=100,help='initial value for T0')
    parser.add_argument('--A_init',type=str,choices=['spec','true','zero','GD','noisetrue','noisezero'],default='zero')
    parser.add_argument('--task',type=str,choices=['rate','convergence','hyperparameter','debug'],default='rate')

    # debug and visualize
    parser.add_argument('--visualize',action ='store_true')
    parser.add_argument('--print_log',action='store_true')


    args = parser.parse_args()
    # check and set other parameters based on args
    r1 = r2 = args.dgp_r
    if args.dgp_r1 is not None:
        r1 = args.dgp_r1 
    if args.dgp_r2 is not None:
        r2 = args.dgp_r2
    if args.visualize:
        args.stop_method = 'none'
    if args.task == 'convergence':
        args.max_iter = 10000; args.stop_method = 'none'

    # logging file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    wkdir = f'{dir_path}/result/{args.task}/{args.exp_name}'
    if not os.path.exists(f'{wkdir}/data'):
        os.makedirs(f'{wkdir}/data')
        print(f'create folder {wkdir}/data')


    for rep in range(args.n_rep):
        np.random.seed(rep+args.seed)
        # generate max length of y
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

        if true_A.shape[-1] < args.T0:
            true_A = np.concatenate((true_A,np.zeros((args.N,args.N,args.T0 -true_A.shape[-1]))),axis=-1)

        #### save data
        np.savetxt(f'{wkdir}/data/r1{r1}_r2{r2}_N{args.N}_T{args.T}_y_{rep+args.seed}.csv',y,delimiter=",")

if __name__ == "__main__":
   main()