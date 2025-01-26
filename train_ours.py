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
from utils.utils import unfold,get_acf
from utils.visualize import draw_acf

def main():

    """
    command line: python train.py --dgp arma --dgp_p 1 --dgp_r 4 --schedule half --stop_method Adiff --T 2700 --T0 79 --s 12 --N 20 --lr 0.02 --print_log --n_rep 1 --A_init true
    """

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
    parser.add_argument('--task',type=str,choices=['rate','convergence','hyperparameter','debug'],default='debug')

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
    if not os.path.exists(wkdir):
        try:
            os.makedirs(f'{wkdir}/log')
            os.makedirs(f'{wkdir}/csv')
            os.makedirs(f'{wkdir}/fig')
        except:
            pass
    if args.thresholding_option == 'hard':
        filename = f'{wkdir}/log/p{args.dgp_p}r{args.dgp_r}s{args.s}T{args.T}T0{args.T0}_init{args.A_init}_rank{args.dgp_r}_N{args.N}.log'
        print("make file")    
    elif args.thresholding_option == 'soft':
        filename = f'{wkdir}/log/p{args.dgp_p}r{args.dgp_r}lmda{args.lmda}T{args.T}T0{args.T0}_init{args.A_init}_rank{args.dgp_r}_N{args.N}.log'

    logging.basicConfig(filename=filename, encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


    logging.critical(f"""DGP setting:\n--dgp {args.dgp} --dgp_p {args.dgp_p} --dgp_r {args.dgp_r} --s {args.s} --T {args.T} --T0 {args.T0} --N {args.N} --rho {args.rho}\nExperiment setting:\n--n_rep {args.n_rep} --a {args.a} --b {args.b} --s {args.s} --lr {args.lr} --A_init {args.A_init} --stop_method {args.stop_method} --schedule {args.schedule} --seed {args.seed}""")

    total_est_err = np.zeros(args.n_rep)
    est_err = np.zeros(args.n_rep)
    est_aprox_err = np.zeros(args.n_rep)
    est_trunc_err = np.zeros(args.n_rep)
    loss_list = np.zeros(args.n_rep)
    mean_err_path = np.zeros(args.max_iter); mean_log_est_path = np.zeros(args.max_iter)
    sparsity_level = []
    forecast_err = np.zeros((args.n_rep,2))
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
        # np.savetxt(f'{wkdir}/r1{r1}_r2{r2}_N{args.N}_T{args.T}_y_{rep+args.seed}.csv',y,delimiter=",")

        if args.iterative:
            A = train(full_y=y,r1=r1,r2=r2,a=args.a,b=args.b,s=args.s,P_init=args.T0_init)
            _,_,P = A.shape
            est_err = np.linalg.norm(unfold(A-true_A[:,:,:P],1),ord='fro')**2 + np.linalg.norm(unfold(true_A[:,:,P:],1),ord='fro')**2

        else:
            y = y.T
            X = np.zeros((args.T-args.T0,args.N,args.T0))
            for i in range(args.T0):
                X[:,:,i] = y[(args.T0-1-i):args.T-1-i,:]
            y_target = y[-1]
            y = y[args.T0:-1,:]
            x = X[-1,:,:]
            X = X[:-1,:,:]

            y_hat = np.einsum('TNP,iNP->Ti',X,true_A[:,:,:args.T0])
            true_loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/args.T

            if rep == 0:
                A_fro_norm_list = np.linalg.norm(true_A[:,:,:args.T0],ord='fro',axis=(0,1))
                logging.info('true A fro norm^2: {}, true loss: {}\nfro norm list: {}'.format(np.linalg.norm(unfold(true_A,1),ord='fro')**2, true_loss, A_fro_norm_list) )
                if not os.path.exists('temp'):
                    os.makedirs('temp')
                np.savetxt('temp/y.csv',y,delimiter=',')

            A,norm_0_idx,loss,err_path,est_path = train_epoch(
                y=y,X=X,max_iter=args.max_iter,stop_thres=args.stop_thres,step_size=args.lr,
                P=args.T0,r1=r1,r2=r2,a=args.a,b=args.b,s=args.s,true_A=true_A[:,:,:args.T0], 
                A_init_method=args.A_init,print_log=args.print_log,
                thresholding_option=args.thresholding_option,thresholding_interval=args.thresholding_interval,
                stop_method=args.stop_method,schedule=args.schedule,lmda=args.lmda)
            if loss is None:
                continue

            y_forecast = np.einsum('NP,iNP->i',x,A)
            y_forecast_true = np.einsum('NP,iNP->i',x,true_A[:,:,:args.T0])
            forecast_err[rep,0] = np.linalg.norm(y_forecast-y_target)
            forecast_err[rep,1] = np.linalg.norm(y_forecast_true-y_target)
            total_est_err[rep] = np.linalg.norm(unfold(A-true_A[:,:,:args.T0],1),ord='fro')**2 + np.linalg.norm(unfold(true_A[:,:,args.T0:],1),ord='fro')**2
            est_aprox_err[rep] = np.linalg.norm(unfold(true_A[:,:,norm_0_idx],1),ord='fro')**2
            true_A[:,:,norm_0_idx] = 0
            est_err[rep] = np.linalg.norm(unfold(A-true_A[:,:,:args.T0],1),ord='fro')**2
            est_trunc_err[rep] = np.linalg.norm(unfold(true_A[:,:,args.T0:],1),ord='fro')**2
            loss_list[rep] = loss
            mean_err_path += est_path; mean_log_est_path += np.log(est_path/est_path[0])
            # print(A-true_A[:,:,:P])
        norm_not0_index = np.delete(np.arange(args.T0),norm_0_idx)
        logging.info('non-zero indices: {}'.format(norm_not0_index))
        logging.info('replication no.{}: err {}'.format(rep,total_est_err[rep]))
        if args.thresholding_option == "soft":
            sparsity_level.append(len(norm_not0_index))
            average_sparsity = np.average(np.array(sparsity_level))
        

    mean_err_path /= args.n_rep
    mean_log_est_path /= args.n_rep
    if args.task == 'convergence':
        # plt.figure()
        # plt.plot(mean_err_path)
        # plt.savefig('{}/fig/p{}r{}T{}_cvg.png'.format(wkdir,args.dgp_p,args.dgp_r,args.T))
        plt.figure()
        plt.plot(mean_log_est_path)
        plt.savefig('{}/fig/cvg_.png'.format(wkdir,args.dgp_p,args.dgp_r,args.T,args.N))
        plt.figure()
        mean_ratio_err_path = mean_err_path[1:]/mean_err_path[:-1]
        print(mean_err_path)
        plt.plot(mean_ratio_err_path)
        # plt.ylim(0.8,1.2)
        plt.savefig('{}/fig/rat_.png'.format(wkdir,args.dgp_p,args.dgp_r,args.T,args.N))
        np.savetxt(f"wkdir/convergence_{args.A_init}.csv",mean_err_path)
        

    logging.critical("End replication, mean err^2: {}, min_loss: {}\n mean est err^2: {}, mean approx err^2: {}, mean trunc err^2: {}\n".format(np.average(total_est_err),np.average(loss_list), np.average(est_err), np.average(est_aprox_err), np.average(est_trunc_err)))

    if args.task == 'rate':
        if args.thresholding_option == "hard":
            with open(f'{wkdir}/csv/est_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}_T0{args.T0}_s{args.s}.csv','a') as f:
                f.write(f'{args.seed},{args.n_rep},T{args.T}_T0{args.T0}_N{args.N}_s{args.s}_A_init_{args.A_init},{np.average(total_est_err)},{np.average(est_err)},{np.average(est_aprox_err)},{np.average(est_trunc_err)}\n')
            with open(f'{wkdir}/forecast_ours_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}_T0{args.T0}_s{args.s}.csv','a') as f:
                f.write(f'{args.seed},{args.n_rep},T{args.T}_T0{args.T0}_N{args.N}_s{args.s}_A_init_{args.A_init},{np.average(forecast_err[:,0])},{np.average(forecast_err[:,1])}\n')
            np.savetxt(f'{wkdir}/csv/forecast_ours_p{args.dgp_p}_r1{r1}_r2{r2}_N{args.N}_T{args.T}_T0{args.T0}_s{args.s}_seed{args.seed}.csv',forecast_err)
        elif args.thresholding_option == "soft":
            with open('{}/csv/p{}_r1{}_r2{}_N{}.csv'.format(wkdir,args.dgp_p,r1,r2,args.N),'a') as f:
                f.write(f'{args.n_rep},T{args.T}_T0{args.T0}_N{args.N}_lmda{args.lmda}_A_init_{args.A_init},{np.average(total_est_err)},{np.average(est_err)},{np.average(est_aprox_err)},{np.average(est_trunc_err)},{average_sparsity},{sparsity_level}\n')
    if args.task == 'convergence':
        np.savetxt("{}/csv/p{}r{}T{}N{}.csv".format(wkdir,args.dgp_p,args.dgp_r,args.T,args.N), mean_log_est_path, delimiter=",")


if __name__ == "__main__":
   main()
   