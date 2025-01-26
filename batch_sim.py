import subprocess
import argparse
import os

## default VARMA setting ##
exp_name = f'varma_forecast'
r = 4
rho = 0.7
s = 10
N =100
T = 2000
T0 = 44
init = 'true'
n_rep = 10
num_proc = 1 
rep_per_proc = int(n_rep/num_proc)

# batch_no = 0
common_command = ['--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(T),'--T0',str(T0),'--N',str(N),'--exp_name',exp_name,'--s','10','--A_init','true']
# common_command = ['--dgp','season_ar_diffrank','--season','4','--dgp_p','4','--dgp_r1','4','--dgp_r2','2','--rho',str(rho),'--n_rep',str(rep_per_proc),'--T','1200','--T0','34','--N',str(N),'--exp_name',"var_forecast",'--s','5','--A_init','true']


# # uncomment the model if don't need to run
model_list = [
    # 'ar',
    # 'var',
    # 'var_lasso',
    # 'varma',
    # 'factor_aug',
    # 'approx_varma',
    # "mlr_shorr",
    "ours"
]

if 'ar' in model_list:
##### AR1 #####
    command = ['python','sim_ar.py','--ar_p','1'] + common_command
    subprocess.run(command)
##### AR2 #####
    command = ['python','sim_ar.py','--ar_p','2'] + common_command
    subprocess.run(command)

if 'var' in model_list:
##### VAR1 #####
    command = ['python','sim_var.py','--ar_p','1'] + common_command
    subprocess.run(command)
# ##### VAR2 #####
    command = ['python','sim_var.py','--ar_p','2'] + common_command
    subprocess.run(command)

if 'var_lasso' in model_list:
##### VAR Lasso #####
    command = ['python','sim_basu.py','--VARpen','L1'] + common_command
    subprocess.run(command)

if 'mlr_shorr' in model_list:
##### MLR & SHORR #####
    command = ['python','generate_forecast_data.py'] + common_command
    subprocess.run(command)
    command = ['julia','/home/r13user3/Documents/KX/glasso/wd.jl',str(r),str(r),str(N),str(T),str(s),str(T0),exp_name,str(rep_per_proc)]
    subprocess.run(command)


if 'factor_aug' in model_list:
##### Diffusion index #####
    command = ['python','sim_factor_aug.py','--k_factor',str(r),'--ar_p',str(s)] + common_command
    subprocess.run(command)

if 'varma' in model_list:
##### VARMA L1 ##### (~24 hr)
    command = ['python','sim_basu.py','--VARpen','L1','--VARMApen','L1'] + common_command
    subprocess.run(command)
##### VARMA Hlag ##### (~36 hr)
    command = ['python','sim_basu.py','--VARpen','HLag','--VARMApen','HLag'] + common_command
    subprocess.run(command)

if 'ours' in model_list:
##### Ours ##### 
    command = ['python','train_ours.py','--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),'--seed','0']+common_command
    subprocess.run(command)

if 'approx_varma' in model_list:
    command = ['python','sim_approx_varma.py','--ar_p',str(T0)] + common_command
    subprocess.run(command)