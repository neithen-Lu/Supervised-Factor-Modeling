import subprocess
import os


# ours
T0 = 3
s = 1
# ours & wd
r1 = 1
r2 = 1
r3 = 1
# DFM & diffusion index
k_factor = 1
factor_order = 1
# diffusion index
ar_p = 1
ttratio = 0.7

# comment the model if don't need to run
model_list = [
    # 'rw',
    # 'ar',
    # 'var',
    # 'var_lasso',
#     'varma',
#     'mlr_shorr',
#     'factor_aug',
#     'approx_varma',
    'ours',
]
# choose exp_name
exp_name = 'qd_small'
# exp_name = 'qd_medium'
# exp_name = 'qd_large'
data_path = f'data/{exp_name}_normed.csv'
common_command = ['--exp_name',exp_name,'--data_path',data_path,'--ttratio',str(ttratio)]

wkdir = 'result/'+exp_name
if not os.path.exists(wkdir):
    os.makedirs(wkdir)

if 'rw' in model_list:
##### Random walk #####
    command = ['python','real_var.py','--random_walk'] + common_command
    subprocess.run(command)

if 'ar' in model_list:
##### AR1 #####
    command = ['python','real_ar.py','--ar_p','1'] + common_command
    subprocess.run(command)
##### AR2 #####
    command = ['python','real_ar.py','--ar_p','2'] + common_command
    subprocess.run(command)

if 'var' in model_list:
##### VAR1 #####
    command = ['python','real_var.py','--ar_p','1'] + common_command
    subprocess.run(command)
##### VAR2 #####
    command = ['python','real_var.py','--ar_p','2'] + common_command
    subprocess.run(command)

if 'var_lasso' in model_list:
##### VAR Lasso #####
    command = ['python','real_basu.py','--VARpen','L1'] + common_command
    subprocess.run(command)

if 'mlr_shorr' in model_list:
##### MLR & SHORR #####
    command = ['julia','wd.jl',data_path,str(ar_p),str(r1),str(r2),str(r3),str(ttratio),exp_name]
    subprocess.run(command)


if 'factor_aug' in model_list:
##### Diffusion index #####
    command = ['python','real_factoraug.py','--k_factor',"1",'--ar_p',"1"] + common_command
    subprocess.run(command)

if 'varma' in model_list:
##### VARMA L1 ##### (~24 hr)
    command = ['python','real_basu.py','--VARpen','L1','--VARMApen','L1'] + common_command
    subprocess.run(command)
##### VARMA Hlag ##### (~36 hr)
    command = ['python','real_basu.py','--VARpen','HLag','--VARMApen','HLag'] + common_command
    subprocess.run(command)

if 'ours' in model_list:
##### Ours ##### 
    if exp_name == 'qd_large':
        command = ['python','real_ours.py','--T0',"3",'--r1',"1",'--r2',"1",'--s','1','--lr','5e-4','--max_iter','5000'] + common_command # small
    if exp_name == 'qd_medium':
        command = ['python','real_ours.py','--T0',"3",'--r1',"1",'--r2',"1",'--s','1','--lr','5e-4','--max_iter','5000'] + common_command # medium
    if exp_name == 'qd_small':
        command = ['python','real_ours.py','--T0',"10",'--r1',"1",'--r2',"3",'--s','3','--lr','1e-6','--max_iter','5000'] + common_command # large
    subprocess.run(command)

if 'approx_varma' in model_list:
##### approx_varma ##### 
    command = ['python','real_approx_varma.py','--ar_p',str(ar_p)] + common_command
    subprocess.run(command)


