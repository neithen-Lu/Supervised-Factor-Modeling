import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import logging
from scipy.stats import ortho_group
import time

import utils.utils as utils
from utils.utils import fold,unfold,tucker_product

## Initialization
def spectral_initialize(y,X,P,r1,r2):
    """
    Spectral initialization
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    T,N = y.shape
    # spectral initialization
    A = np.zeros((N,N*P))
    for t in range(T):
        A = A + np.outer(y[t,:],X[t,:,:])
    A = A/(T-P)
    # fold A into a tensor
    A = np.array(fold(A,(N,N,P),1))

    # HOOI to get a low rank version
    A,U = tucker(A,rank=[r1,r2,P])
    G = np.einsum('ijk,lk->ijl',A,U[2])
    return G,U[0],U[1]

def GD_initialize(y,X,r1,r2,a,b,step_size,iter,true_A,print_log):
    T,N,P = X.shape
    gradient_A = 2*np.einsum('TNP,Ti->iNP', X,-y)/T # (N,N,P)
    if np.any(np.isnan(gradient_A)):
        raise ValueError
    U1 = ortho_group.rvs(N)[:,:r1]
    U2 = ortho_group.rvs(N)[:,:r2]
    G = np.zeros((r1,r2,P))
    best_loss = np.inf

    for i in range(iter):
    # gradient step for U1, U2, G
        U1 = U1 - step_size * (unfold(gradient_A,1) @ np.kron(np.identity(P),U2) @ unfold(G,1).T + a * (U1@(U1.T@U1-b**2*np.identity(r1))))
        U2 = U2 - step_size * (unfold(gradient_A,2) @ np.kron(U1,np.identity(P)) @ unfold(G,2).T + a * (U2@(U2.T@U2-b**2*np.identity(r2))))
        G = G - step_size * (tucker_product(gradient_A,U1.T,U2.T))
        # hard thresholding
        A = tucker_product(G,U1,U2)
        y_hat = np.einsum('TNP,iNP->Ti',X,A)
        gradient_A = 2*np.einsum('TNP,Ti->iNP', X,y_hat-y)/T # (N,N,P)
        loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/T
        if true_A is not None:
            err = np.linalg.norm(unfold(A-true_A[:,:,:P],1),ord='fro')**2
        if print_log:
            print(err)
            print('init iter: {}, init loss: {}'.format(i,loss))

        if loss < best_loss:
            best_loss = loss
            G_best = G; U1_best = U1; U2_best = U2
    # print("fro norm list: {}".format(np.linalg.norm(A,ord='fro',axis=(0,1))))
    return G_best,U1_best,U2_best

## Algorithm componnets

def hard_thresholding(A, s):
    """
    Only keep the slices (in 3rd mode) with top-s F norm, set other slices to 0
    Input: 
        A (N,N,P)
        s: number of slices to keep
    Output:
        norm_0_idx: the indices of slices to be set to 0
    """
    _,_,P = A.shape
    norm_list=np.zeros(P)
    for i in range(P):
        norm_list[i] = np.linalg.norm(A[:,:,i],ord='fro')
    norm_0_idx = np.argsort(norm_list)[:P-s] #bottom P-s 
    return norm_0_idx

def soft_thresholding(A,G,lmda):
    """
    Soft thresholding to each slice of A (in rd mode) by lmda
    Input: 
        A (N,N,P)
        lmda: penalty term
    Output:
        A: tensor after soft tresholding
    """
    _,_,P = A.shape
    norm_0_idx = []
    for i in range(P):
        if np.linalg.norm(A[:,:,i],ord='fro')-lmda < 0:
            G[:,:,i] = 0
            norm_0_idx.append(i)
        else:
            G[:,:,i] = G[:,:,i] * (1 - lmda/np.linalg.norm(A[:,:,i],ord='fro'))
    return G,norm_0_idx

def AGD(G,U1,U2,y,X,a,b,step_size,s,thresholding_option,lmda):
    """
    Alternating gradient descent algorithm
    Input:
        G (r_1,r_2,P)
        U1 (N,r_1)
        U2 (N,r_2)
        y (T,N)
        X (T,N,P)
        a
        b
        step_size: learning rate
        s: number of slices to keep
    Output:
        A (N,N,P)
    """
    T,N,P = X.shape
    _,r1 = U1.shape
    _,r2 = U2.shape
    # calculate A and gradient of A
    A = tucker_product(G,U1,U2)
    y_hat = np.einsum('TNP,iNP->Ti',X,A)
    loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/T
    gradient_A = 2*np.einsum('TNP,Ti->iNP', X,y_hat-y)/T # (N,N,P)
    # print('gradient: ',np.linalg.norm(unfold(gradient_A,1),ord='fro'))
    if np.any(np.isnan(gradient_A)):
        logging.critical('nan apeared in gradient, skip this replication')
        return (None,None,None,None,None)
    # gradient step for U1, U2, G
    U1_new = U1 - step_size * (unfold(gradient_A,1) @ np.kron(np.identity(P),U2) @ unfold(G,1).T + a * (U1@(U1.T@U1-b**2*np.identity(r1))))
    U2_new = U2 - step_size * (unfold(gradient_A,2) @ np.kron(U1,np.identity(P)) @ unfold(G,2).T + a * (U2@(U2.T@U2-b**2*np.identity(r2))))
    G_new = G - step_size * (tucker_product(gradient_A,U1.T,U2.T))
    A = tucker_product(G_new,U1_new,U2_new)
    if thresholding_option == 'hard':
    # hard thresholding
        norm_0_idx = hard_thresholding(A,s)
        G_new[:,:,norm_0_idx] = 0
    elif thresholding_option == 'soft':
        G_new,norm_0_idx = soft_thresholding(A=A,G=G_new,lmda=lmda)
    elif thresholding_option == 'none':
        norm_0_idx = []
    
    return G_new,U1_new,U2_new,loss,norm_0_idx

def train_epoch(y,X,P,r1,r2,a,b,s,stop_method,schedule,max_iter=10000,stop_thres=1e-5,step_size=1e-3,init_step_size=5e-3,init_iter=1000,
                true_A=None,A_init_method=None,print_log=False,thresholding_option='hard',thresholding_interval=10,lmda=0,init_seed=0):
    """
    The main train function

    """
    T,N = y.shape
    # get initial values
    if A_init_method == 'true':
        A_init = true_A[:,:,:P]
        A,U = tucker(A_init,rank=[r1,r2,P])
        G = np.einsum('ijk,lk->ijl',A,U[2])
        U1 = U[0]; U2 = U[1]
    elif A_init_method == 'noisetrue':
        # noise = np.einsum('ijk,k->ijk',np.random.normal(scale=0.1,size=true_A[:,:,:P].shape),np.array([1/t for t in range(1,P+1)]))
        # # for j in range(P):
        # #     if np.linalg.norm(noise[:,:,j],ord='fro') > np.linalg.norm(true_A,ord='fro',axis=(0,1))[-1]:
        # #         noise[:,:,j] = noise[:,:,j] * (np.linalg.norm(true_A,ord='fro',axis=(0,1)))[-1] / np.linalg.norm(noise[:,:,j],ord='fro')
        # A_init = true_A[:,:,:P] + noise
        # print(np.linalg.norm(A_init,ord='fro',axis=(0,1)))
        A_init = true_A[:,:,:P]
        A,U = tucker(A_init,rank=[r1,r2,P])
        G = np.einsum('ijk,lk->ijl',A,U[2])
        noise = np.einsum('ijk,k->ijk',np.random.normal(scale=0.1,size=G.shape),np.array([1/t for t in range(1,P+1)]))
        G = G + noise
        U1 = U[0]; U2 = U[1]
    elif A_init_method == 'noisezero':
        np.random.seed(init_seed)
        U1 = ortho_group.rvs(N)[:,:r1]
        U2 = ortho_group.rvs(N)[:,:r2]
        G = np.zeros((r1,r2,P))
        np.random.seed(int(time.time()))
        noise = np.einsum('ijk,k->ijk',np.random.normal(scale=0.01,size=G.shape),np.array([1/t for t in range(1,P+1)]))
        G = G + noise
    elif A_init_method == 'zero':
        U1 = ortho_group.rvs(N)[:,:r1]
        U2 = ortho_group.rvs(N)[:,:r2]
        G = np.zeros((r1,r2,P))
    elif A_init_method == 'spec':
        G,U1,U2 = spectral_initialize(y=y,X=X,P=P,r1=r1,r2=r2)
    elif A_init_method == 'GD':
        G,U1,U2 = GD_initialize(y=y,X=X,r1=r1,r2=r2,a=a,b=b,step_size=init_step_size,iter=init_iter,true_A=true_A,print_log=print_log)
        

    A_old = tucker_product(G,U1,U2)
    # print('init: ',np.linalg.norm(unfold(A_old,1),ord='fro')**2)
    best_loss = np.inf; A_diff = np.inf; loss_increase_count = 0
    err_path = np.zeros(max_iter); est_path = np.zeros(max_iter)
    for iter in range(max_iter):
        ################# lr schedule ###############
        if schedule == 'half':
            schedule_step = 500
            if iter % schedule_step == schedule_step -1:
                step_size *= 0.5

        ####### deal with thresholding interval #####
        if iter % thresholding_interval == 0:
            G,U1,U2,loss,norm_0_idx = AGD(G=G,U1=U1,U2=U2,y=y,X=X,a=a,b=b,step_size=step_size,s=s,thresholding_option=thresholding_option,lmda=lmda)
            if print_log:# and iter < 1000: 
                print(f'Perform thresholding, non-zero indices: {np.delete(np.arange(P),norm_0_idx)}')
        else:
            G,U1,U2,loss,norm_0_idx = AGD(G=G,U1=U1,U2=U2,y=y,X=X,a=a,b=b,step_size=step_size,s=s,thresholding_option='none',lmda=lmda)
        if G is None:
            return None,None,None,None,None
        A = tucker_product(G,U1,U2)
        A_diff = np.linalg.norm(unfold(A-A_old,1),ord='fro')
        A_old = A

        # calculate loss and keep track of best loss
        y_hat = np.einsum('TNP,iNP->Ti',X,A)
        loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/T
        if loss > best_loss:
            loss_increase_count += 1
        else:
            best_loss = loss; loss_increase_count = 0
            A_best = A
            best_norm_0_idx = norm_0_idx

        # for convergence log
        if true_A is not None:
            err_path[iter] = np.linalg.norm(unfold(A-true_A[:,:,:P],1),ord='fro')**2 + np.linalg.norm(unfold(true_A[:,:,P:],1),ord='fro')**2
            true_A_copy = true_A.copy()
            true_A_copy[:,:,norm_0_idx] = 0
            est_path[iter] = np.linalg.norm(unfold(A-true_A[:,:,:P],1),ord='fro')**2
        if print_log and iter % 50 == 0:
        # if print_log and iter < 50:
            print('iter: {}, loss: {}'.format(iter,loss))
            # print('non-zero indices: {}'.format(np.delete(np.arange(P),norm_0_idx)))
            print('est err: ',est_path[iter])
            print('A fnorm: ',np.linalg.norm(unfold(A,1),ord='fro')**2)
        
        ################## stop criteria ##################
        if ((stop_method == 'loss') and (A_diff < stop_thres or loss_increase_count > 500)):
            logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
            # threshold the returned A
            if thresholding_option == 'hard':
            # hard thresholding
                norm_0_idx = hard_thresholding(A_best,s)
                A_best[:,:,norm_0_idx] = 0
            elif thresholding_option == 'soft':
                if norm_0_idx:
                    norm_0_idx
                else:
                    norm_0_idx = []
            elif thresholding_option == 'none':
                norm_0_idx = []
            return A_best,norm_0_idx,best_loss,err_path,est_path
        elif (stop_method == 'Adiff' and A_diff < stop_thres): 
            # mix two conditions:
            # 1. frobenius norm of (A_k - A_{k-1}) < epsilon
            # 2. loss does not decrease for 200 iterations
            logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
            # threshold the returned A
            if thresholding_option == 'hard':
            # hard thresholding
                norm_0_idx = hard_thresholding(A,s)
                A[:,:,norm_0_idx] = 0
            elif thresholding_option == 'soft':
                if norm_0_idx:
                    norm_0_idx
                else:
                    norm_0_idx = []
            elif thresholding_option == 'none':
                norm_0_idx = []
            return A,norm_0_idx,loss,err_path,est_path

    logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
    if stop_method == 'loss':
        if thresholding_option == 'hard':
        # hard thresholding
            norm_0_idx = hard_thresholding(A_best,s)
            A_best[:,:,norm_0_idx] = 0
        elif thresholding_option == 'soft':
            if norm_0_idx:
                norm_0_idx
            else:
                norm_0_idx = []
        elif thresholding_option == 'none':
            norm_0_idx = []
        return A_best,norm_0_idx,best_loss,err_path,est_path
    else:
        if thresholding_option == 'hard':
        # hard thresholding
            norm_0_idx = hard_thresholding(A,s)
            A[:,:,norm_0_idx] = 0
        elif thresholding_option == 'soft':
            if norm_0_idx:
                norm_0_idx
            else:
                norm_0_idx = []
        elif thresholding_option == 'none':
            norm_0_idx = []
        return A,norm_0_idx,loss,err_path,est_path


def train(full_y,r1,r2,a,b,s,P_init,P_lwb=10):
    """
    Iteratively reduce T0 by half if the latter half contains all 0
    """
    N,T = full_y.shape
    P = P_init
    while P > P_lwb:
        print('T0 reduced to {}'.format(P))
        # each epoch, produce corresponding y and X
        y = full_y.T
        X = np.zeros((T-P,N,P))
        for i in range(P):
            X[:,:,i] = y[(P-i):T-i,:]
        y = y[P:,:]
        # train one epoch
        A,norm_0_idx,loss,err_path = train_epoch(y=y,X=X,P=P,r1=r1,r2=r2,a=a,b=b,s=s) 
        # check if the latter half of A is all 0
        if np.all(A[:,:,int(np.ceil(P/2)):] == 0):
            P = int(np.ceil(P/2))
        else:
            print('Cannot reduce T0')
            break

    return A,norm_0_idx,loss,err_path