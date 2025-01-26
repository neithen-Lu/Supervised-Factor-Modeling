import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import logging

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

def tucker_product(G,U1,U2):
    """
    Compute tucker product in the first two modes
    Output:
        G times_1 U1 times_2 U2
    """
    return np.einsum('ij,kjl->kil',U2,np.einsum('ij,jkl->ikl',U1,G))

def unfold(tensor, mode):
    """
    Unfold a three-way tensor into a matrix
    """
    shape = tensor.shape
    if mode == 1:
        matrix = np.transpose(tensor,(0, 2, 1)).reshape(shape[0], -1)
    elif mode == 2:
        matrix = np.transpose(tensor,(1, 2, 0)).reshape(shape[1], -1)
    elif mode == 3:
        matrix = np.transpose(tensor,(2, 1, 0)).reshape(shape[2], -1)
    return matrix

def fold(matrix, shape, mode):
    """
    Fold a matrix into a three-way tensor
    """
    if mode == 1:
        shape = [shape[0], shape[2], shape[1]]
        tensor = np.transpose(matrix.reshape(shape),(0, 2, 1))
    elif mode == 2:
        shape = [shape[1], shape[2], shape[0]]
        tensor = np.transpose(matrix.reshape(shape),(2, 0, 1))
    elif mode == 3:
        shape = [shape[2], shape[1], shape[0]]
        tensor = np.transpose(matrix.reshape(shape),(2, 1, 0))
    return tensor

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
    if np.any(np.isnan(gradient_A)):
        raise ValueError
    # gradient step for U1, U2, G
    U1_new = U1 - step_size * (unfold(gradient_A,1) @ np.kron(np.identity(P),U2) @ unfold(G,1).T + a * (U1@(U1.T@U1-b**2*np.identity(r1))))
    U2_new = U2 - step_size * (unfold(gradient_A,2) @ np.kron(U1,np.identity(P)) @ unfold(G,2).T + a * (U2@(U2.T@U2-b**2*np.identity(r2))))
    G_new = G - step_size * (tucker_product(gradient_A,U1.T,U2.T))
    # hard thresholding
    A = tucker_product(G_new,U1_new,U2_new)
    # print("fro norm list: {}".format(np.linalg.norm(A,ord='fro',axis=(0,1))))
    if thresholding_option == 'hard':
    # hard thresholding
        norm_0_idx = hard_thresholding(A,s)
        G_new[:,:,norm_0_idx] = 0
    elif thresholding_option == 'soft':
        G_new,norm_0_idx = soft_thresholding(A=A,G=G_new,lmda=lmda)
    elif thresholding_option == 'none':
        norm_0_idx = []

    return G_new,U1_new,U2_new,loss,norm_0_idx

def DFM(G,y,X,step_size,s):
    """
    Dynamic factor modeling: 
    "y" is the factor
    "X" is the factorized input
    """
    T,r,P = X.shape
    y_hat = np.einsum('TrP,irP->Ti',X,G)
    loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/T
    gradient_G = 2*np.einsum('TrP,Ti->irP', X,y_hat-y)/T # (r,r,P)
    if np.any(np.isnan(gradient_G)):
        raise ValueError
    G_new = G - step_size * gradient_G
    # hard thresholding
    norm_0_idx = hard_thresholding(G_new,s)
    G_new[:,:,norm_0_idx] = 0
    return G_new,loss,norm_0_idx

def initialize(y,X,P,r1,r2):
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

def train_epoch(y,X,P,r1,r2,a,b,s,lmda=0,thresholding_option='hard',max_iter=10000,step_size=1e-3,A_init=None,print_log=False,min_loss=0,early_stop=False,true_A=None):
    """
    The main train function

    """
    T,_ = y.shape
    # get initial values
    if A_init is not None:
        A,U = tucker(A_init,rank=[r1,r2,P])
        G = np.einsum('ijk,lk->ijl',A,U[2])
        U1 = U[0]; U2 = U[1]
    else:
        G,U1,U2 = initialize(y=y,X=X,P=P,r1=r1,r2=r2)

    A_old = tucker_product(G,U1,U2)
    iter = 0
    loss = np.inf
    A_diff = np.inf
    err_path = np.zeros(max_iter)
    while iter < max_iter:
        G,U1,U2,loss,norm_0_idx = AGD(G=G,U1=U1,U2=U2,y=y,X=X,a=a,b=b,step_size=step_size,s=s,thresholding_option=thresholding_option,lmda=lmda)
        if print_log and iter < 10:
            print('non-zero indices: {}'.format(np.delete(np.arange(50),norm_0_idx)))
        A = tucker_product(G,U1,U2)
        A_diff = np.linalg.norm(unfold(A-A_old,1),ord='fro')
        A_old = A
        if true_A is not None:
            err_path[iter] = np.linalg.norm(unfold(A-true_A[:,:,:P],1),ord='fro')**2 + np.linalg.norm(unfold(true_A[:,:,P:],1),ord='fro')**2
        iter += 1
        if print_log:
            if iter % 50 == 0:
                print('iter: {}, loss: {}'.format(iter,loss))
        if A_diff < min_loss and early_stop:
            y_hat = np.einsum('TNP,iNP->Ti',X,A)
            loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/T
            logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
            return tucker_product(G,U1,U2),U1,U2,norm_0_idx,loss,err_path

    logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
    return tucker_product(G,U1,U2),U1,U2,norm_0_idx,loss,err_path

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

def train_epoch_shared_subspace(y,X,P,r1,s,max_iter=1000,min_loss=1e-5,step_size=1e-3,
                G_init=None,print_log=False,early_stop=False):
    """
    The main train function for dynamic factor model

    """
    T,_ = y.shape

    G_old = np.zeros((r1,r1,P))
    G = G_old
    iter = 0
    loss = np.inf
    A_diff = np.inf
    while iter < max_iter:
        G,loss,norm_0_idx = DFM(G=G,y=y,X=X,step_size=step_size,s=s)
        if print_log and iter < 10:
            print('non-zero indices: {}'.format(np.delete(np.arange(50),norm_0_idx)))
        G_diff = np.linalg.norm(unfold(G-G_old,1),ord='fro')
        iter += 1
        if print_log:
            if iter % 50 == 0:
                print('iter: {}, loss: {}'.format(iter,loss))
        if G_diff < min_loss and early_stop:
            y_hat = np.einsum('TNP,iNP->Ti',X,G)
            loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/T
            logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
            return G,norm_0_idx,loss

    logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
    return G,norm_0_idx,loss
