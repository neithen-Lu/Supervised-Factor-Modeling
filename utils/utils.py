import numpy as np
import pandas as pd
from subprocess import Popen,PIPE
import os

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

def get_dgp_setting(rho,T1,r,N,C_T0,C_s,C_l):
    """
    Use rate to automatically determine T, T0, s and/or gamma (threshold for choosing s)
    """
    # for T0 and s rate
    T0 = np.round(np.log(np.divide(T1,C_T0)) / np.log(1/rho)).astype(int)
    s = np.round(C_s * np.log(T1) / np.log(1/rho)).astype(int)
    T = T1 + T0
    # for lambda and threshold gamma
    gamma = np.sqrt((r*N+np.log(T0))/T1)*C_l
    return (T,T0,s,gamma)

def get_acf():
    """
    Use R to get the operator norm of autocovariance matrices
    R read in ./temp/y.csv
    output ./temp/acf.csv
    """
    cmd = ["Rscript", "acf.r"]

    p = Popen(cmd, cwd=os.getcwd(),stdin=PIPE, stdout=PIPE, stderr=PIPE)     
    output = p.communicate()
    acf = pd.read_csv(os.getcwd()+'/temp/acf.csv')['x']
    return acf