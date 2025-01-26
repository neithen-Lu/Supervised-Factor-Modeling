"""
Model parameters:
p: AR part order
q: MA part order
r: no. of real eigenvalues
s: no. of complex eigenvalue pairs
d = p+r+2s
N: dimension of the time series vector
T: length of the time series

Variables:
A: N*N*T tensor
G: N*N*d tensor
L: T*d matrix
w: d*1 vector
lmbd: r*1 vector
gamma: s*1 vector
theta: s*1 vector

Data:
y: N*T matrix, stored as column vectors, from old (left) to new (right)
"""

# from math import sqrt
import numpy as np
import scipy
from tensorOp import tensor_op
import tensorly as tl
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
from sklearn import linear_model
import torch
from scipy.stats import ortho_group


##################
# Initialization #
##################
def init_A_spr_var(y, P, lagr_mul_var, max_iter=1000):
    N = y.shape[0]
    T = y.shape[1]
    
    vec_y = np.ravel(y[:,P:].reshape((-1,1)))
    linear_matrix_1 = np.zeros((T - P,N*P))
    for i in range(T - P):
        linear_matrix_1[i,:] = y[:,range(i+P-1,i-1,-1)].T.reshape((1,-1))
    linear_matrix = np.kron(np.eye(N), linear_matrix_1)
    
    lasso = linear_model.Lasso(alpha=lagr_mul_var, max_iter=max_iter)  # max_iter=1000
    lasso.fit(linear_matrix, vec_y)
    lasso_coef = lasso.coef_
                     
    A = torch.from_numpy(lasso_coef.reshape((N,P,N))).permute((0,2,1))                    
    return A

def get_L_MA(lmbd, gamma, theta, r, s, P):  # checked
    """
    Compute the L_MA matrix given the parameters
    Set size to be P (truncated)
    """
    L = np.zeros((P, r + 2 * s))
    tri_series = np.zeros((P, 2 * s))
    for i in range(P):
        tri_series[i, ::2] = np.cos((i + 1) * np.array(theta))
        tri_series[i, 1::2] = np.sin((i + 1) * np.array(theta))
        for j in range(r):
            L[i, j] = np.power(lmbd[j], i + 1)
        for j in range(s):
            L[i, r + 2 * j:r + 2 * j + 2] = np.power(gamma[j], i + 1)
    # np.concatenate return a combined matrix
    # np.einsum('ij,ij->ij', A, B) reutrn the dot prod of A and B
    new = np.concatenate([L[:, :r], np.einsum('ij,ij -> ij', L[:, r:], tri_series)], axis=1)
    return new


def get_L(lmbd, gamma, theta, r, s, P, p):  # checked
    """
    Compute the L matrix given the parameters
    Set size to be P (truncated)
    """
    L_MA = get_L_MA(lmbd, gamma, theta, r, s, P - p)
    L = np.zeros((P, p + r + 2 * s))
    L[:p, :p] = np.identity(p)
    L[p:, p:] = L_MA
    return L


def get_G(A, L):
    """
    Restore G from A and L
    G = A inv(L'L)L'
    """
    factor = np.matmul(np.linalg.inv(np.matmul(L.T, L)), L.T)
    G = mode_dot(A, factor, 2)
    return G


#################
# Loss function #
#################

def loss_vec(Y, X1, A, T):
    return sum(np.linalg.norm(Y - tensor_op.unfold(A[:, :, :T - 1], 1).numpy() @ X1, ord=2, axis=0) ** 2) / T

#############
# Gradients #
#############
def vec_grad_lmbd_row(N, lmbd_k, k, G_i, L, Y_col_i, X1, p, T):
    # (T-1) x 1 response vector: Y[i, :]' for row-wise estimation
    #
    # N(T-1) x (T-1) predictor matrix (lag is truncated at T-1):
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [  0, y_1, y_2, ..., y_{T-2}],
    #       [  0,   0, y_1, ..., y_{T-3}],
    #       ......
    #       [  0,   0,   0, ...,    y_1 ]]
    # 
    # Matrix form of the VAR(infty) model: Y = A_(1) X + E
    # where Y = [y_2, y_3, ..., y_T] is N x (T-1) matrix
    #       A_(1) is the mode-1 matricization of tensor A
    #       X is the counterpart of X1 without truncation and initialization
    #
    # L is T x d matrix
    # G_i is N x d matrix for row-wise estimation

    # Get the first (T-1) rows of L
    L_temp = np.copy(L[:(T - 1), :]) # (T-1) x d matrix
    # (T-1) x 1 vector of residuals for row-wise estimation
    a = Y_col_i - X1.T @ (G_i @ L_temp.T).flatten(order='F')
    
    power_series = np.arange(1, T - p) # [1, 2, ..., T-p-1]
    lmbd_power = np.power(lmbd_k, power_series - 1)  # [lmbd^0, ..., (T-p-1)*lmbd^{T-p-2}]
    lmbd_power_series = power_series * lmbd_power
    inner_grad = np.zeros(T-1)
    # inner_grad is (T-1) x 1 vector
    inner_grad[p:] = -(G_i[:, p + k] @ kron([lmbd_power_series.T, np.identity(N)])@ X1[:(N * (T - p - 1)), :(T - p - 1)]).T
    
    summand_j = 2 * np.inner(a, inner_grad)
    return summand_j / T

def vec_grad_gamma_theta_row(N, eta_k, k, G_i, L, Y_col_i, X1, p, r, T):
    gamma_k = eta_k[0]
    theta_k = eta_k[1]

    # Get the first (T-1) rows of L
    L_temp = np.copy(L[:(T - 1), :]) # (T-1) x d matrix
    # (T-1) x 1 vector of residuals for row-wise estimation
    a = Y_col_i - X1.T @ (G_i @ L_temp.T).flatten(order='F')
    
    power_series = np.arange(1, T - p) # [1, 2, ..., T-p-1]
    gamma_power = np.power(gamma_k, power_series - 1)  # [gamma^0, ..., (T-p-1)*gamma^{T-p-2}]
    gamma_power_series_cos = power_series * gamma_power * np.cos(theta_k * power_series)
    gamma_power_series_sin = power_series * gamma_power * np.sin(theta_k * power_series)
    A = G_i[:, p + r + 2 * k]
    B = G_i[:, p + r + 2 * k + 1]
    # inner_grad_gamma is (T-1) x 1 vector, and so is inner_grad_theta
    inner_grad_gamma = np.zeros(T-1)
    inner_grad_theta = np.zeros(T-1)
    inner_grad_gamma[p:] = - ( (A @ kron([gamma_power_series_cos.T, np.identity(N)]) \
                                 + B @ kron([gamma_power_series_sin.T, np.identity(N)])) @ X1[:(N * (T - p - 1)), :(T - p - 1)] ).T 
    inner_grad_theta[p:] = ( (A @ kron([gamma_power_series_sin.T * gamma_k, np.identity(N)]) \
                               - B @ kron([gamma_power_series_cos.T * gamma_k, np.identity(N)])) @ X1[:(N * (T - p - 1)), :(T - p - 1)] ).T
    
    summand_gamma = 2 * np.inner(a, inner_grad_gamma)
    summand_theta = 2 * np.inner(a, inner_grad_theta)

    return summand_gamma / T, summand_theta / T


def vec_grad_lmbd(N, lmbd_k, k, G, L, Y, X1, p, T):
    # N x (T-1) response matrix: Y = [y_2, y_3, ..., y_T]
    #
    # N(T-1) x (T-1) predictor matrix (lag is truncated at T-1):
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [  0, y_1, y_2, ..., y_{T-2}],
    #       [  0,   0, y_1, ..., y_{T-3}],
    #       ......
    #       [  0,   0,   0, ...,    y_1 ]]
    # 
    # Matrix form of the VAR(infty) model: Y = A_(1) X + E
    # where Y = [y_2, y_3, ..., y_T] is N x (T-1) matrix
    #       A_(1) is the mode-1 matricization of tensor A
    #       X is the counterpart of X1 without truncation and initialization
    #
    # L is T x d matrix
    # G is N x N x d tensor

    # Get the first (T-1) rows of L
    L_temp = np.copy(L[:(T - 1), :]) # (T-1) x d matrix
    # N x (T-1) matrix of residuals 
    a = Y - (tensor_op.unfold(mode_dot(G, L_temp, 2), 1).numpy() @ X1) 
    
    power_series = np.arange(1, T - p) # [1, 2, ..., T-p-1]
    lmbd_power = np.power(lmbd_k, power_series - 1)  # [lmbd^0, ..., (T-p-1)*lmbd^{T-p-2}]
    lmbd_power_series = power_series * lmbd_power
    inner_grad = np.zeros((N, T-1))
    inner_grad[:, p:] = -G[:, :, p + k] @ kron([lmbd_power_series.T, np.identity(N)])@ X1[:(N * (T - p - 1)), :(T - p - 1)] 
    
    summand_j = 2 * np.einsum('ij,ij->', a, inner_grad)
    return summand_j / T


def vec_grad_gamma_theta(N, eta_k, k, G, L, Y, X1, p, r, T):
    gamma_k = eta_k[0]
    theta_k = eta_k[1]

    # Get the first (T-1) rows of L
    L_temp = np.copy(L[:(T - 1), :]) # (T-1) x d matrix
    # N x (T-1) matrix of residuals
    a = Y - (tensor_op.unfold(mode_dot(G, L_temp, 2), 1).numpy() @ X1) 
    
    power_series = np.arange(1, T - p) # [1, 2, ..., T-p-1]
    gamma_power = np.power(gamma_k, power_series - 1)  # [gamma^0, ..., (T-p-1)*gamma^{T-p-2}]
    gamma_power_series_cos = power_series * gamma_power * np.cos(theta_k * power_series)
    gamma_power_series_sin = power_series * gamma_power * np.sin(theta_k * power_series)
    A = G[:, :, p + r + 2 * k]
    B = G[:, :, p + r + 2 * k + 1]
    inner_grad_gamma = np.zeros((N, T-1))
    inner_grad_theta = np.zeros((N, T-1))
    inner_grad_gamma[:, p:] = - (A @ kron([gamma_power_series_cos.T, np.identity(N)]) \
                                 + B @ kron([gamma_power_series_sin.T, np.identity(N)])) @ X1[:(N * (T - p - 1)), :(T - p - 1)]
    inner_grad_theta[:, p:] = (A @ kron([gamma_power_series_sin.T * gamma_k, np.identity(N)]) \
                               - B @ kron([gamma_power_series_cos.T * gamma_k, np.identity(N)])) @ X1[:(N * (T - p - 1)), :(T - p - 1)]
    
    summand_gamma = 2 * np.einsum('ij,ij->', a, inner_grad_gamma)
    summand_theta = 2 * np.einsum('ij,ij->', a, inner_grad_theta)

    return summand_gamma / T, summand_theta / T


###############################################################################
### Truncated version of gradients 
def vec_grad_lmbd_row_trunc(N, lmbd_k, k, G_i, L, Y_col_i, X1, p, T, P_trunc):
    # (T-1) x 1 response vector: Y[i, :]' for row-wise estimation
    #
    # Define N*P_trunc x (T-1) matrix X1 based on x
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [y_0, y_1, y_2, ..., y_{T-2}],
    #       [y_{-1}, y_0, y_1, ..., y_{T-3}],
    #       ......
    #       [y_{-(P_trunc-2)}, y_{-(P_trunc-3)},y_{-(P_trunc-4)}, ..., y_{T-P_trunc}]] 
    # L is P_trunc x d matrix
    # G_i is N x d matrix for row-wise estimation
    
    # (T-1) x 1 vector of residuals for row-wise estimation
    a = Y_col_i - X1.T @ (G_i @ L.T).flatten(order='F')
    
    power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
    lmbd_power = np.power(lmbd_k, power_series - 1)  # [lmbd^0, ..., (P_trunc-p)*lmbd^{P_trunc-p-1}]
    lmbd_power_series = power_series * lmbd_power
    inner_grad = np.zeros(T-1)
    # inner_grad is (T-1) x 1 vector
    inner_grad[p:] = -(G_i[:, p + k] @ kron([lmbd_power_series.T, np.identity(N)])@ X1[:(N * (P_trunc - p)), :(T - p - 1)]).T
    
    summand_j = 2 * np.inner(a, inner_grad)
    return summand_j / T

def vec_grad_gamma_theta_row_trunc(N, eta_k, k, G_i, L, Y_col_i, X1, p, r, T, P_trunc):
    gamma_k = eta_k[0]
    theta_k = eta_k[1]

    # (T-1) x 1 vector of residuals for row-wise estimation
    a = Y_col_i - X1.T @ (G_i @ L.T).flatten(order='F')
    
    power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
    gamma_power = np.power(gamma_k, power_series - 1)  # [gamma^0, ..., (P_trunc-p)*gamma^{P_trunc-p-1}]
    gamma_power_series_cos = power_series * gamma_power * np.cos(theta_k * power_series)
    gamma_power_series_sin = power_series * gamma_power * np.sin(theta_k * power_series)
    A = G_i[:, p + r + 2 * k]
    B = G_i[:, p + r + 2 * k + 1]
    # inner_grad_gamma is (T-1) x 1 vector, and so is inner_grad_theta
    inner_grad_gamma = np.zeros(T-1)
    inner_grad_theta = np.zeros(T-1)
    inner_grad_gamma[p:] = - ( (A @ kron([gamma_power_series_cos.T, np.identity(N)]) \
                                 + B @ kron([gamma_power_series_sin.T, np.identity(N)])) @ X1[:(N * (P_trunc - p)), :(T - p - 1)] ).T 
    inner_grad_theta[p:] = ( (A @ kron([gamma_power_series_sin.T * gamma_k, np.identity(N)]) \
                               - B @ kron([gamma_power_series_cos.T * gamma_k, np.identity(N)])) @ X1[:(N * (P_trunc - p)), :(T - p - 1)] ).T
    
    summand_gamma = 2 * np.inner(a, inner_grad_gamma)
    summand_theta = 2 * np.inner(a, inner_grad_theta)

    return summand_gamma / T, summand_theta / T


def vec_grad_lmbd_trunc(N, lmbd_k, k, G, L, Y, X1, p, T, P_trunc):
    # N x (T-1) response matrix: Y = [y_2, y_3, ..., y_T]
    #
    # Define N*P_trunc x (T-1) matrix X1 based on x
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [y_0, y_1, y_2, ..., y_{T-2}],
    #       [y_{-1}, y_0, y_1, ..., y_{T-3}],
    #       ......
    #       [y_{-(P_trunc-2)}, y_{-(P_trunc-3)},y_{-(P_trunc-4)}, ..., y_{T-P_trunc}]] 
    # L is P_trunc x d matrix 
    # G is N x N x d tensor

    # N x (T-1) matrix of residuals 
    a = Y - (tensor_op.unfold(mode_dot(G, L, 2), 1).numpy() @ X1) 
    
    power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
    lmbd_power = np.power(lmbd_k, power_series - 1)  # [lmbd^0, ..., (P_trunc-p)*lmbd^{P_trunc-p-1}]
    lmbd_power_series = power_series * lmbd_power
    inner_grad = np.zeros((N, T-1))
    inner_grad[:, p:] = -G[:, :, p + k] @ kron([lmbd_power_series.T, np.identity(N)])@ X1[:(N * (P_trunc - p)), :(T - p - 1)] 
    
    summand_j = 2 * np.einsum('ij,ij->', a, inner_grad)
    return summand_j / T


def vec_grad_gamma_theta_trunc(N, eta_k, k, G, L, Y, X1, p, r, T, P_trunc):
    gamma_k = eta_k[0]
    theta_k = eta_k[1]

    # N x (T-1) matrix of residuals
    a = Y - (tensor_op.unfold(mode_dot(G, L, 2), 1).numpy() @ X1) 
    
    ### !!! power_series = np.arange(1, T - p) # [1, 2, ..., T-p-1]
    power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
    ### !!! gamma_power = np.power(gamma_k, power_series - 1)  # [gamma^0, ..., (T-p-1)*gamma^{T-p-2}]
    gamma_power = np.power(gamma_k, power_series - 1)  # [gamma^0, ..., (T-p-1)*gamma^{P_trunc-p-1}]
    gamma_power_series_cos = power_series * gamma_power * np.cos(theta_k * power_series)
    gamma_power_series_sin = power_series * gamma_power * np.sin(theta_k * power_series)
    A = G[:, :, p + r + 2 * k]
    B = G[:, :, p + r + 2 * k + 1]
    inner_grad_gamma = np.zeros((N, T-1))
    inner_grad_theta = np.zeros((N, T-1))
    inner_grad_gamma[:, p:] = - (A @ kron([gamma_power_series_cos.T, np.identity(N)]) \
                                 + B @ kron([gamma_power_series_sin.T, np.identity(N)])) @ X1[:(N * (P_trunc - p)), :(T - p - 1)]
    inner_grad_theta[:, p:] = (A @ kron([gamma_power_series_sin.T * gamma_k, np.identity(N)]) \
                               - B @ kron([gamma_power_series_cos.T * gamma_k, np.identity(N)])) @ X1[:(N * (P_trunc - p)), :(T - p - 1)]
    
    summand_gamma = 2 * np.einsum('ij,ij->', a, inner_grad_gamma)
    summand_theta = 2 * np.einsum('ij,ij->', a, inner_grad_theta)

    return summand_gamma / T, summand_theta / T


###############################################################################
### Truncated version of gradients 2 (drop P_trunc samples, no initialization needed)
def vec_grad_lmbd_row_trunc2(N, lmbd_k, k, G_i, L, Y_col_i, X1, p, T1, P_trunc):
    # T1 x 1 response vector: Y[i, :]' for row-wise estimation
    #
    # Define N*P_trunc x T1 matrix X1 based on x, with T1=T-P_trunc
    # X1 = [[  y_{P_trunc}, ...,        y_{T-1}],
    #       [y_{P_trunc-1}, ...,        y_{T-2}],
    #                      ......
    #       [          y_1, ..., y_{T-P_trunc}]]
    # 
    # L is P_trunc x d matrix
    # G_i is N x d matrix for row-wise estimation

    # T1 x 1 vector of residuals for row-wise estimation
    a = Y_col_i - X1.T @ (G_i @ L.T).flatten(order='F')
    
    power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
    lmbd_power = np.power(lmbd_k, power_series - 1)  # [lmbd^0, ..., (P_trunc-p)*lmbd^{P_trunc-p-1}]
    lmbd_power_series = power_series * lmbd_power
    # inner_grad is T1 x 1 vector
    inner_grad = -(G_i[:, p + k] @ kron([lmbd_power_series.T, np.identity(N)])@ X1[N*p:, :T1]).T
    
    summand_j = 2 * np.inner(a, inner_grad)
    return summand_j / T1

def vec_grad_gamma_theta_row_trunc2(N, eta_k, k, G_i, L, Y_col_i, X1, p, r, T1, P_trunc):
    gamma_k = eta_k[0]
    theta_k = eta_k[1]

    # (T-1) x 1 vector of residuals for row-wise estimation
    a = Y_col_i - X1.T @ (G_i @ L.T).flatten(order='F')
    
    power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
    gamma_power = np.power(gamma_k, power_series - 1)  # [gamma^0, ..., (P_trunc-p)*gamma^{P_trunc-p-1}]
    gamma_power_series_cos = power_series * gamma_power * np.cos(theta_k * power_series)
    gamma_power_series_sin = power_series * gamma_power * np.sin(theta_k * power_series)
    A = G_i[:, p + r + 2 * k]
    B = G_i[:, p + r + 2 * k + 1]
    # inner_grad_gamma is T1 x 1 vector, and so is inner_grad_theta
    inner_grad_gamma = - ( (A @ kron([gamma_power_series_cos.T, np.identity(N)]) \
                                 + B @ kron([gamma_power_series_sin.T, np.identity(N)])) @ X1[N*p:, :T1] ).T 
    inner_grad_theta = ( (A @ kron([gamma_power_series_sin.T * gamma_k, np.identity(N)]) \
                               - B @ kron([gamma_power_series_cos.T * gamma_k, np.identity(N)])) @ X1[N*p:, :T1] ).T
    
    summand_gamma = 2 * np.inner(a, inner_grad_gamma)
    summand_theta = 2 * np.inner(a, inner_grad_theta)

    return summand_gamma / T1, summand_theta / T1


def vec_grad_lmbd_trunc2(N, lmbd_k, k, G, L, Y, X1, p, T1, P_trunc):
     # T1 x 1 response vector: Y[i, :]' for row-wise estimation
     #
     # Define N*P_trunc x T1 matrix X1 based on x, with T1=T-P_trunc
     # X1 = [[  y_{P_trunc}, ...,        y_{T-1}],
     #       [y_{P_trunc-1}, ...,        y_{T-2}],
     #                      ......
     #       [          y_1, ..., y_{T-P_trunc}]]
     # 
     # L is P_trunc x d matrix
     # G is N x N x d tensor

     # N x T1 matrix of residuals 
     a = Y - (tensor_op.unfold(mode_dot(G, L, 2), 1).numpy() @ X1) 
     
     power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
     lmbd_power = np.power(lmbd_k, power_series - 1)  # [lmbd^0, ..., (P_trunc-p)*lmbd^{P_trunc-p-1}]
     lmbd_power_series = power_series * lmbd_power
     inner_grad = -G[:, :, p + k] @ kron([lmbd_power_series.T, np.identity(N)])@ X1[N*p:, :T1]
     
     summand_j = 2 * np.einsum('ij,ij->', a, inner_grad)
     return summand_j / T1


def vec_grad_gamma_theta_trunc2(N, eta_k, k, G, L, Y, X1, p, r, T1, P_trunc):
     gamma_k = eta_k[0]
     theta_k = eta_k[1]

     # N x T1 matrix of residuals
     a = Y - (tensor_op.unfold(mode_dot(G, L, 2), 1).numpy() @ X1) 
     
     power_series = np.arange(1, P_trunc - p + 1) # [1, 2, ..., P_trunc-p]
     gamma_power = np.power(gamma_k, power_series - 1)  # [gamma^0, ..., (P_trunc-p)*gamma^{P_trunc-p-1}]
     gamma_power_series_cos = power_series * gamma_power * np.cos(theta_k * power_series)
     gamma_power_series_sin = power_series * gamma_power * np.sin(theta_k * power_series)
     A = G[:, :, p + r + 2 * k]
     B = G[:, :, p + r + 2 * k + 1]
     # inner_grad_gamma is N x T1 matrix, and so is inner_grad_theta
     inner_grad_gamma = - (A @ kron([gamma_power_series_cos.T, np.identity(N)]) \
                                  + B @ kron([gamma_power_series_sin.T, np.identity(N)])) @ X1[N*p:, :T1]
     inner_grad_theta = (A @ kron([gamma_power_series_sin.T * gamma_k, np.identity(N)]) \
                                - B @ kron([gamma_power_series_cos.T * gamma_k, np.identity(N)])) @ X1[N*p:, :T1]
     
     summand_gamma = 2 * np.einsum('ij,ij->', a, inner_grad_gamma)
     summand_theta = 2 * np.einsum('ij,ij->', a, inner_grad_theta)

     return summand_gamma / T1, summand_theta / T1
###############################################################################