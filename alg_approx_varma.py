"""
Block coordinate descent algorithm for omega and G
"""

import numpy as np
import scipy
import tensorly as tl
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
from help_func import *

###############################################################################
### Rowwise estimation without lag truncation
def BCD_row(y, p, r, s, lmbd_g, n_iter, lr_omega, y_init,\
            lmbd_init, gamma_init, theta_init, A_init, G_init, L_init, \
            stop_thres, stop_method, epsilon=0.05):
    stop_thres1 = stop_thres
    stop_thres2 = stop_thres
    # y = [y_1, y_2, ..., y_T], N x T data matrix
	# Initialization for y_init = [y_{-(T-3)}, ..., y_{-1}, y_0], N X (T-2) matrix
    # model orders: p, r, s
    # n_iter: max number of iterations
    # lr_omega: step length
    # initial values of parameters: lmbd_init, gamma_init, theta_init, G_init
    # epsilon: small number to prevent touching boundaries of parameter space
    N = y.shape[0]
    T = y.shape[1]
    d = p + r + 2 * s
    # Loss = np.inf
    Loss = np.zeros(N)

    # Define the soft thresholding function
    def soft_threshold_all(beta):
        if beta > lr_omega * lmbd_g:
            return beta - lr_omega * lmbd_g
        elif beta < -lr_omega * lmbd_g:
            return beta + lr_omega * lmbd_g
        else:
            return 0.0

    # Initialization for omega, N copies (columns), and G
    lmbd_all = np.repeat(np.array(lmbd_init)[:,np.newaxis], N, 1)
    gamma_all = np.repeat(np.array(gamma_init)[:,np.newaxis], N, 1)
    theta_all = np.repeat(np.array(theta_init)[:,np.newaxis], N, 1)
    G = np.copy(G_init)
    
    # Initialization for L and A
    L = np.copy(L_init)
    A = np.copy(A_init) # N x N x T tensor

    # Response Y = [y_2, y_3, ..., y_T], N x (T-1) matrix
    Y = y[:, 1:]
	
	# Concatenate matrices y and y_init horizontally to get [y_{-(T-3)}, ..., y_T], N x (2T-2) matrix
    y_complete = np.concatenate((y_init, y), axis=1)
    
    # Flip the time order of the N x (2T - 2) data matrix y_complete and then vectorize to get
    # x = [y_T', ..., y_1', y_0', ..., y_{-(T-3)}']', N(2T-2) x 1 vector
    x = np.reshape(np.flip(y_complete, axis=1), (-1, 1), order='F')  # vectorized y_complete
    
    # Define N(T-1) x (T-1) matrix X1 based on x
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [y_0, y_1, y_2, ..., y_{T-2}],
    #       [y_{-1}, y_0, y_1, ..., y_{T-3}],
    #       ......
    #       [y_{-(T-3)}, y_{-(T-2)},y_{-(T-1)}, ...,    y_1 ]]
    # which stores the corresponding predictors for the Y_col
    #
    # Matrix form of the VAR(infty) model: Y = A_(1) X + E
    # where Y = [y_2, y_3, ..., y_T] is N x (T-1) matrix
    #       A_(1) is the mode-1 matricization of tensor A
    #       X is the counterpart of X1 without truncation and initialization
    #
    # Vector form of the model: Y_col = (X' \otimes I_N) vec(A_(1)) + E_col
    # where Y_col is the vectorization of Y
    
    X1 = np.zeros((N * (T - 1), T - 1))  
    for i in range(T - 1):
        X1[:, i:i + 1] = x[(T - 1 - i) * N: (2 * T - 2 - i) * N]
        
    power_series = np.arange(1, T - p + 1)  
     
    # Block coordinate descent if rows are estimated separately
    flag_maxiter = np.zeros(N) 
    iter_no_row = np.zeros(N)  
    
    for i in range(N):
        Y_col_i = Y[i, :].T # Y_col_i = [y_{i,2}, y_{i,3}, ..., y_{i,T}]', (T-1) x 1 vector
        '''
        Model for the ith row:
            Y_col_i = X'a_i + E_col_i, 
        where a_i' is the ith row of A(1)
        '''
        # Initialization for omega_i and g_i
        lmbd_i = np.copy(lmbd_init[:])
        gamma_i = np.copy(gamma_init[:])
        theta_i = np.copy(theta_init[:])
        G_i = np.copy(G_init[i,:,:]) # ith horizontal slide of G, N x d matrix
        A_i = A_init[i, :, :] # N x T matrix   
        
        for iter_no in range(n_iter):
            # pre_A = np.copy(A) # previous iterate of A
            # pre_G = np.copy(G) # previous iterate of G
            pre_lmbd_i = np.copy(lmbd_i) # previous iterate of lmbd
            # Update lmbd
            for k in range(r):    
                # Get gradient for lmbd
                grad = vec_grad_lmbd_row(N, lmbd_i[k], k, G_i, L, Y_col_i, X1, p, T) 
                temp_lmbd_i = lmbd_i[k] - lr_omega * grad          
                lmbd_i[k] = sorted([1 - epsilon, -1 + epsilon, temp_lmbd_i])[1]
                # Update L
                L[p:, p + k] = np.power(lmbd_i[k], power_series)
 
            pre_gamma_i = np.copy(gamma_i) # previous iterate of gamma
            pre_theta_i = np.copy(theta_i) # previous iterate of theta
            # update gamma and theta
            for k in range(s):
                # Get gradients for gamma and theta
                grad_gamma, grad_theta = vec_grad_gamma_theta_row(N, [gamma_i[k], theta_i[k]], k, G_i, L, Y_col_i, X1, p, r, T)
                temp_gamma_i = gamma_i[k] - lr_omega * grad_gamma
                temp_theta_i = theta_i[k] - lr_omega * grad_theta
                gamma_i[k] = sorted([1 - epsilon, 0, temp_gamma_i])[1]
                theta_i[k] = sorted([np.pi / 2 - epsilon, -np.pi / 2 + epsilon, temp_theta_i])[1]
                # Update L
                L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma_i[k], power_series), np.cos(power_series * theta_i[k]))
                L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma_i[k], power_series), np.sin(power_series * theta_i[k]))
 
            
            pre_G_i = np.copy(G_i) # previous iterate of G_i
 
            X_g = X1.T @ kron([L[:T - 1, :], np.identity(N)])
            # Update G_i
            G_i_vec = G_i.flatten(order='F') # Nd x 1 vector g_i
            G_i_grad_vec = 2 * X_g.T @ (X_g @ G_i_vec - Y_col_i) / T # Nd x 1 vector
            G_i_vec_update = (G_i_vec - lr_omega * G_i_grad_vec).reshape((-1))
            # Update G_i with soft-thresholding
            G_i = np.array(list(map(soft_threshold_all, G_i_vec_update))).reshape((N, d), order='F')
 
            pre_A_i = np.copy(A_i) # previous iterate of A_i, ith horizontal slice of A  !!!!!!
            A_i = G_i @ L.T # N x T matrix
            
            # Track convergence
            print("Row: ", i, [np.linalg.norm(G_i - pre_G_i, ord='fro'),
                   np.linalg.norm(np.concatenate([lmbd_i - pre_lmbd_i, gamma_i - pre_gamma_i, theta_i - pre_theta_i]), ord=2)])
            # Stopping criteria
            if (stop_method == 'SepEst') and (np.linalg.norm(G_i - pre_G_i, ord='fro') < stop_thres1) and (
                    np.linalg.norm(np.concatenate([lmbd_i - pre_lmbd_i, gamma_i - pre_gamma_i, theta_i - pre_theta_i]), ord=2) < stop_thres2):
                break;
            elif (stop_method == 'Est') and (np.linalg.norm(A_i - pre_A_i, ord='fro') < stop_thres):
                break;
            
        A[i,:,:] = A_i    
        G[i,:,:] = G_i
        a = Y_col_i - X1.T @ A_i[:, :(T - 1)].flatten(order='F')
        Loss[i] = sum(a ** 2) / T
        lmbd_all[:,i] = lmbd_i
        gamma_all[:,i] = gamma_i
        theta_all[:,i] = theta_i
        iter_no_row[i] = iter_no
        print("Row: ", i, "No. of iter: ", iter_no + 1)
        if (iter_no == n_iter - 1):
            print("Row: ", i, "Max iteration number reached")
            flag_maxiter[i] = 1
    
    # lmbd = lmbd_all.mean(axis=1)
    # gamma = gamma_all.mean(axis=1)
    # theta = theta_all.mean(axis=1)
    
    ## Compute final estimate of A and average squared loss
    # L = get_L(lmbd, gamma, theta, r, s, T, p)  
    # A = tl.tenalg.mode_dot(G, L, 2)
    # Loss = loss_vec(Y, X1, A, T)

    # return lmbd, gamma, theta, G, Loss, flag_maxiter, iter_no_row
    return lmbd_all, gamma_all, theta_all, G, A, Loss, flag_maxiter, iter_no_row

###############################################################################
### Joint estimation without lag truncation
def BCD(y, p, r, s, lmbd_g, n_iter, lr_omega, y_init,\
        lmbd_init, gamma_init, theta_init, A_init, G_init, L_init, \
        stop_thres, stop_method, epsilon=0.05):
    stop_thres1 = stop_thres
    stop_thres2 = stop_thres
    # y = [y_1, y_2, ..., y_T], N x T data matrix
    # model orders: p, r, s
    # n_iter: max number of iterations
    # lr_omega: step length
    # initial values of parameters: lmbd_init, gamma_init, theta_init, G_init
    # epsilon: small number to prevent touching boundaries of parameter space
    N = y.shape[0]
    T = y.shape[1]
    d = p + r + 2 * s
    Loss = np.inf
    
    # Define the soft thresholding function
    def soft_threshold_all(beta):
        if beta > lr_omega * lmbd_g:
            return beta - lr_omega * lmbd_g
        elif beta < -lr_omega * lmbd_g:
            return beta + lr_omega * lmbd_g
        else:
            return 0.0

    # Initialization for omega and G
    lmbd = np.copy(lmbd_init[:])
    gamma = np.copy(gamma_init[:])
    theta = np.copy(theta_init[:])
    G = np.copy(G_init)
    # Initialization for L and A
    L = np.copy(L_init)
    A = A_init

    # Response Y = [y_2, y_3, ..., y_T], N x (T-1) matrix
    Y = y[:, 1:]
    # Vectorize Y to Y_col = [y_2', y_3', ..., y_T']', N(T-1) x 1 vector
    Y_col = np.reshape(Y, (-1, 1), order='F') 
    
	# Concatenate matrices y and y_init horizontally to get [y_{-(T-3)}, ..., y_T], N x (2T-2) matrix
    y_complete = np.concatenate((y_init, y), axis=1)
    
    # Flip the time order of the N x (2T - 2) data matrix y_complete and then vectorize to get
    # x = [y_T', ..., y_1', y_0', ..., y_{-(T-3)}']', N(2T-2) x 1 vector
    x = np.reshape(np.flip(y_complete, axis=1), (-1, 1), order='F')  # vectorized y_complete
    
    # Define N(T-1) x (T-1) matrix X1 based on x
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [y_0, y_1, y_2, ..., y_{T-2}],
    #       [y_{-1}, y_0, y_1, ..., y_{T-3}],
    #       ......
    #       [y_{-(T-3)}, y_{-(T-2)},y_{-(T-1)}, ...,    y_1 ]]
    # which stores the corresponding predictors for the Y_col
    #
    # Matrix form of the VAR(infty) model: Y = A_(1) X + E
    # where Y = [y_2, y_3, ..., y_T] is N x (T-1) matrix
    #       A_(1) is the mode-1 matricization of tensor A
    #       X is the counterpart of X1 without truncation and initialization
    #
    # Vector form of the model: Y_col = (X' \otimes I_N) vec(A_(1)) + E_col
    # where Y_col is the vectorization of Y
    
    X1 = np.zeros((N * (T - 1), T - 1))  
    for i in range(T - 1):
        X1[:, i:i + 1] = x[(T - 1 - i) * N: (2 * T - 2 - i) * N]
    
    power_series = np.arange(1, T - p + 1)  
    
    # Block coordinate descent if all rows are estimated simultaneously
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd) # previous iterate of lmbd
        # Update lmbd
        for k in range(r):    
            # Get gradient for lmbd
            grad = vec_grad_lmbd(N, lmbd[k], k, G, L, Y, X1, p, T) 
            temp_lmbd = lmbd[k] - lr_omega * grad          
            lmbd[k] = sorted([1 - epsilon, -1 + epsilon, temp_lmbd])[1]
            # Update L
            L[p:, p + k] = np.power(lmbd[k], power_series)

        pre_gamma = np.copy(gamma) # previous iterate of gamma
        pre_theta = np.copy(theta) # previous iterate of theta
        # update gamma and theta
        for k in range(s):
            # Get gradients for gamma and theta
            grad_gamma, grad_theta = vec_grad_gamma_theta(N, [gamma[k], theta[k]], k, G, L, Y, X1, p, r, T)
            temp_gamma = gamma[k] - lr_omega * grad_gamma
            temp_theta = theta[k] - lr_omega * grad_theta
            gamma[k] = sorted([1 - epsilon, 0, temp_gamma])[1]
            theta[k] = sorted([np.pi / 2 - epsilon, -np.pi / 2 + epsilon, temp_theta])[1]
            # Update L
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))


        pre_G = np.copy(G) # previous iterate of G
        
        X_g = kron([X1.T @ kron([L[:T - 1, :], np.identity(N)]), np.identity(N)])

        # Update G
        G_vec = tensor_op.unfold(G, 1).numpy().reshape((-1, 1), order='F')
        G_grad_vec = 2 * X_g.T @ (X_g @ G_vec - Y_col) / T

        G_vec_update = (G_vec - lr_omega * G_grad_vec).reshape((-1))
        # Update G with soft-thresholding
        G = np.array(list(map(soft_threshold_all, G_vec_update))).reshape((N, N, d), order='F')

        pre_A = np.copy(A) # previous iterate of A
        A = tl.tenalg.mode_dot(G, L, 2)
        
        # Track convergence
        # print([np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'),
        #        np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2)])

        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres1) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres2):
            break;
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            break;

    # print("No. of iter: ", iter_no + 1)
    # if (iter_no == n_iter - 1):
    #     print("Max iteration number reached")
    #     flag_maxiter = 1

    Loss = loss_vec(Y, X1, A, T)
    residual = Y - tensor_op.unfold(A[:, :, :T - 1], 1).numpy() @ X1
    
    return lmbd, gamma, theta, G, Loss, flag_maxiter, iter_no, residual

###############################################################################
###############################################################################
### Rowwise estimation with lag truncation
def BCD_row_trunc(y, p, r, s, lmbd_g, n_iter, P_trunc, lr_omega, y_init,\
                  lmbd_init, gamma_init, theta_init, A_init, G_init, L_init, \
                  stop_thres, stop_method, epsilon=0.05):
    stop_thres1 = stop_thres
    stop_thres2 = stop_thres
    # y = [y_1, y_2, ..., y_T], N x T data matrix
	# Initialization for y_init = [y_{-(P_trunc-2)}, ..., y_{-1}, y_0], N X (P_trunc-1) matrix
    # model orders: p, r, s
    # n_iter: max number of iterations
    # lr_omega: step length
    # initial values of parameters: lmbd_init, gamma_init, theta_init, G_init
    # epsilon: small number to prevent touching boundaries of parameter space
    N = y.shape[0]
    T = y.shape[1]
    d = p + r + 2 * s
    # Loss = np.inf
    Loss = np.zeros(N)

    # Define the soft thresholding function
    def soft_threshold_all(beta):
        if beta > lr_omega * lmbd_g:
            return beta - lr_omega * lmbd_g
        elif beta < -lr_omega * lmbd_g:
            return beta + lr_omega * lmbd_g
        else:
            return 0.0

    # Initialization for omega, N copies (columns), and G
    lmbd_all = np.repeat(np.array(lmbd_init)[:,np.newaxis], N, 1)
    gamma_all = np.repeat(np.array(gamma_init)[:,np.newaxis], N, 1)
    theta_all = np.repeat(np.array(theta_init)[:,np.newaxis], N, 1)
    G = np.copy(G_init)
    
    # Initialization for L and A
    L = L_init[:P_trunc, :]  # P_trunc x d matrix
    A = A_init[:, :, :P_trunc]  # N x N x P_trunc tensor
    A_full = np.copy(A_init) # N x N x T tensor
    
    
    # Response Y = [y_2, y_3, ..., y_T], N x (T-1) matrix
    Y = y[:, 1:]
	
	# Concatenate matrices y and y_init horizontally to get [y_{-(P_trunc-2)}, ..., y_T], N x (T+P_trunc-1) matrix
    y_complete = np.concatenate((y_init, y), axis=1)
    
    # Flip the time order of the N x (T+P_trunc-1) data matrix y_complete and then vectorize to get
    # x = [y_T', ..., y_1', y_0', ..., y_{-(P_trunc-2)}']', N(T+P_trunc-1) x 1 vector
    x = np.reshape(np.flip(y_complete, axis=1), (-1, 1), order='F')  # vectorized y_complete
    
    # Define N*P_trunc x (T-1) matrix X1 based on x
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [y_0, y_1, y_2, ..., y_{T-2}],
    #       [y_{-1}, y_0, y_1, ..., y_{T-3}],
    #       ......
    #       [y_{-(P_trunc-2)}, y_{-(P_trunc-3)},y_{-(P_trunc-4)}, ..., y_{T-P_trunc}]]
    # which stores the corresponding predictors for the Y_col
    #
    # Matrix form of the VAR(infty) model: Y = A_(1) X + E
    # where Y = [y_2, y_3, ..., y_T] is N x (T-1) matrix
    #       A_(1) is the mode-1 matricization of tensor A
    #       X is the counterpart of X1 without truncation and initialization
    #
    # Vector form of the model: Y_col = (X' \otimes I_N) vec(A_(1)) + E_col
    # where Y_col is the vectorization of Y
    
    X1 = np.zeros((N * P_trunc, T - 1))  
    for i in range(T - 1):
        X1[:, i:i + 1] = x[(T - 1 - i) * N: (T - 1 - i + P_trunc) * N]
        
    power_series = np.arange(1, P_trunc - p + 1)  # [1, 2, ..., P_trunc-p]
     
    # Block coordinate descent if rows are estimated separately
    flag_maxiter = np.zeros(N) 
    iter_no_row = np.zeros(N)  
    for i in range(N):
        Y_col_i = Y[i, :].T # Y_col_i = [y_{2,T}, y_{3,T}, ..., y_{i,T}]', (T-1) x 1 vector
        '''
        Model for the ith row:
            Y_col_i = X'a_i + E_col_i, 
        where a_i' is the ith row of A(1)
        '''
        # Initialization for omega_i and g_i
        lmbd_i = np.copy(lmbd_init[:])
        gamma_i = np.copy(gamma_init[:])
        theta_i = np.copy(theta_init[:])
        G_i = np.copy(G_init[i,:,:]) # ith horizontal slide of G, N x d matrix
        A_i = A[i, :, :] # N x P_trunc matrix
              
        for iter_no in range(n_iter):
            # pre_A = np.copy(A) # previous iterate of A
            # pre_G = np.copy(G) # previous iterate of G
            pre_lmbd_i = np.copy(lmbd_i) # previous iterate of lmbd
            # Update lmbd
            for k in range(r):    
                # Get gradient for lmbd
                grad = vec_grad_lmbd_row_trunc(N, lmbd_i[k], k, G_i, L, Y_col_i, X1, p, T, P_trunc) 
                temp_lmbd_i = lmbd_i[k] - lr_omega * grad          
                lmbd_i[k] = sorted([1 - epsilon, -1 + epsilon, temp_lmbd_i])[1]
                # Update L
                L[p:, p + k] = np.power(lmbd_i[k], power_series)
 
            pre_gamma_i = np.copy(gamma_i) # previous iterate of gamma
            pre_theta_i = np.copy(theta_i) # previous iterate of theta
            # update gamma and theta
            for k in range(s):
                # Get gradients for gamma and theta
                grad_gamma, grad_theta = vec_grad_gamma_theta_row_trunc(N, [gamma_i[k], theta_i[k]], k, G_i, L, Y_col_i, X1, p, r, T, P_trunc)
                temp_gamma_i = gamma_i[k] - lr_omega * grad_gamma
                temp_theta_i = theta_i[k] - lr_omega * grad_theta
                gamma_i[k] = sorted([1 - epsilon, 0, temp_gamma_i])[1]
                theta_i[k] = sorted([np.pi / 2 - epsilon, -np.pi / 2 + epsilon, temp_theta_i])[1]
                # Update L
                L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma_i[k], power_series), np.cos(power_series * theta_i[k]))
                L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma_i[k], power_series), np.sin(power_series * theta_i[k]))
 
            
            pre_G_i = np.copy(G_i) # previous iterate of G_i
 
            X_g = X1.T @ kron([L, np.identity(N)])
            # Update G_i
            G_i_vec = G_i.flatten(order='F') # Nd x 1 vector g_i
            G_i_grad_vec = 2 * X_g.T @ (X_g @ G_i_vec - Y_col_i) / T # Nd x 1 vector
            G_i_vec_update = (G_i_vec - lr_omega * G_i_grad_vec).reshape((-1))
            # Update G_i with soft-thresholding
            G_i = np.array(list(map(soft_threshold_all, G_i_vec_update))).reshape((N, d), order='F')
 
            pre_A_i = np.copy(A_i) # previous iterate of A_i, ith horizontal slice of A

            A_i = G_i @ L.T # N x P_trunc matrix
                      
            # Track convergence
            print("Row: ", i, [np.linalg.norm(G_i - pre_G_i, ord='fro'),
                   np.linalg.norm(np.concatenate([lmbd_i - pre_lmbd_i, gamma_i - pre_gamma_i, theta_i - pre_theta_i]), ord=2)])
            # Stopping criteria
            if (stop_method == 'SepEst') and (np.linalg.norm(G_i - pre_G_i, ord='fro') < stop_thres1) and (
                    np.linalg.norm(np.concatenate([lmbd_i - pre_lmbd_i, gamma_i - pre_gamma_i, theta_i - pre_theta_i]), ord=2) < stop_thres2):
                break;
            elif (stop_method == 'Est') and (np.linalg.norm(A_i - pre_A_i, ord='fro') < stop_thres):
                break;
                
        # Compute T x d matrix L
        L_full = get_L(lmbd_i, gamma_i, theta_i, r, s, T, p)
        A_full[i,:,:] = G_i @ L_full.T
        A[i,:,:] = A_i 
        G[i,:,:] = G_i
        a = Y_col_i - X1.T @ A[i, :, :(T - 1)].flatten(order='F')
        Loss[i] = sum(a ** 2) / T
        lmbd_all[:,i] = lmbd_i
        gamma_all[:,i] = gamma_i
        theta_all[:,i] = theta_i
        iter_no_row[i] = iter_no
        # print("Row: ", i, "No. of iter: ", iter_no + 1)
        # if (iter_no == n_iter - 1):
        #     print("Row: ", i, "Max iteration number reached")
        #     flag_maxiter[i] = 1
    
    # lmbd = lmbd_all.mean(axis=1)
    # gamma = gamma_all.mean(axis=1)
    # theta = theta_all.mean(axis=1)
    
    ## Compute loss
    # L = get_L(lmbd, gamma, theta, r, s, P_trunc, p)  
    # A = tl.tenalg.mode_dot(G, L, 2)  # N x N x P_trunc tensor   
    # Loss = loss_vec(Y, X1, A, T)

    # return lmbd, gamma, theta, G, Loss, flag_maxiter, iter_no_row
    return lmbd_all, gamma_all, theta_all, G, A_full, Loss, flag_maxiter, iter_no_row


###############################################################################
### Joint estimation with lag truncation
def BCD_trunc(y, p, r, s, lmbd_g, n_iter, P_trunc, lr_omega, y_init,\
              lmbd_init, gamma_init, theta_init, A_init, G_init, L_init, \
              stop_thres, stop_method, epsilon=0.05):
    stop_thres1 = stop_thres
    stop_thres2 = stop_thres
    # y = [y_1, y_2, ..., y_T], N x T data matrix
    # model orders: p, r, s
    # n_iter: max number of iterations
    # lr_omega: step length
    # initial values of parameters: lmbd_init, gamma_init, theta_init, G_init
    # epsilon: small number to prevent touching boundaries of parameter space
    N = y.shape[0]
    T = y.shape[1]
    d = p + r + 2 * s
    Loss = np.inf
    
    # Define the soft thresholding function
    def soft_threshold_all(beta):
        if beta > lr_omega * lmbd_g:
            return beta - lr_omega * lmbd_g
        elif beta < -lr_omega * lmbd_g:
            return beta + lr_omega * lmbd_g
        else:
            return 0.0

    # Initialization for omega and G
    lmbd = np.copy(lmbd_init[:])
    gamma = np.copy(gamma_init[:])
    theta = np.copy(theta_init[:])
    G = np.copy(G_init)

    # Initialization for L and A
    L = L_init[:P_trunc, :]  # P_trunc x d matrix
    A = A_init[:, :, :P_trunc]  # N x N x P_trunc tensor

    # Response Y = [y_2, y_3, ..., y_T], N x (T-1) matrix
    Y = y[:, 1:]
    # Vectorize Y to Y_col = [y_2', y_3', ..., y_T']', N(T-1) x 1 vector
    Y_col = np.reshape(Y, (-1, 1), order='F') 
    
    # Concatenate matrices y and y_init horizontally to get [y_{-(P_trunc-2)}, ..., y_T], N x (T+P_trunc-1) matrix
    y_complete = np.concatenate((y_init, y), axis=1)
    
    # Flip the time order of the N x (T+P_trunc-1) data matrix y_complete and then vectorize to get
    # x = [y_T', ..., y_1', y_0', ..., y_{-(P_trunc-2)}']', N(T+P_trunc-1) x 1 vector
    x = np.reshape(np.flip(y_complete, axis=1), (-1, 1), order='F')  # vectorized y_complete
    
    # Define N*P_trunc x (T-1) matrix X1 based on x
    # X1 = [[y_1, y_2, y_3, ..., y_{T-1}],
    #       [y_0, y_1, y_2, ..., y_{T-2}],
    #       [y_{-1}, y_0, y_1, ..., y_{T-3}],
    #       ......
    #       [y_{-(P_trunc-2)}, y_{-(P_trunc-3)},y_{-(P_trunc-4)}, ..., y_{T-P_trunc}]]
    # which stores the corresponding predictors for the Y_col
    #
    # Matrix form of the VAR(infty) model: Y = A_(1) X + E
    # where Y = [y_2, y_3, ..., y_T] is N x (T-1) matrix
    #       A_(1) is the mode-1 matricization of tensor A
    #       X is the counterpart of X1 without truncation and initialization
    #
    # Vector form of the model: Y_col = (X' \otimes I_N) vec(A_(1)) + E_col
    # where Y_col is the vectorization of Y
    
    X1 = np.zeros((N * P_trunc, T - 1))  
    for i in range(T - 1):
        X1[:, i:i + 1] = x[(T - 1 - i) * N: (T - 1 - i + P_trunc) * N]
    
    power_series = np.arange(1, P_trunc - p + 1)  # [1, 2, ..., P_trunc-p]
    
    # Block coordinate descent if all rows are estimated simultaneously
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd) # previous iterate of lmbd
        # Update lmbd
        for k in range(r):    
            # Get gradient for lmbd
            grad = vec_grad_lmbd_trunc(N, lmbd[k], k, G, L, Y, X1, p, T, P_trunc) 
            temp_lmbd = lmbd[k] - lr_omega * grad          
            lmbd[k] = sorted([1 - epsilon, -1 + epsilon, temp_lmbd])[1]
            # Update L
            L[p:, p + k] = np.power(lmbd[k], power_series)

        pre_gamma = np.copy(gamma) # previous iterate of gamma
        pre_theta = np.copy(theta) # previous iterate of theta
        # update gamma and theta
        for k in range(s):
            # Get gradients for gamma and theta
            grad_gamma, grad_theta = vec_grad_gamma_theta_trunc(N, [gamma[k], theta[k]], k, G, L, Y, X1, p, r, T, P_trunc)
            temp_gamma = gamma[k] - lr_omega * grad_gamma
            temp_theta = theta[k] - lr_omega * grad_theta
            gamma[k] = sorted([1 - epsilon, 0, temp_gamma])[1]
            theta[k] = sorted([np.pi / 2 - epsilon, -np.pi / 2 + epsilon, temp_theta])[1]
            # Update L
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))


        pre_G = np.copy(G) # previous iterate of G
        
        X_g = kron([X1.T @ kron([L, np.identity(N)]), np.identity(N)])

        # Update G
        G_vec = tensor_op.unfold(G, 1).numpy().reshape((-1, 1), order='F')
        G_grad_vec = 2 * X_g.T @ (X_g @ G_vec - Y_col) / T

        G_vec_update = (G_vec - lr_omega * G_grad_vec).reshape((-1))
        # Update G with soft-thresholding
        G = np.array(list(map(soft_threshold_all, G_vec_update))).reshape((N, N, d), order='F')

        pre_A = np.copy(A) # previous iterate of A
        A = tl.tenalg.mode_dot(G, L, 2)
        
        # Track convergence
        # print([np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'),
        #        np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2)])

        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres1) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres2):
            break;
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            break;

    # print("No. of iter: ", iter_no + 1)
    # if (iter_no == n_iter - 1):
    #     print("Max iteration number reached")
    #     flag_maxiter = 1

    Loss = loss_vec(Y, X1, A, T)
    
    return lmbd, gamma, theta, G, Loss, flag_maxiter, iter_no


###############################################################################
###############################################################################
### Rowwise estimation with lag truncation method 2 - Not Used
### (drop P_trunc samples, no initial value needed)
def BCD_row_trunc2(y, p, r, s, lmbd_g, n_iter, P_trunc, lr_omega, \
                   lmbd_init, gamma_init, theta_init, A_init, G_init, L_init, \
                   stop_thres, stop_method, epsilon=0.05):
    stop_thres1 = stop_thres
    stop_thres2 = stop_thres
    # y = [y_1, y_2, ..., y_T], N x T data matrix
    # model orders: p, r, s
    # n_iter: max number of iterations
    # lr_omega: step length
    # initial values of parameters: lmbd_init, gamma_init, theta_init, G_init
    # epsilon: small number to prevent touching boundaries of parameter space
    N = y.shape[0]
    T = y.shape[1]
    T1 = T - P_trunc
    d = p + r + 2 * s
    Loss = np.inf
    
    # Define the soft thresholding function
    def soft_threshold_all(beta):
        if beta > lr_omega * lmbd_g:
            return beta - lr_omega * lmbd_g
        elif beta < -lr_omega * lmbd_g:
            return beta + lr_omega * lmbd_g
        else:
            return 0.0

    # Initialization for omega, N copies (columns), and G
    lmbd_all = np.repeat(np.array(lmbd_init)[:,np.newaxis], N, 1)
    gamma_all = np.repeat(np.array(gamma_init)[:,np.newaxis], N, 1)
    theta_all = np.repeat(np.array(theta_init)[:,np.newaxis], N, 1)
    G = np.copy(G_init)
    
    # Initialization for L and A
    L = L_init[:P_trunc, :]  # P_trunc x d matrix
    A = A_init[:, :, :P_trunc]  # N x N x P_trunc tensor

    # Response Y = [y_{P_trunc+1}, y_{P_trunc+2}, ..., y_T], N x T1 matrix, with T1=T-P_trunc
    Y = y[:, P_trunc:]
    
    # Flip the time order of the N x T data matrix y and then vectorize to get
    # x = [y_T', y_{T-1}', ..., y_1']', NT x 1 vector
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  
    
    # Define N*P_trunc x T1 matrix X1 based on x, with T1=T-P_trunc
    # X1 = [[  y_{P_trunc}, ...,        y_{T-1}],
    #       [y_{P_trunc-1}, ...,        y_{T-2}],
    #                      ......
    #       [          y_1, ..., y_{T-P_trunc}]]
    # which stores the corresponding predictors for the Y_col
    #
    # Matrix form of the VAR(P_trunc) model: Y = A_(1) X + E
    # where Y = [y_{P_trunc+1}, y_{P_trunc+2}, ..., y_T], N x T1 matrix, with T1=T-P_trunc
    #       A_(1) is the mode-1 matricization of tensor A
    #
    # Vector form of the model: Y_col = (X' \otimes I_N) vec(A_(1)) + E_col
    # where Y_col is the vectorization of Y
    
    X1 = np.zeros((N * P_trunc, T1))  
    for i in range(T1):
        X1[:, i:i + 1] = x[(T - i - P_trunc) * N : (T - i) * N]
        
    power_series = np.arange(1, P_trunc - p + 1)  # [1, 2, ..., P_trunc-p]
     
    # Block coordinate descent if rows are estimated separately
    flag_maxiter = np.zeros(N) 
    iter_no_row = np.zeros(N)  
    for i in range(N):
        Y_col_i = Y[i, :].T # Y_col_i = [y_{i,P_trunc+1}, y_{i,P_trunc+2}, ..., y_{i,T}]', T1 x 1 vector
        '''
        Model for the ith row:
            Y_col_i = X'a_i + E_col_i, 
        where a_i' is the ith row of A(1)
        '''
        # Initialization for omega_i and g_i
        lmbd_i = np.copy(lmbd_init[:])
        gamma_i = np.copy(gamma_init[:])
        theta_i = np.copy(theta_init[:])
        G_i = np.copy(G_init[i,:,:]) # ith horizontal slide of G, N x d matrix
                   
        for iter_no in range(n_iter):
            # pre_A = np.copy(A) # previous iterate of A
            # pre_G = np.copy(G) # previous iterate of G
            pre_lmbd_i = np.copy(lmbd_i) # previous iterate of lmbd
            # Update lmbd
            for k in range(r):    
                # Get gradient for lmbd
                grad = vec_grad_lmbd_row_trunc2(N, lmbd_i[k], k, G_i, L, Y_col_i, X1, p, T1, P_trunc) 
                temp_lmbd_i = lmbd_i[k] - lr_omega * grad          
                lmbd_i[k] = sorted([1 - epsilon, -1 + epsilon, temp_lmbd_i])[1]
                # Update L
                L[p:, p + k] = np.power(lmbd_i[k], power_series)
 
            pre_gamma_i = np.copy(gamma_i) # previous iterate of gamma
            pre_theta_i = np.copy(theta_i) # previous iterate of theta
            # update gamma and theta
            for k in range(s):
                # Get gradients for gamma and theta
                grad_gamma, grad_theta = vec_grad_gamma_theta_row_trunc2(N, [gamma_i[k], theta_i[k]], k, G_i, L, Y_col_i, X1, p, r, T1, P_trunc)
                temp_gamma_i = gamma_i[k] - lr_omega * grad_gamma
                temp_theta_i = theta_i[k] - lr_omega * grad_theta
                gamma_i[k] = sorted([1 - epsilon, 0, temp_gamma_i])[1]
                theta_i[k] = sorted([np.pi / 2 - epsilon, -np.pi / 2 + epsilon, temp_theta_i])[1]
                # Update L
                L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma_i[k], power_series), np.cos(power_series * theta_i[k]))
                L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma_i[k], power_series), np.sin(power_series * theta_i[k]))
 
            
            pre_G_i = np.copy(G_i) # previous iterate of G_i
 
            X_g = X1.T @ kron([L, np.identity(N)])  # T1 x Nd matrix
            # Update G_i
            G_i_vec = G_i.flatten(order='F') # Nd x 1 vector g_i
            G_i_grad_vec = 2 * X_g.T @ (X_g @ G_i_vec - Y_col_i) / T # Nd x 1 vector
            G_i_vec_update = (G_i_vec - lr_omega * G_i_grad_vec).reshape((-1))
            # Update G_i with soft-thresholding
            G_i = np.array(list(map(soft_threshold_all, G_i_vec_update))).reshape((N, d), order='F')
 
            pre_A_i = np.copy(A[i,:,:]) # previous iterate of A_i, ith horizontal slice of A
            A_i = G_i @ L.T # N x P_trunc matrix
            
            # Track convergence
            # print("Row: ", i, [np.linalg.norm(G_i - pre_G_i, ord='fro'),
            #        np.linalg.norm(np.concatenate([lmbd_i - pre_lmbd_i, gamma_i - pre_gamma_i, theta_i - pre_theta_i]), ord=2)])
            # Stopping criteria
            if (stop_method == 'SepEst') and (np.linalg.norm(G_i - pre_G_i, ord='fro') < stop_thres1) and (
                    np.linalg.norm(np.concatenate([lmbd_i - pre_lmbd_i, gamma_i - pre_gamma_i, theta_i - pre_theta_i]), ord=2) < stop_thres2):
                break;
            elif (stop_method == 'Est') and (np.linalg.norm(A_i - pre_A_i, ord='fro') < stop_thres):
                break;
                
        # A[i,:,:] = A_i
        G[i,:,:] = G_i
        lmbd_all[:,i] = lmbd_i
        gamma_all[:,i] = gamma_i
        theta_all[:,i] = theta_i
        iter_no_row[i] = iter_no
        print("Row: ", i, "No. of iter: ", iter_no + 1)
        if (iter_no == n_iter - 1):
            print("Row: ", i, "Max iteration number reached")
            flag_maxiter[i] = 1
    
    # Final estimates
    lmbd = lmbd_all.mean(axis=1)
    gamma = gamma_all.mean(axis=1)
    theta = theta_all.mean(axis=1)
     
    # Compute loss
    L = get_L(lmbd, gamma, theta, r, s, P_trunc, p)  
    A = tl.tenalg.mode_dot(G, L, 2)  # N x N x P_trunc tensor
    Loss = loss_vec(Y, X1, A, T1)

    return lmbd, gamma, theta, G, Loss, flag_maxiter, iter_no_row


###############################################################################
### Joint estimation with lag truncation method 2 - Not used
### (drop P_trunc samples, no initial value needed)
def BCD_trunc2(y, p, r, s, lmbd_g, n_iter, P_trunc, lr_omega, \
               lmbd_init, gamma_init, theta_init, A_init, G_init, L_init, \
               stop_thres, stop_method, epsilon=0.05):
    stop_thres1 = stop_thres
    stop_thres2 = stop_thres
    # y = [y_1, y_2, ..., y_T], N x T data matrix
    # model orders: p, r, s
    # n_iter: max number of iterations
    # lr_omega: step length
    # initial values of parameters: lmbd_init, gamma_init, theta_init, G_init
    # epsilon: small number to prevent touching boundaries of parameter space
    N = y.shape[0]
    T = y.shape[1]
    T1 = T - P_trunc
    d = p + r + 2 * s
    Loss = np.inf
    
    # Define the soft thresholding function
    def soft_threshold_all(beta):
        if beta > lr_omega * lmbd_g:
            return beta - lr_omega * lmbd_g
        elif beta < -lr_omega * lmbd_g:
            return beta + lr_omega * lmbd_g
        else:
            return 0.0

    # Initialization for omega and G
    lmbd = np.copy(lmbd_init[:])
    gamma = np.copy(gamma_init[:])
    theta = np.copy(theta_init[:])
    G = np.copy(G_init)
    
    # Initialization for L and A
    L = L_init[:P_trunc, :]  # P_trunc x d matrix
    A = A_init[:, :, :P_trunc]  # N x N x P_trunc tensor

    # Response Y = [y_{P_trunc+1}, y_{P_trunc+2}, ..., y_T], N x T1 matrix, with T1=T-P_trunc
    Y = y[:, P_trunc:]
    # Vectorize Y to Y_col = [y_{P_trunc+1}', y_{P_trunc+2}', ..., y_T']', N*T1 x 1 vector
    Y_col = np.reshape(Y, (-1, 1), order='F') 
    
    # Flip the time order of the N x T data matrix y and then vectorize to get
    # x = [y_T', y_{T-1}', ..., y_1']', NT x 1 vector
    x = np.reshape(np.flip(y, axis=1), (-1, 1), order='F')  
    
    # Define N*P_trunc x T1 matrix X1 based on x, with T1=T-P_trunc
    # X1 = [[  y_{P_trunc}, ...,        y_{T-1}],
    #       [y_{P_trunc-1}, ...,        y_{T-2}],
    #                      ......
    #       [          y_1, ..., y_{T-P_trunc}]]
    # which stores the corresponding predictors for the Y_col
    #
    # Matrix form of the VAR(P_trunc) model: Y = A_(1) X + E
    # where Y = [y_{P_trunc+1}, y_{P_trunc+2}, ..., y_T], N x T1 matrix, with T1=T-P_trunc
    #       A_(1) is the mode-1 matricization of tensor A
    #
    # Vector form of the model: Y_col = (X' \otimes I_N) vec(A_(1)) + E_col
    # where Y_col is the vectorization of Y
    
    X1 = np.zeros((N * P_trunc, T1))  
    for i in range(T1):
        X1[:, i:i + 1] = x[(T - i - P_trunc) * N : (T - i) * N]
    
    power_series = np.arange(1, P_trunc - p + 1)  # [1, 2, ..., P_trunc-p]
    
    # Block coordinate descent if all rows are estimated simultaneously
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd) # previous iterate of lmbd
        # Update lmbd
        for k in range(r):    
            # Get gradient for lmbd
            grad = vec_grad_lmbd_trunc2(N, lmbd[k], k, G, L, Y, X1, p, T1, P_trunc) 
            temp_lmbd = lmbd[k] - lr_omega * grad          
            lmbd[k] = sorted([1 - epsilon, -1 + epsilon, temp_lmbd])[1]
            # Update L
            L[p:, p + k] = np.power(lmbd[k], power_series)

        pre_gamma = np.copy(gamma) # previous iterate of gamma
        pre_theta = np.copy(theta) # previous iterate of theta
        # update gamma and theta
        for k in range(s):
            # Get gradients for gamma and theta
            grad_gamma, grad_theta = vec_grad_gamma_theta_trunc2(N, [gamma[k], theta[k]], k, G, L, Y, X1, p, r, T1, P_trunc)
            temp_gamma = gamma[k] - lr_omega * grad_gamma
            temp_theta = theta[k] - lr_omega * grad_theta
            gamma[k] = sorted([1 - epsilon, 0, temp_gamma])[1]
            theta[k] = sorted([np.pi / 2 - epsilon, -np.pi / 2 + epsilon, temp_theta])[1]
            # Update L
            L[p:, p + r + 2 * k] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.cos(power_series * theta[k]))
            L[p:, p + r + 2 * k + 1] = np.einsum('i,i->i', np.power(gamma[k], power_series), np.sin(power_series * theta[k]))


        pre_G = np.copy(G) # previous iterate of G
        
        X_g = kron([X1.T @ kron([L, np.identity(N)]), np.identity(N)])

        # Update G
        G_vec = tensor_op.unfold(G, 1).numpy().reshape((-1, 1), order='F')
        G_grad_vec = 2 * X_g.T @ (X_g @ G_vec - Y_col) / T

        G_vec_update = (G_vec - lr_omega * G_grad_vec).reshape((-1))
        # Update G with soft-thresholding
        G = np.array(list(map(soft_threshold_all, G_vec_update))).reshape((N, N, d), order='F')

        pre_A = np.copy(A) # previous iterate of A
        A = tl.tenalg.mode_dot(G, L, 2)
        
        # Track convergence
        print([np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro'),
               np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2)])

        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G - pre_G, 1), ord='fro') < stop_thres1) and (
                np.linalg.norm(np.concatenate([lmbd - pre_lmbd, gamma - pre_gamma, theta - pre_theta]), ord=2) < stop_thres2):
            break;
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A - pre_A, 1), ord='fro') < stop_thres):
            break;

    print("No. of iter: ", iter_no + 1)
    if (iter_no == n_iter - 1):
        print("Max iteration number reached")
        flag_maxiter = 1

    Loss = loss_vec(Y, X1, A, T)
   
    return lmbd, gamma, theta, G, Loss, flag_maxiter, iter_no