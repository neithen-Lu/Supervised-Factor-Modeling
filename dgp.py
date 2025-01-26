"""
Generate N-dimensional time series {y} in AR form given length T, coefficients {Aj} with j from 1 to T
"""
import numpy as np
from scipy.stats import ortho_group
from utils.utils import unfold
import math

def DGP_VAR(N,T,burn,P,r,rho=0.7):
    """
    P: VAR order
    r: rank
    """
    A = np.zeros((N,N,P))
    eigenspace = ortho_group.rvs(N)
    for p in range(P):
        H = np.diag(np.repeat((-1)**p * math.comb(P,p+1) * np.power(rho,p+1),r))
        A[:,:,p] = eigenspace[:,:r] @ H @ eigenspace[:,:r].T
    M1=np.concatenate((np.eye((P-1)*N),np.zeros(((P-1)*N,N))),axis=1)
    comp_mat = np.concatenate((unfold(A,1), M1),0)
    val, __ = np.linalg.eig(comp_mat)
    # print("the absolute eigenvalues of the companion matrix are", np.abs(val))
    assert np.max(np.abs(val)) < 1, "stationary test not pass!"
    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(P,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = np.einsum('ijk,jk->i',A,np.flip(y[:,t-P:t],1)) + eps[:,t] # pay attention to A1 * y(t-1),... Ap * y(t-p)
    return (y[:,burn:],A)


def DGP_VAR_DIFFRANK(N,T,burn,P,r1=4,r2=8,rho=0.7):
    """
    P: VAR order
    r1,r2: rank
    """
    A = np.zeros((N,N,P))
    eigenspace1 = ortho_group.rvs(N)
    eigenspace2 = ortho_group.rvs(N)
    r = max(r1,r2)
    eigenspace1 = np.concatenate((eigenspace1[:,:r1],np.zeros((N,r-r1))))
    eigenspace2 = np.concatenate((eigenspace2[:,:r2],np.zeros((N,r-r2))))
    for p in range(P):
        H = np.identity(r)
        A[:,:,p] = np.power(rho,p) * eigenspace1 @ H @ eigenspace2.T
    M1=np.concatenate((np.eye((P-1)*N),np.zeros(((P-1)*N,N))),axis=1)
    comp_mat = np.concatenate((unfold(A,1), M1),0)
    val, __ = np.linalg.eig(comp_mat)
    # print("the absolute eigenvalues of the companion matrix are", np.abs(val))
    assert np.max(np.abs(val)) < 1, "stationary test not pass!"
    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(P,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = np.einsum('ijk,jk->i',A,np.flip(y[:,t-P:t],1)) + eps[:,t] # pay attention to A1 * y(t-1),... Ap * y(t-p)
    return (y[:,burn:],A)

def DGP_SEASON_VAR_BIC(N,T,burn,r,rho=0.7,season=4,p=1,sp=2):
    """
    only for VAR(1)xseason(2) for now
    p: VAR order
    sp: seasonal VAR order
    season: seasonal period
    r: rank
    """
    P = p+season*sp
    if rho<0.7:
        rho = 1.2*rho
    A = np.zeros((N,N,P))
    eigenspace = ortho_group.rvs(N)
    H1 = np.diag(np.repeat(rho,r))
    A[:,:,0] = eigenspace[:,:r] @ H1 @ eigenspace[:,:r].T
    H2 = np.diag(np.repeat(2*rho,r))
    A[:,:,season-1] = eigenspace[:,:r] @ H2 @ eigenspace[:,:r].T
    H3 = np.diag(np.repeat(-2*rho**2,r))
    A[:,:,season] = eigenspace[:,:r] @ H3 @ eigenspace[:,:r].T
    H4 = np.diag(np.repeat(-1*rho**2,r))
    A[:,:,season*2-1] = eigenspace[:,:r] @ H4 @ eigenspace[:,:r].T
    H5 = np.diag(np.repeat(rho**3,r))
    A[:,:,season*2] = eigenspace[:,:r] @ H5 @ eigenspace[:,:r].T

    M1=np.concatenate((np.eye((P-1)*N),np.zeros(((P-1)*N,N))),axis=1)
    comp_mat = np.concatenate((unfold(A,1), M1),0)
    val, __ = np.linalg.eig(comp_mat)
    # print("the absolute eigenvalues of the companion matrix are", np.abs(val)) 
    # if np.max(np.abs(val)) < 1:
    #     print("stationary test passed!")
    assert np.max(np.abs(val)) < 1, "stationary test not pass!"
    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(P,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = np.einsum('ijk,jk->i',A,np.flip(y[:,t-P:t],1)) + eps[:,t] # pay attention to A1 * y(t-1),... Ap * y(t-p)
    # print(y)
    return (y[:,burn:],A)

def DGP_SEASON_VAR(N,T,burn,r,rho=0.7,season=4,p=1,sp=2):
    """
    only for VAR(1)xseason(2) for now
    p: VAR order
    sp: seasonal VAR order
    season: seasonal period
    r: rank
    """
    P = p+season*sp
    A = np.zeros((N,N,P))
    eigenspace = ortho_group.rvs(N)
    H1 = np.diag(np.repeat(rho,r))
    A[:,:,0] = eigenspace[:,:r] @ H1 @ eigenspace[:,:r].T
    H2 = np.diag(np.repeat(2*rho**season,r))
    A[:,:,season-1] = eigenspace[:,:r] @ H2 @ eigenspace[:,:r].T
    H3 = np.diag(np.repeat(-2*rho**(season+1),r))
    A[:,:,season] = eigenspace[:,:r] @ H3 @ eigenspace[:,:r].T
    H4 = np.diag(np.repeat(-1*rho**(season*sp),r))
    A[:,:,season*2-1] = eigenspace[:,:r] @ H4 @ eigenspace[:,:r].T
    H5 = np.diag(np.repeat(rho**(season*sp+1),r))
    A[:,:,season*2] = eigenspace[:,:r] @ H5 @ eigenspace[:,:r].T

    M1=np.concatenate((np.eye((P-1)*N),np.zeros(((P-1)*N,N))),axis=1)
    comp_mat = np.concatenate((unfold(A,1), M1),0)
    val, __ = np.linalg.eig(comp_mat)
    # print("the absolute eigenvalues of the companion matrix are", np.abs(val)) 
    # if np.max(np.abs(val)) < 1:
    #     print("stationary test passed!")
    assert np.max(np.abs(val)) < 1, "stationary test not pass!"
    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(P,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = np.einsum('ijk,jk->i',A,np.flip(y[:,t-P:t],1)) + eps[:,t] # pay attention to A1 * y(t-1),... Ap * y(t-p)
    return (y[:,burn:],A)


def DGP_SEASON_VAR_DIFFRANK(N,T,burn,r1=4,r2=8,rho=0.7,season=4,p=1,sp=2):
    """
    only for VAR(1)xseason(2) for now
    p: VAR order
    sp: seasonal VAR order
    season: seasonal period
    r: rank
    """
    P = p+season*sp
    A = np.zeros((N,N,P))
    r_min = min(r1,r2)
    r_max = max(r1,r2)
    assert r_max <= r_min*2, 'difference between r1 and r2 should below a multiple'
    eigenspace1 = ortho_group.rvs(N)[:,:r_min]
    eigenspace2 = ortho_group.rvs(N)[:,:r_max]

    # stacked U and V
    U = np.zeros((N,r_min,P))
    V = np.zeros((N,r_min,P))

    # lag 0,3,4,7,8 have values
    U[:,:,0] = eigenspace1
    U[:,:,season-1] = eigenspace1
    U[:,:,season] = eigenspace1
    U[:,:,season*2-1] = eigenspace1
    U[:,:,season*2] = eigenspace1

    V[:,:,0] = eigenspace2[:,:r_min]
    V[:,:,season-1] = eigenspace2[:,r_max-r_min:]
    idx = np.random.randint(r_max, size=r_min)
    V[:,:,season] = eigenspace2[:,idx]
    idx = np.random.randint(r_max, size=r_min)
    V[:,:,season*2-1] = eigenspace2[:,idx]
    idx = np.random.randint(r_max, size=r_min)
    V[:,:,season*2] = eigenspace2[:,idx]

    # make U always left and V always right
    if r1 > r2:
        temp = U; U = V; V = temp

    # H1 = np.diag(np.repeat(rho,r_min))
    # A[:,:,0] = U[:,:,0] @ H1 @ V[:,:,0].T
    H2 = np.diag(np.repeat(2*rho**season,r_min))
    A[:,:,season-1] = U[:,:,season-1] @ H2 @ V[:,:,season-1].T
    H3 = np.diag(np.repeat(-2*rho**(season+1),r_min))
    A[:,:,season] = U[:,:,season] @ H3 @ V[:,:,season].T
    H4 = np.diag(np.repeat(-1*rho**(season*sp),r_min))
    A[:,:,season*2-1] = U[:,:,season*2-1] @ H4 @ V[:,:,season*2-1].T
    H5 = np.diag(np.repeat(rho**(season*sp+1),r_min))
    A[:,:,season*2] = U[:,:,season*2] @ H5 @ V[:,:,season*2].T

    M1=np.concatenate((np.eye((P-1)*N),np.zeros(((P-1)*N,N))),axis=1)
    comp_mat = np.concatenate((unfold(A,1), M1),0)
    val, __ = np.linalg.eig(comp_mat)
    # print("the absolute eigenvalues of the companion matrix are", np.abs(val)) 
    # if np.max(np.abs(val)) < 1:
    #     print("stationary test passed!")
    assert np.max(np.abs(val)) < 1, "stationary test not pass!"
    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(P,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = np.einsum('ijk,jk->i',A,np.flip(y[:,t-P:t],1)) + eps[:,t] # pay attention to A1 * y(t-1),... Ap * y(t-p)
    return (y[:,burn:],A)

def DGP_SEASON_VAR_SOFT(N,T,burn,r,rho=0.7,season=4,p=1,sp=2):
    """
    only for VAR(1)xseason(2) for now
    p: VAR order
    sp: seasonal VAR order
    season: seasonal period
    r: rank
    """
    P = p+season*sp
    A = np.zeros((N,N,P))
    eigenspace = ortho_group.rvs(N)
    H1 = np.diag(np.repeat(rho,r))
    A[:,:,0] = eigenspace[:,:r] @ H1 @ eigenspace[:,:r].T
    H2 = np.diag(np.repeat(2*rho**season,r))
    A[:,:,season-1] = eigenspace[:,:r] @ H2 @ eigenspace[:,:r].T
    H3 = np.diag(np.repeat(-2*rho**(season+1),r))
    A[:,:,season] = eigenspace[:,:r] @ H3 @ eigenspace[:,:r].T
    H4 = np.diag(np.repeat(-1*rho**(season+1),r))
    A[:,:,season*2-1] = eigenspace[:,:r] @ H4 @ eigenspace[:,:r].T
    H5 = np.diag(np.repeat(rho**(season+2),r))
    A[:,:,season*2] = eigenspace[:,:r] @ H5 @ eigenspace[:,:r].T

    M1=np.concatenate((np.eye((P-1)*N),np.zeros(((P-1)*N,N))),axis=1)
    comp_mat = np.concatenate((unfold(A,1), M1),0)
    val, __ = np.linalg.eig(comp_mat)
    # print("the absolute eigenvalues of the companion matrix are", np.abs(val)) 
    # if np.max(np.abs(val)) < 1:
    #     print("stationary test passed!")
    assert np.max(np.abs(val)) < 1, "stationary test not pass!"
    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(P,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = np.einsum('ijk,jk->i',A,np.flip(y[:,t-P:t],1)) + eps[:,t] # pay attention to A1 * y(t-1),... Ap * y(t-p)
    return (y[:,burn:],A)
# def DGP_MIX(N,T,burn,P,r,rho=0.7,pi=0.7,season=4,p=1,q=1,sp=2):
#     """
#     only for VARMA(1,1)xseason(2,0) for now
#     P: cutoff
#     p: VAR order
#     q: VMA order
#     sp: seasonal VAR order
#     season: seasonal period
#     r: rank
#     """
#     A = np.zeros((N,N,P))
#     D = np.zeros((r,r,P))
#     eigenspace = ortho_group.rvs(N)
#     J = np.diag(np.repeat(rho,r))
#     H1 = np.diag(np.repeat(pi,r))
#     H2 = np.diag(np.repeat(2*np.power(pi,4),r))
#     H3 = np.diag(np.repeat((-1)*np.power(pi,8),r))
#     for j in range(3):
#         D[:,:,j] = np.diag(np.repeat(np.power(rho, j),r)) @ (H1-J)
#         A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
#     D[:,:,3] = J @ D[:,:,2] + H2
#     A[:,:,3] = eigenspace[:,:r] @ D[:,:,3] @ eigenspace[:,:r].T
#     for j in range(4,7):
#         D[:,:,j] = np.diag(np.repeat(np.power(rho, j-4),r)) @ (J @ D[:,:,3] - H1 @ H2)
#         A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
#     D[:,:,7] = J @ D[:,:,6] + H3
#     A[:,:,7] = eigenspace[:,:r] @ D[:,:,7] @ eigenspace[:,:r].T
#     for j in range(8,P):
#         D[:,:,j] = np.diag(np.repeat(np.power(rho, j-8),r)) @ (J @ D[:,:,7] - H1 @ H3)
#         A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
#     # generate y
#     y = np.zeros((N,T+burn))
#     eps = np.zeros((N,T+burn))
#     for t in range(P,T+burn):
#         eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
#         y[:,t] = np.einsum('ijk,jk->i',A,np.flip(y[:,t-P:t],1)) + eps[:,t] # pay attention to A1 * y(t-1),... Ap * y(t-p)
#     print(np.linalg.norm(A,axis=(0,1)))
#     return (y[:,burn:],A)

def DGP_MIX_BIC(N,T,burn,P,r,rho=0.7,season=4,p=1,sp=2):
    """
    only for VMA(1)xseasonVAR(2,0) for now
    P: cutoff
    q: VMA order
    sp: seasonal VAR order
    season: seasonal period
    r: rank
    """
    pi = 0.7 # Can be changed
    eigenspace = ortho_group.rvs(N)
    J = np.diag(np.repeat(rho,r))
    H1 = np.diag(np.repeat(2*pi,r))
    H2 = np.diag(np.repeat(-1*np.power(pi,2),r))
    # generate y
    Theta = eigenspace[:,:r] @ J @ eigenspace[:,:r].T
    Phi1 = eigenspace[:,:r] @ H1 @ eigenspace[:,:r].T
    Phi2 = eigenspace[:,:r] @ H2 @ eigenspace[:,:r].T
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(8,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi1 @ y[:,t-4] + Phi2 @ y[:,t-8] - Theta @ eps[:,t-1] + eps[:,t]
    
    # generate A
    A = np.zeros((N,N,P))
    D = np.zeros((r,r,P))
    for j in range(3):
        D[:,:,j] = - np.diag(np.repeat(np.power(rho, j+1),r))
        A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
    D[:,:,3] = J @ D[:,:,2] + H1
    A[:,:,3] = eigenspace[:,:r] @ D[:,:,3] @ eigenspace[:,:r].T
    for j in range(4,7):
        D[:,:,j] = np.diag(np.repeat(np.power(rho, j-3),r)) @ D[:,:,3]
        A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
    D[:,:,7] = J @ D[:,:,6] + H2
    A[:,:,7] = eigenspace[:,:r] @ D[:,:,7] @ eigenspace[:,:r].T
    for j in range(8,P):
        D[:,:,j] = np.diag(np.repeat(np.power(rho, j-7),r)) @ D[:,:,7]
        A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
    print(np.linalg.norm(A,axis=(0,1)))
    return (y[:,burn:],A)

def DGP_MIX(N,T,burn,P,r,rho=0.7,season=4,p=1,sp=2):
    """
    only for VMA(1)xseasonVAR(2,0) for now
    P: cutoff
    q: VMA order
    sp: seasonal VAR order
    season: seasonal period
    r: rank
    """
    pi = rho # Can be changed
    eigenspace = ortho_group.rvs(N)
    J = np.diag(np.repeat(rho,r))
    H1 = np.diag(np.repeat(4*np.power(pi,4),r))
    H2 = np.diag(np.repeat((-4)*np.power(pi,8),r))
    # generate y
    Theta = eigenspace[:,:r] @ J @ eigenspace[:,:r].T
    Phi1 = eigenspace[:,:r] @ H1 @ eigenspace[:,:r].T
    Phi2 = eigenspace[:,:r] @ H2 @ eigenspace[:,:r].T
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(8,T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi1 @ y[:,t-4] + Phi2 @ y[:,t-8] - Theta @ eps[:,t-1] + eps[:,t]
    
    # generate A
    A = np.zeros((N,N,P))
    D = np.zeros((r,r,P))
    for j in range(3):
        D[:,:,j] = - np.diag(np.repeat(np.power(rho, j+1),r))
        A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
    D[:,:,3] = J @ D[:,:,2] + H1
    A[:,:,3] = eigenspace[:,:r] @ D[:,:,3] @ eigenspace[:,:r].T
    for j in range(4,7):
        D[:,:,j] = np.diag(np.repeat(np.power(rho, j-3),r)) @ D[:,:,3]
        A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
    D[:,:,7] = J @ D[:,:,6] + H2
    A[:,:,7] = eigenspace[:,:r] @ D[:,:,7] @ eigenspace[:,:r].T
    for j in range(8,P):
        D[:,:,j] = np.diag(np.repeat(np.power(rho, j-7),r)) @ D[:,:,7]
        A[:,:,j] = eigenspace[:,:r] @ D[:,:,j] @ eigenspace[:,:r].T
    return (y[:,burn:],A)

def DGP_VARMA_BIC(N,T,burn,p,r,rho=0.7):
    lmbd = np.repeat(rho,r)
    # two coef matrices
    eigenspace = ortho_group.rvs(N)
    J = np.diag(lmbd)
    H = np.diag(np.repeat(-0.9,r))

    Theta = eigenspace[:,:r] @ J @ eigenspace[:,:r].T
    Phi = eigenspace[:,:r] @ H @ eigenspace[:,:r].T

    # express as SARMA form
    if p == 1:
        B = eigenspace
        B_minus = B.T #@ (Phi-Theta)
        G = np.zeros((N,N,p+r))
        # G[:,:,0] = eigenspace[:,:r+2*s] @ (H-J) @ eigenspace[:,:r+2*s].T
        G[:,:,0] = Phi - Theta
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:]) @ (Phi-Theta)
        
    elif p == 0:
        Phi = np.zeros((N,N))
        B = eigenspace
        B_minus = -B.T
        G = np.zeros((N,N,p+r))
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])

    L = get_L(lmbd,r,T,p)
    A = np.einsum('ijk,lk->ijl',G,L)

    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi @ y[:,t-1] - Theta @ eps[:,t-1] + eps[:,t]
    return (y[:,burn:],A)

def DGP_VARMA(N,T,burn,p,r,rho=0.7):
    lmbd = np.repeat(rho,r)
    # two coef matrices
    eigenspace = ortho_group.rvs(N)
    J = np.diag(lmbd)
    H = np.diag(np.repeat(-0.5,r))

    Theta = eigenspace[:,:r] @ J @ eigenspace[:,:r].T
    Phi = eigenspace[:,:r] @ H @ eigenspace[:,:r].T

    # express as SARMA form
    if p == 1:
        B = eigenspace
        B_minus = B.T #@ (Phi-Theta)
        G = np.zeros((N,N,p+r))
        # G[:,:,0] = eigenspace[:,:r+2*s] @ (H-J) @ eigenspace[:,:r+2*s].T
        G[:,:,0] = Phi - Theta
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:]) @ (Phi-Theta)
        
    elif p == 0:
        Phi = np.zeros((N,N))
        B = eigenspace
        B_minus = -B.T
        G = np.zeros((N,N,p+r))
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])

    L = get_L(lmbd,r,T,p)
    A = np.einsum('ijk,lk->ijl',G,L)

    # generate y
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi @ y[:,t-1] - Theta @ eps[:,t-1] + eps[:,t]
    return (y[:,burn:],A,eps[:,t])

# np.random.seed(1)
# DGP_VARMA(10,100,20,0,2,0)
def get_L_MA(lmbd,r,P): # checked
    """
    Compute the L_MA matrix given the parameters
    Set size to be P (truncated)
    """
    L = np.zeros((P,r))
    for i in range(P):
        for j in range(r):
            L[i,j] = np.power(lmbd[j],i+1)
    return L

def get_L(lmbd,r,P,p): # checked
    """
    Compute the L matrix given the parameters
    Set size to be P (truncated)
    """
    L_MA = get_L_MA(lmbd,r,P-p)
    L = np.zeros((P,p+r))
    L[:p,:p] = np.identity(p)
    L[p:,p:] = L_MA
    return L

def DGP_SEASON_VARMA(N,T,burn,p,r,rho=0.7,season=3):
    lmbd = np.repeat(rho,r)
    # two coef matrices
    eigenspace = ortho_group.rvs(N)
    J = np.diag(lmbd)
    H = np.diag(np.repeat(-0.5,r))

    Theta = eigenspace[:,:r] @ J @ eigenspace[:,:r].T
    Phi = eigenspace[:,:r] @ H @ eigenspace[:,:r].T

    # express as SARMA form
    if p == 1:
        B = eigenspace
        B_minus = B.T #@ (Phi-Theta)
        G = np.zeros((N,N,p+r))
        # G[:,:,0] = eigenspace[:,:r+2*s] @ (H-J) @ eigenspace[:,:r+2*s].T
        G[:,:,0] = Phi - Theta
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:]) @ (Phi-Theta)
        

    elif p == 0:
        Phi = np.zeros((N,N))
        B = eigenspace
        B_minus = -B.T
        G = np.zeros((N,N,p+r))
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])

    L = get_season_L(lmbd,r,T,p,season)
    A = np.einsum('ijk,lk->ijl',G,L)

    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi @ y[:,t-season] - Theta @ eps[:,t-season] + eps[:,t]
    return (y[:,burn:],A)

def get_season_L_MA(lmbd,r,P,season):
    """
    Compute the L_MA matrix given the parameters
    Set size to be P (truncated)
    """
    L = np.zeros((P,r))
    for i in range(P):
        if i % season == season-1:
            for j in range(r):
                L[i,j] = np.power(lmbd[j],(i+1)/season)
    return L

def get_season_L(lmbd,r,P,p,season): # checked
    """
    Compute the L matrix given the parameters
    Set size to be P (truncated)
    """
    L_MA = get_season_L_MA(lmbd,r,P-p,season)
    L = np.zeros((P,p+r))
    L[:p,:p] = np.identity(p)
    L[p:,p:] = L_MA
    return L

if __name__ == "__main__":
    # DGP_VAR(N=10,T=1000,burn=200,P=4,r=4,rho=0.8)
    # print(DGP_SEASON_VAR(N=10,T=100,burn=200,r=4,rho=0.8))
    # result=DGP_MIX(N=10,T=100,burn=200,P=36,r=4,rho=0.7,pi=0.6)
    # DGP_VARMA(N=10,T=100,burn=200,p=1,r=4,rho=0.7)
    # DGP_SEASON_VARMA(N=10,T=100,burn=200,p=1,r=4,rho=0.7,season=3)
    result = DGP_MIX_BIC(N=10,T=100,burn=200,P=36,r=4,rho=0.7)