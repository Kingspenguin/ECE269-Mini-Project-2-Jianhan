import numpy as np
import random
import scipy
from math import log
from tqdm import tqdm

signal_choices = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]

def gen_A(m, n, mean=0, var=1):

    A = np.random.normal(mean, var, (m,n))
    for j in range(A.shape[1]):  
        col_norm = np.linalg.norm(A[:,j])
        for i in range(A.shape[0]): 
            A[i,j] /= col_norm

    return A


def gen_x(s, N):

    v = np.zeros(N)
    index_set = np.random.choice(range(N), s, replace=False)

    for i in index_set:
        v[i] = np.random.choice(signal_choices)

    return v, index_set


def OMP(A,y,stop=np.infty,error_thresh=0.01):

    M = A.shape[0]
    N = A.shape[1]    
    r = np.copy(y)
    x_pre = np.zeros(N)
    Sup_A = []
    runs = 0
    
    while np.linalg.norm(r)>error_thresh and runs<stop:
        scores = A.T.dot(r)
        idx = np.argmax(abs(scores))
        Sup_A.append(idx)
        basis = A[:,Sup_A]
        x_pre[Sup_A] = np.linalg.inv(np.dot(basis.T,basis)).dot(basis.T).dot(y)
        r = y - A.dot(x_pre)        
        runs += 1
    return x_pre.T,Sup_A

def gen_noise(vector, variance):
    noise = np.random.normal(0, variance, vector.shape)
    noise_norm = np.linalg.norm(noise)
    return vector + noise, noise_norm

def OMP_rec(A, y, error_thresh=0.01, stop= 100000):

    r = np.copy(y)
    x_pre = np.zeros_like(r)
    Sup_A = np.array([], dtype=int)
    runs = 0
    while (np.linalg.norm(r) > error_thresh and runs < stop):
        inner = np.squeeze(np.abs(np.inner(A.transpose(), r.transpose())))
        lambdak = np.argmax(inner)
        Sup_A=np.append(Sup_A,lambdak)
        basis = A[:, Sup_A]
        x_pre = np.linalg.pinv(basis).dot(y)
        r = y - basis.dot(x_pre)
        runs +=1
    return x_pre, Sup_A