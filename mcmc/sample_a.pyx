#!/usr/bin/env python
# coding: utf-8

from scipy.special import beta as BETA
import numpy as np
cimport numpy as np


def get_Nikt(A, K, Nt):
    """
        Returns n_kt^{i} for every i,k,t
    """
    maxNt, D, T = A.shape
    Nikt = np.zeros((maxNt, K, T), dtype=np.int32) - 1
    for t in range(T):
        for i in range(Nt[t]):
            for k in range(K):
                Nikt[i,k,t] = sum(A[i,:,t] == k)
    return Nikt

def get_Ndkv(np.ndarray A, np.ndarray W, np.ndarray Nt, int K):
    """
        Return N_dk^v  (# of the dth disease presence status out of all dth disease slots assigned to the kth cluster)
        Notation
            k : cluster
            d : disease
            t : time
        Note
            see the screenshot for the detail
    """
    cdef int maxNt, D, T
    maxNt = A.shape[0]
    D = A.shape[1]
    T = A.shape[2]
    
    cdef np.ndarray Nkdv = np.zeros((D, K, 2), dtype=np.int32)
    cdef int [:,:,:] Nkdv_view = Nkdv
    
    for k in range(K):
        for d in range(D):
            for v in range(2):
                Nkdv_view[d,k,v] = sum(W[:,d,:][A[:,d,:] == k] == v) # N_dk^{v}
    return Nkdv


def update_A(np.ndarray A, np.ndarray W, np.ndarray Ndkv, np.ndarray Nikt, np.ndarray Z, np.ndarray Phi, np.ndarray Nt, double eta1, double eta2):
    """
        Return a sample from the posterior distribution of A

    """     
    cdef int [:,:,:] W_view = W
    cdef int [:,:,:] A_view = A
    cdef int [:,:,:] Ndkv_view = Ndkv
    cdef int [:,:,:] Nikt_view = Nikt
    cdef int [:,:,:] Z_view = Z
    cdef double [:] Phi_view = Phi
    cdef int [:] Nt_view = Nt
    
    
    
    maxNt = A.shape[0]
    D = A.shape[1]
    T = A.shape[2]
    
    K = Z.shape[1]
    
    
    cdef np.ndarray eta = np.zeros((K,2))
    cdef np.ndarray Anew = np.zeros((maxNt, D, T), dtype=np.int32) - 1

    cdef int t, i, d, k    
    
    eta[:,0] = np.repeat(eta2, K)
    eta[:,1] = np.repeat(eta1, K)
    
    
    for t in range(T):
        for i in range(Nt[t]):
            
            if A_view[i,0,t] < 0:
                continue
                
            Flag = np.zeros((D,K))
            for d in range(D):
                Nikt_view[i,A[i,d,t],t] = max(Nikt_view[i,A[i,d,t],t]-1, 0)
                Ndkv_view[d,A[i,d,t],W_view[i,d,t]] = max(Ndkv_view[d,A[i,d,t],W_view[i,d,t]]-1, 0)
                
                p = (Nikt_view[i,:,t] + np.multiply(Z_view[i,:,t], Phi_view)) * (eta[:,W_view[i,d,t]] + Ndkv_view[d,:,W_view[i,d,t]]) / (np.sum(eta,axis=1) + np.sum(Ndkv_view[d], axis=1))
                
                for k in range(K):
                    if Z_view[i,k,t] == 0:
                        p[k] = 0  

                Anew[i,d,t] = np.random.choice(K, size=1, p=p/np.sum(p))
                Nikt_view[i,Anew[i,d,t],t] += 1
                Ndkv_view[d,Anew[i,d,t],W_view[i,d,t]] += 1
    return Anew

