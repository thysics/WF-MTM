#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.special import binom
from scipy.special import loggamma, comb
from scipy.special import beta as BETA

# This sampler works fine!!


def update_Phi(Phi, Z, Nikt, Nt, D, Count, trueGam, propScale):
    """
        returns a sample from the posterior joint distribution of Phi and Gamma
        
        Note
            if no active cluster is present across individual for specific k and t,
                we reject the proposal
            Count
                dictionary; count[(k,t)] == # of acceptance
            lgamma is used since gamma.logpdf (from scipy) underflows
            
    """
    maxNt, K, T = Z.shape
    Phinew = np.zeros(K) - 1
    
    def get_ZPhi(Z, Phi, Nt):
        """
            Returns phi[t,k] * Z[i,k,t]
        """
        maxNt, K, T = Z.shape

        ZPhi = np.zeros((maxNt, K, T)) - 1

        for t in range(T):
            for k in range(K):
                ZPhi[:Nt[t], k, t] = Z[:Nt[t],k,t] * Phi[k] # Each entry is phi[t,k] * Z[i,k,t]
        return ZPhi
    
    ZPhi = get_ZPhi(Z, Phi, Nt)
    
    for k in range(K):
        Phinew[k], accept = logMH(current = Phi[k], Phisum = np.sum(np.delete(ZPhi, k, axis=1), axis=1).reshape(-1)                                   , Zk = Z[:,k,:].reshape(-1), Nk = Nikt[:,k,:].reshape(-1), D=D, trueGam=trueGam, propScale=propScale)
        Count[k] += accept        
    return Phinew, Count

def logMH(current, Phisum, Zk, Nk, D, trueGam, propScale):
    accept = 0
    new = current
    propose = np.random.normal(loc = current, scale = propScale)
    if propose > 0:
        u = np.random.uniform(0,1,1)[0]
        p = logjoint(propose, Phisum, Zk, Nk, D, trueGam) - logjoint(current, Phisum, Zk, Nk, D, trueGam)
#         print("u:{} p:{}".format(np.log(u), p))
        if p == 0:
#             print("check!")
            return new, accept
        if np.log(u) < p:
            new = propose
            accept += 1
    return new, accept

def logjoint(Phi, Phisum, Zk, Nk, D, trueGam):
    prior = lgamma(Phi,trueGam,1)
    likelihood = 0
    for i in range(len(Zk)):
        if np.sum(Zk) == 0:
            continue
        # if Phisum[i] = 0, then n^{i}_{kt} is D w.p 1. (since no other cluster exists)
        if (Zk[i] == 1) and (Phisum[i] > 0):
#             print("Nk: {}".format(Nk))
#             print("Nk[i]: {}".format(Nk[i]))
            likelihood += np.log(comb(D, Nk[i]))                             + np.log(BETA(Nk[i] + Phi, D - Nk[i] + Phisum[i]))                             - np.log(BETA(Phi, Phisum[i]))
#             print(np.log(comb(D, Nk[i])), np.log(BETA(Nk[i] + Phi, D - Nk[i] + Phisum[i])),  np.log(BETA(Phi, Phisum[i])))
#     print("prior: {} likelihood: {}".format(prior, likelihood))
    return prior + likelihood

def lgamma(x, gamm, be):
    return (gamm - 1) * np.log(x) - be * x - loggamma(gamm) + gamm * np.log(be)

