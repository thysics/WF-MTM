#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import bernoulli
from itertools import chain, combinations
from scipy.special import loggamma
from scipy.special import binom

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])

def lnBeta(Array):
    assert type(Array).__module__ == np.__name__, "input should be numpy.array"
    Array = Array[Array > 0]

    if len(Array) == 1:
        return 0

    return sum([loggamma(item) for item in Array]) - loggamma(sum(Array))

def get_HBsample(X, idx):
    U = X.copy()
    if len(idx) > 0:
        U[np.array(idx)] -= 1
    return abs(U)

def compute_newZit(Zit, Nit, Xt, Phi ):
    if all([Zit[item] > 0 for item in np.argwhere(Nit > 0)]):
        term1 = np.log(multinomial(Nit[Nit > 0]))
        term2 = lnBeta(Zit*Phi + Nit)
        term3 = lnBeta(Zit*Phi)
        prior = sum(bernoulli.logpmf(Zit, p=Xt))

        return term1 + term2 - term3 + prior
    return -1e11


def logsumexp(x):
    """
    input
        x: log of weight
    output
        normalized weight
    """
    maxwgt = np.max(x)
    M = len(x)
    const = maxwgt + np.log(np.sum(np.exp(x - np.repeat(maxwgt, M))))
    unwgt = x - np.repeat(const, M)

    # newwgt: noramlized weight among particles (except for the reference)
    newwgt = np.exp(unwgt)

    return newwgt

def update_Z_Hammingball(X, Phi, Nikt, Nt, R):
    """
        Additional element; R: hammingball radius
        Row-wise update Z_{it} when K is small (less than 4)
        It computes probability for each Z_it relying on the exhaustive enumerations
    """
    maxNt, K, T = Nikt.shape
    newZ = np.zeros((maxNt, K, T), dtype=np.int32)

    for t in range(T):
        for i in range(Nt[t]):
            if sum(Nikt[i, :, t]) <= 0:
                newZ[i, :, t] = np.repeat(0, K)
                continue
            # First conduct determinstic update (Z(i,t,k) = 1 if N(i,t,k) = 1)
            base = (Nikt[i, :, t] > 0).astype(np.int32)

            # Conduct Gibbs update (for Z(i,t,k) such that N(i,t,k) = 0)

            # Restrict the vector of topic that will be updated
            HB = list(chain(*(combinations(np.arange(K)[Nikt[i, :, t] == 0], x) for x in range(R + 1))))

            # HB sampler step

            # 1. Draw U | X_{t}
            # HB: all possible indices that Z can change within Hammingball with radius m
            # Uidx: choose one index (will turn into U) from HB
            Uidx = HB[np.random.choice(np.arange(len(HB)), size=1)[0]]
            U = get_HBsample(base, Uidx)

            # 2. Draw X_{t+1} | U
            vals = [get_HBsample(U, idx) for idx in HB]

            pvals = list(map(lambda x: compute_newZit(np.array(x), Nikt[i, :, t], X[:, t], Phi), vals))
            #             print(vals, np.round(logsumexp(pvals),3))
            newZ[i, :, t] = vals[np.random.choice(np.arange(len(vals)), p=logsumexp(pvals))]

    return newZ

