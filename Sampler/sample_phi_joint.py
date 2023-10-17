import numpy as np
from scipy.special import binom
from scipy.special import loggamma, comb

def lnBeta(Array):
    assert type(Array).__module__ == np.__name__, "input should be numpy.array"
    Array = Array[Array > 0]

    if len(Array) == 1:
        return 0

    return sum([loggamma(item) for item in Array]) - loggamma(sum(Array))

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


def update_Phi(Phi, Z, Nikt, D, Count, trueGam, propScale):
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

    Phinew, accept = logMH(current=Phi, Z=Z, N=Nikt, D=D, trueGam=trueGam, propScale = propScale)
    Count += accept

    return Phinew, Count


def logMH(current, Z, N, D, trueGam, propScale):
    accept = 0
    new = current
    propose = np.random.normal(loc=current, scale=propScale, size=len(current))
    if all(propose > 0):
        u = np.random.uniform(0, 1, 1)[0]
        p = logjoint(propose, Z, N, D, trueGam) - logjoint(current, Z, N, D, trueGam)
        #         print("u:{} p:{}".format(np.log(u), p))
        #         print("{}".format(np.log(u) < p))
        if p == 0:
            return new, accept
        if np.log(u) < p:
            new = propose
            accept += 1
    return new, accept


def prob_nit(Zit, Nit, Phi):
    if (sum(Zit) <= 0) | (sum(Nit) <= 0) :
        return 0
    if all([Zit[item] > 0 for item in np.argwhere(Nit > 0)]):
        term1 = np.log(multinomial(Nit[Nit > 0]))
        term2 = lnBeta(Zit * (Phi + Nit))
        term3 = lnBeta(Zit * Phi)
        return term1 + term2 - term3
    print("ERROR")
    return False


def logjoint(Phi, Z, N, D, trueGam):
    prior = sum(lgamma(Phi, trueGam, 1.))

    _, K, T = Z.shape
    likelihood = 0
    for t in range(T):
        likelihood += sum(
            np.apply_along_axis(lambda x: prob_nit(x[:K], x[K:], Phi=Phi), 1, np.hstack([Z[:, :, t], N[:, :, t]])))

    return prior + likelihood


def lgamma(x, gamm, be):
    return (gamm - 1) * np.log(x) - be * x - loggamma(gamm) + gamm * np.log(be)

