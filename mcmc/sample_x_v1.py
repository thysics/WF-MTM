import numpy as np

def exact_WF(x, theta1, theta2, diff_unit):
    """
        Simulate WF-diffusion from (x,t)
        The function assumes that theta1 + theta2 != 1
    """

    beta = (1 / 2) * (theta1 + theta2 - 1) * diff_unit
    eta = beta / (np.exp(beta) - 1)
    mu = 2 * eta / diff_unit
    var = mu * (eta + beta) ** 2 * (1 + (eta / (eta + beta)) - 2 * eta) * (beta ** (-2))

    # Simulate A (Ancestral process)
    M_ = np.random.normal(loc=mu, scale=var, size=1)
    M = max(int(M_), 0)
    L = np.random.binomial(n=M, p=x)

    # to prevent underflow, add 1e-99 to the result
    return np.random.beta(theta1 + L, theta2 + M - L) + 1e-99


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


def resample(newtrj, k, ntk, docnum, prewgt, string='normal'):
    """
    resample particles for a given feature k
    prevent underflow, we use log-sum-exp trick
    Input:
        newtrj: trajectory simulated from time t-1 to time t + reference trajectory at time t
        alc: array that includes feature allociation for a given time t, 1*K
        docnum: number of objects for a given time t
        k: given feature k
    Output: resampled particles and their indices
    """
    M = len(newtrj)
    wgt = np.zeros(M)
    logwgt = np.zeros(M)
    logwgt = ntk[k] * np.log(newtrj) + (docnum - ntk[k]) * np.log((np.repeat(1, M) - newtrj)) + prewgt

    # how to deal with the weight of the reference trajectory?
    if string == 'normal':
        newwgt = logsumexp(logwgt[:M - 1])
        indice = np.random.choice(np.arange(M - 1), size=M - 1, replace=True, p=newwgt)
        retrj = np.take(newtrj, indice)
        wgt = np.take(logwgt, np.append(indice, M - 1))
        return retrj, indice, wgt

    else:
        newwgt = logsumexp(logwgt)
        indice = np.random.choice(np.arange(M), size=1, replace=True, p=newwgt)
        retrj = np.take(newtrj, indice)
        return retrj, indice


def update_X_exact(ref, Z, Nt_Dis, M, theta1, theta2, diffusion_unit, Nt_NoDis):
    """
    Particle Filtering to draw a sample of trajectories
    Input:
        ref: reference trajectory
        Z: cluster allocation matrix
        Nt_Dis: # of subjects with at least one disease across time
        N: # of obv. across time (excluding null individual)
        step, dt, alpha, beta: parameters for WF-diffusion
        M: number of particles
        diffusion unit: dt
        Nt_NoDis: # of subjects without any diseases in the dataset
    Output:
        newref: new reference trajectory
    """
    maxNt, K, T = Z.shape

    ptc = np.zeros((M + 1, K, T))
    trj = np.zeros((M + 1, K, T))
    wgt = np.zeros((M + 1, K, T))

    ind = np.zeros((M + 1, K, T)).astype(int) + M
    newtrj = np.zeros((K, T))

    # correctly compute n_tk
    n_tk = [np.sum(Z[:Nt_Dis[t], :, t], axis=0) for t in range(T)]

    Nt = Nt_Dis + Nt_NoDis
    Nt = Nt.astype(np.int32)
    # Intialization

    for k in range(K):
        ptc[:M, k, 0] = np.random.beta(a=theta1 + n_tk[0][k], b=theta2 + Nt[0] - n_tk[0][k], size=M)
        wgt[:, k, 0] = np.zeros(M + 1)  # we save log (weight)

    ptc[M, :, :] = ref
    trj[M, :, :] = ref

    trj[:, :, 0] = ptc[:, :, 0]

    # SMC step
    for t in range(T - 2):
        for k in range(K):
            ptc[:M, k, t + 1] = list(map(lambda x: exact_WF(x, theta1, theta2, diffusion_unit), trj[:M, k, t]))
            # Resample
            if isinstance(n_tk[t + 1], np.ndarray) == False:
                trj[:M, k, t + 1] = ptc[:M, k, t + 1]
                ind[:M, k, t + 1] = np.arange(M)
                wgt[:, k, t + 1] = wgt[:, k, t]
                continue
            trj[:M, k, t + 1], ind[:M, k, t + 1], wgt[:, k, t + 1] = resample(ptc[:, k, t + 1], k, n_tk[t + 1],
                                                                              Nt[t + 1], wgt[:, k, t], string='normal')

    # Final step
    for k in range(K):
        ptc[:M, k, T - 1] = list(map(lambda x: exact_WF(x, theta1, theta2, diffusion_unit), trj[:M, k, T - 2]))
        newtrj[k, T - 1], find = resample(ptc[:, k, T - 1], k, n_tk[T - 1], Nt[T - 1], wgt[:, k, T - 2], string='final')

        find = find[0].astype(int)
        for t in reversed(range(T - 1)):
            index = ind[find, k, t]
            newtrj[k, t] = trj[find, k, t]
            find = index

    return newtrj