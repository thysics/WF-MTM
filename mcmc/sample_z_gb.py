from scipy.special import beta as BETA
import numpy as np

def update_Z(Z, Phi, X, Nt, Nikt, D):
    """
        Returns a sample from the posterior distribution of Z

    """
    maxNt, K, T = Z.shape

    Znew = Z.copy()

    for t in range(T):
        for i in range(Nt[t]):
            if sum(Nikt[i, :, t]) <= 0:  # if it is healthy individual: ENCODE HEALTHY ONES WITH -1
                Znew[i, :, t] = np.repeat(0, K)
                continue
            # To improve mixing, we choose the first entry to update at random.
            temp = np.arange(K)
            np.random.shuffle(temp)
            for k in temp:
                if Nikt[i, k, t] > 0:
                    Znew[i, k, t] = 1
                else:
                    _Phi = sum(Znew[i, :k, t] * Phi[:k]) + sum(Znew[i, k + 1:, t] * Phi[k + 1:])

                    Ctilda = BETA(Phi[k], D + _Phi) / BETA(Phi[k], _Phi)
                    # To prevent underflow
                    Ctilda = max(Ctilda, 1e-99)
                    Znew[i, k, t] = np.random.binomial(1, p=X[k, t] / (X[k, t] + ((1 - X[k, t]) / Ctilda)), size=1)
    return Znew