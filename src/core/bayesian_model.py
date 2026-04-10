"""
Core Bayesian model functions for NHPP mixture analysis.

Implements posterior probability calculations, cluster assignments,
and chain diagnostics for Hierarchical DP-Mixture Poisson processes.
"""

import numpy as np
import mpmath as mp
from scipy.stats import multivariate_normal
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from .config import (
    Cadena,
    SimPoisson,
    Tmax,
    pT,
    L,
    alpha0,
    IW,
    Gaus,
    GammaPrior,
)


def BetaInv(alpha):
    """Compute the beta function B(alpha)."""
    return mp.beta(*[mp.mpf(x) for x in alpha])


def GammaVectorized(x):
    """Compute gamma function for array input."""
    return [mp.gamma(mp.mpf(xi)) for xi in x]


def NormalZ(i):
    """
    Assign cluster label to event i via probabilistic selection.
    
    Parameters
    ----------
    i : int
        Event index
    
    Returns
    -------
    int
        Assigned cluster label (0 to L-1)
    """
    Zprob = np.zeros(L)
    for lindex in range(L):
        Zprob[lindex] = (
            Cadena[-1][1][int(SimPoisson[i, 3])][lindex]
            * multivariate_normal.pdf(
                SimPoisson[i, [0, 1]],
                mean=Cadena[-1][4][lindex][0],
                cov=Cadena[-1][4][lindex][1],
            )
        )
    dens = Zprob[:] / np.sum(Zprob[:])
    return np.random.choice(L, p=dens)


def AsignarZ_Todos(SimPoisson_data, ultimo_estado, L_val):
    """
    Vectorized cluster assignment for all events.
    
    Parameters
    ----------
    SimPoisson_data : ndarray
        Spatio-temporal point data (N x 4): [lon, lat, time, time_partition]
    ultimo_estado : tuple
        Latest MCMC state [Alpha, Beta, gammaParam, Z, Phi]
    L_val : int
        Number of spatial clusters
    
    Returns
    -------
    ndarray
        Cluster assignments for all events
    """
    N = SimPoisson_data.shape[0]
    pesos = ultimo_estado[1]  # Beta weights
    Phi = ultimo_estado[4]  # Cluster parameters

    indices_t = SimPoisson_data[:, 3].astype(int)
    posiciones = SimPoisson_data[:, [0, 1]]

    # Compute densities for all points
    probs = np.zeros((N, L_val))
    for l in range(L_val):
        mu = Phi[l][0]
        cov = Phi[l][1]
        densidad = multivariate_normal.pdf(posiciones, mean=mu, cov=cov)
        probs[:, l] = pesos[indices_t, l] * densidad

    # Normalize
    sumas = probs.sum(axis=1)
    probs /= sumas[:, np.newaxis]

    return np.array([np.random.choice(L_val, p=p) for p in probs])


def Post(X):
    """
    Compute log posterior probability for MCMC state.
    
    Components:
    - p1: Alpha prior (Gamma)
    - p2: Beta prior (Dirichlet / Beta)
    - p3: Temporal Dirichlet process
    - p4: Temporal cluster coupling
    - p5: Spatial cluster parameters prior (NIW)
    - p6: Cluster weight likelihood
    - p7: Intensity gamma prior
    - p8: Spatial point likelihood
    
    Parameters
    ----------
    X : list
        [Alpha, Beta, gammaParam, Z, Phi]
    
    Returns
    -------
    float
        Log posterior probability
    """
    Alpha, Beta, gammaParam, Z, Phi = X

    # p1: Alpha prior
    p1 = (alpha0 - 1) * np.log(Alpha[0]) - Alpha[0]

    # p2: Beta prior (first partition)
    p2 = (Alpha[0] / L - 1) * np.sum(
        np.log(Beta[0])
    ) + mp.log(BetaInv(np.repeat(Alpha[0] / L, L)))

    # p3: Temporal DP prior on Alpha
    p3 = mp.fsum(
        (Alpha[:-1] - 1) * np.log(Alpha[1:])
        - Alpha[1:]
        - np.array([mp.log(x) for x in GammaVectorized(Alpha[:-1])])
    )

    # p4: Temporal coupling of Beta
    p4 = 0
    for i in range(pT - 1):
        prod = Alpha[i + 1] * Beta[i]
        if all(prod > 0):
            p4 += mp.fsum((prod - 1) * np.log(Beta[i + 1])) + mp.log(
                BetaInv(prod)
            )
        else:
            prod = Alpha[i + 1] * mp.matrix(Beta[i])
            p4 += mp.fsum((prod - 1) * np.log(Beta[i + 1])) + mp.log(
                BetaInv(prod)
            )

    # p5: Spatial cluster priors (NIW)
    p5 = 0
    for i in range(L):
        p5 += IW.logpdf(Phi[i][1]) + Gaus.logpdf(Phi[i][0])

    # p6: Cluster assignment likelihood (weights)
    p6 = 0
    for i in range(pT):
        p6 += np.sum(np.log(Beta[i][Z[SimPoisson[:, -1] == i]]))

    # p7: Gamma intensity prior
    p7 = np.sum(GammaPrior.logpdf(gammaParam))

    # p8: Spatial point likelihood and intensity
    p8 = 0
    for i in range(pT):
        mask = SimPoisson[:, -1] == i
        p8 += np.sum(mask) * np.log(gammaParam[i]) - gammaParam[i] * Tmax / pT

        contador = 0
        for j in Z[mask]:
            p8 += multivariate_normal.logpdf(
                SimPoisson[mask][contador][:2],
                mean=Phi[j][0],
                cov=Phi[j][1],
            )
            contador += 1

    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8


def PostParalel(m):
    """
    Compute log posterior for MCMC iteration m (for parallel evaluation).
    
    Parameters
    ----------
    m : int
        Chain iteration index
    
    Returns
    -------
    float
        Log posterior probability
    """
    if m % (len(Cadena) // 1000) == 0:
        print(f"Progress: {m/len(Cadena):.1%}")

    X = Cadena[m]
    Alpha, Beta, gammaParam, Z, Phi = X

    p1 = (alpha0 - 1) * np.log(Alpha[0]) - Alpha[0]
    p2 = (Alpha[0] / L - 1) * np.sum(
        np.log(Beta[0])
    ) + mp.log(BetaInv(np.repeat(Alpha[0] / L, L)))
    p3 = mp.fsum(
        (Alpha[:-1] - 1) * np.log(Alpha[1:])
        - Alpha[1:]
        - np.array([mp.log(x) for x in GammaVectorized(Alpha[:-1])])
    )

    p4 = 0
    for i in range(pT - 1):
        prod = Alpha[i + 1] * Beta[i]
        if all(prod > 0):
            p4 += mp.fsum((prod - 1) * np.log(Beta[i + 1])) + mp.log(
                BetaInv(prod)
            )
        else:
            prod = Alpha[i + 1] * mp.matrix(Beta[i])
            p4 += mp.fsum((prod - 1) * np.log(Beta[i + 1])) + mp.log(
                BetaInv(prod)
            )

    p5 = 0
    for i in range(L):
        p5 += IW.logpdf(Phi[i][1]) + Gaus.logpdf(Phi[i][0])

    p6 = 0
    for i in range(pT):
        p6 += np.sum(np.log(Beta[i][Z[SimPoisson[:, -1] == i]]))

    p7 = np.sum(GammaPrior.logpdf(gammaParam))

    p8 = 0
    for i in range(pT):
        mask = SimPoisson[:, -1] == i
        p8 += np.sum(mask) * np.log(gammaParam[i]) - gammaParam[i] * Tmax / pT

        contador = 0
        for j in Z[mask]:
            p8 += multivariate_normal.logpdf(
                SimPoisson[mask][contador][:2],
                mean=Phi[j][0],
                cov=Phi[j][1],
            )
            contador += 1

    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8


def CadenaMarginalBeta(i):
    """
    Extract marginal trace of Beta parameter for cluster i.
    
    Parameters
    ----------
    i : int
        Cluster index
    
    Returns
    -------
    ndarray
        Beta trace across chain iterations
    """
    graf = np.zeros(len(Cadena))
    for j in range(len(Cadena)):
        graf[j] = Cadena[j][1][0][i]
    return graf
