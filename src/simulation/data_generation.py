"""
Synthetic data generation for NHPP analysis.

Generates spatio-temporal point patterns from intensity-modulated Poisson processes.
"""

import numpy as np
from scipy.stats import poisson, multivariate_normal


def thinning_sampler(intensity_func, T_max, domain, n_events_expected=100, seed=None):
    """
    Generate spatio-temporal points via thinning algorithm.
    
    Parameters
    ----------
    intensity_func : callable
        Intensity function λ(x, y, t)
    T_max : float
        Maximum time
    domain : tuple
        ((x_min, x_max), (y_min, y_max))
    n_events_expected : int
        Expected number of events (used for bound approximation)
    seed : int, optional
        Random seed
    
    Returns
    -------
    ndarray
        Points as (x, y, t) array
    """
    if seed is not None:
        np.random.seed(seed)

    (x_min, x_max), (y_min, y_max) = domain
    area = (x_max - x_min) * (y_max - y_min)

    # Find intensity bound
    test_points = np.random.uniform(
        0, 1, (1000, 3)
    )
    test_points[:, 0] = x_min + test_points[:, 0] * (x_max - x_min)
    test_points[:, 1] = y_min + test_points[:, 1] * (y_max - y_min)
    test_points[:, 2] = test_points[:, 2] * T_max
    intensity_max = np.max(
        [intensity_func(p) for p in test_points]
    )

    # Generate candidates
    n_candidates = poisson.rvs(intensity_max * area * T_max, size=1)[0]
    candidates = np.vstack(
        (
            np.random.uniform(x_min, x_max, n_candidates),
            np.random.uniform(y_min, y_max, n_candidates),
            np.sort(np.random.uniform(0, T_max, n_candidates)),
        )
    ).T

    # Thin
    acceptance_probs = np.array(
        [intensity_func(c) / intensity_max for c in candidates]
    )
    accepted = np.random.uniform(0, 1, n_candidates) < acceptance_probs
    return candidates[accepted]


def initialize_chain_state(SimPoisson, pT, L, alpha0, xlim, Xlim, ylim, Ylim, IW, Gaus, GammaPrior):
    """
    Initialize MCMC chain state.
    
    Parameters
    ----------
    SimPoisson : ndarray
        Spatio-temporal data with time partition labels
    pT : int
        Number of temporal partitions
    L : int
        Number of spatial clusters
    alpha0 : float
        DP concentration parameter
    xlim, Xlim, ylim, Ylim : float
        Domain bounds
    IW, Gaus, GammaPrior : scipy.stats objects
        Prior distributions
    
    Returns
    -------
    list
        Initial state [Alpha, Beta, gammaParam, Z, Phi]
    """
    Alpha = alpha0 * np.ones(pT)
    Beta = np.ones((pT, L)) / L
    Z = np.random.choice(np.arange(L), size=len(SimPoisson))
    gammaParam = GammaPrior.rvs(pT)

    Phi = []
    for i in range(L):
        Ssim = IW.rvs()
        muSim = np.hstack(
            (
                np.random.uniform(xlim, Xlim, size=1),
                np.random.uniform(ylim, Ylim, size=1),
            )
        )
        Phi.append([muSim, Ssim])

    return [Alpha, Beta, gammaParam, Z, Phi]
