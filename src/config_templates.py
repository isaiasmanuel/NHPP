"""
Configuration templates for NHPP analyses.

Template configurations for different study areas and hyperparameter sets.
"""

from scipy.stats import invwishart, multivariate_normal, gamma
import numpy as np

# ============================================================================
# MEXICO EARTHQUAKES - COCOS SUBDUCTION ZONE
# ============================================================================

MEXICO_CONFIG = {
    # Geographic domain (longitude, latitude)
    "xlim": -105.5,
    "Xlim": -96.5,
    "ylim": 15.0,
    "Ylim": 20.0,
    # Data filters
    "depth_max": 100,
    "magnitude_min": 4.0,
    "year_min": 2000,
    "exclude_region": ((-98, -96.5), (18, 20)),  # Gulf of Mexico
    # Temporal discretization
    "pT": 4,  # Number of time partitions
    "ClustTiempo": 3,  # Clusters per time period
    # Priors - Normal-Inverse-Wishart for cluster parameters
    "nu": 3,  # Degrees of freedom for Inverse-Wishart
    "Sprior": np.diag((1, 1)),  # Scale matrix
    "mu": np.array((-102, 17)),  # Prior mean for cluster centers
    "lam": 0.01,  # Precision scaling
    # DP concentration
    "alpha0": 1,
    # Intensity (Poisson rate) prior - Gamma
    "pgamma": 1,
    "k": 0.01,
}

# ============================================================================
# SYNTHETIC DATA
# ============================================================================

SYNTHETIC_CONFIG = {
    # Domain bounds
    "xlim": -5,
    "Xlim": 10,
    "ylim": -5,
    "Ylim": 10,
    # Temporal discretization
    "Tmax": 10,
    "pT": 8,  # More partitions for synthetic
    "ClustTiempo": 1,
    # Priors
    "nu": 3,
    "Sprior": np.diag((1, 1)),
    "mu": np.array((1, 1)),
    "lam": 0.1,
    "alpha0": 1,
    "pgamma": 70,
    "k": 0.1,
}


def build_priors(config):
    """
    Build prior distributions from config dict.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    
    Returns
    -------
    dict
        Dictionary of prior objects
    """
    IW = invwishart(df=config["nu"], scale=config["Sprior"])
    Ssim = IW.rvs()
    Gaus = multivariate_normal(mean=config["mu"], cov=Ssim / config["lam"])
    GammaPrior = gamma(a=config["pgamma"] * config["k"], scale=1 / config["k"])

    return {
        "IW": IW,
        "Gaus": Gaus,
        "GammaPrior": GammaPrior,
        "Ssim": Ssim,
    }
