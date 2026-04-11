"""
Configuration templates for NHPP analyses.

Template configurations for different study areas and hyperparameter sets.
"""

from scipy.stats import invwishart, multivariate_normal, gamma
import numpy as np

# ============================================================================
# MLB BASEBALL - PITCH HIT LOCATIONS
# ============================================================================

BASEBALL_CONFIG = {
    # Field coordinate domain (feet, origin at center)
    "xlim": -150,
    "Xlim": 150,
    "ylim": -150,
    "Ylim": 150,
    # Data filters
    "exit_velocity_min": 50,  # mph - minimum hard contact threshold
    "launch_angle_min": -45,  # degrees
    "launch_angle_max": 90,  # degrees
    "exclude_region": None,  # Can exclude foul territory if needed
    # Temporal discretization
    "pT": 4,  # Number of time partitions (e.g., quarters of season)
    "ClustTiempo": 3,  # Clusters per time period
    # Priors - Normal-Inverse-Wishart for cluster parameters
    "nu": 3,  # Degrees of freedom for Inverse-Wishart
    "Sprior": np.diag((50, 50)),  # Scale matrix (field coordinates in feet)
    "mu": np.array((0, 100)),  # Prior mean for cluster centers
    "lam": 0.001,  # Precision scaling
    # DP concentration
    "alpha0": 1,
    # Intensity (Poisson rate) prior - Gamma
    "pgamma": 10,  # Hits per day
    "k": 0.1,
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
