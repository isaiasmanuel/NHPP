"""
Hyperparameter and global configuration management for NHPP model.

This module manages global state for Bayesian model parameters.
"""

import numpy as np

# Global model parameters
Tmax = None
pT = None  # Number of temporal partitions
xlim = None
Xlim = None
ylim = None
Ylim = None
SimPoisson = None  # Spatio-temporal point data
L = None  # Number of spatial clusters
alpha0 = None  # DP concentration parameter
IW = None  # Inverse-Wishart prior
Gaus = None  # Gaussian prior for cluster means
GammaPrior = None  # Gamma prior for intensity parameters
Cadena = None  # MCMC chain


def Variables(valor):
    """
    Initialize model hyperparameters.
    
    Parameters
    ----------
    valor : tuple
        (Tmax, pT, xlim, Xlim, ylim, Ylim, SimPoisson, L, alpha0, IW, Gaus, GammaPrior)
    """
    global Tmax, pT, xlim, Xlim, ylim, Ylim, SimPoisson, L, alpha0, IW, Gaus, GammaPrior
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12 = valor
    Tmax = v1
    pT = v2
    xlim, Xlim = v3, v4
    ylim, Ylim = v5, v6
    SimPoisson = v7
    L = v8
    alpha0 = v9
    IW = v10
    Gaus = v11
    GammaPrior = v12


def Variables2(valor):
    """Update MCMC chain."""
    global Cadena
    Cadena = valor


def Variables3(valor):
    """
    Set temporal parameters.
    
    Parameters
    ----------
    valor : tuple
        (mindate, CutTime)
    """
    global mindate, CutTime
    v1, v2 = valor
    mindate = v1
    CutTime = v2
