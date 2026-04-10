"""
Post-MCMC clustering analysis and diagnostics.

Computes posterior similarity matrices, optimizes cluster assignments, and produces summaries.
"""

import numpy as np
import pandas as pd


def posterior_similarity_matrix(cluster_draws):
    """
    Compute posterior similarity matrix from MCMC draws.
    
    Parameters
    ----------
    cluster_draws : ndarray
        Cluster assignments from chain (iterations x n_events)
    
    Returns
    -------
    ndarray
        Posterior similarity matrix (n_events x n_events)
    """
    n_events = cluster_draws.shape[1]
    psm = np.zeros((n_events, n_events))

    for i in range(n_events):
        for j in range(n_events):
            psm[i, j] = np.mean(cluster_draws[:, i] == cluster_draws[:, j])

    return psm


def binder_loss(partition, psm):
    """
    Compute Binder loss for a partition given PSM.
    
    Parameters
    ----------
    partition : ndarray
        Cluster assignment vector
    psm : ndarray
        Posterior similarity matrix
    
    Returns
    -------
    float
        Binder loss
    """
    n = len(partition)
    loss = 0
    for i in range(n):
        for j in range(i + 1, n):
            if partition[i] == partition[j]:
                loss += 1 - psm[i, j]
            else:
                loss += psm[i, j]
    return loss


def vi_distance(partition1, partition2):
    """
    Compute Variation of Information distance between two partitions.
    
    Parameters
    ----------
    partition1, partition2 : ndarray
        Cluster assignments
    
    Returns
    -------
    float
        VI distance
    """
    n = len(partition1)
    unique1 = np.unique(partition1)
    unique2 = np.unique(partition2)

    # Contingency table
    cont = np.zeros((len(unique1), len(unique2)))
    for i, c1 in enumerate(unique1):
        for j, c2 in enumerate(unique2):
            cont[i, j] = np.sum((partition1 == c1) & (partition2 == c2))

    p = cont / n
    pi = np.sum(p, axis=1)
    pj = np.sum(p, axis=0)

    h1 = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    h2 = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    h12 = -np.sum(p[p > 0] * np.log(p[p > 0]))

    return h1 + h2 - 2 * h12


def save_cluster_summary(cluster_assignments, data, output_path):
    """
    Save cluster analysis summary to file.
    
    Parameters
    ----------
    cluster_assignments : ndarray
        Cluster labels for each event
    data : ndarray or pd.DataFrame
        Original spatio-temporal data
    output_path : str
        Output file path
    """
    summary = pd.DataFrame(
        {
            "longitude": data[:, 0],
            "latitude": data[:, 1],
            "time": data[:, 2],
            "cluster": cluster_assignments,
        }
    )

    summary.to_csv(output_path, index=False)
