"""
Template reference script for post-MCMC analysis.

Computes posterior similarity matrix, optimizes clustering, and produces summaries.
Based on original Clustering.R - shows Python workflow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.postprocessing import (
    posterior_similarity_matrix,
    binder_loss,
    vi_distance,
    save_cluster_summary,
)

# ============================================================================
# STEP 1: LOAD MCMC CHAIN
# ============================================================================
print("Loading MCMC chain...")
chain = np.load("results/chains/chain_synthetic.npy", allow_pickle=True)
print(f"Chain length: {len(chain)}")

# Extract parts
Alpha_chain = np.array([state[0] for state in chain])
Beta_chain = np.array([state[1] for state in chain])
gammaParam_chain = np.array([state[2] for state in chain])
Z_chain = np.array([state[3] for state in chain])
Phi_chain = np.array([state[4] for state in chain])

# ============================================================================
# STEP 2: DISCARD BURNIN AND THIN
# ============================================================================
burnin = len(chain) // 4  # First 25%
thin = 10

Z_post = Z_chain[burnin::thin]
print(f"Posterior samples: {len(Z_post)} (after burnin={burnin}, thin={thin})")

# ============================================================================
# STEP 3: COMPUTE POSTERIOR SIMILARITY MATRIX
# ============================================================================
print("Computing posterior similarity matrix...")
psm = posterior_similarity_matrix(Z_post)

# Visualize PSM
plt.figure(figsize=(8, 8))
plt.imshow(psm, cmap="RdYlBu_r", vmin=0, vmax=1)
plt.colorbar(label="P(same cluster)")
plt.xlabel("Event index")
plt.ylabel("Event index")
plt.title("Posterior Similarity Matrix")
plt.tight_layout()
plt.savefig("results/figures/psm.png", dpi=100)
plt.close()

print("PSM visualization saved to results/figures/psm.png")

# ============================================================================
# STEP 4: FIND OPTIMAL PARTITION - BINDER LOSS
# ============================================================================
print("Optimizing partition (Binder loss)...")

best_partition_binder = None
best_loss = float("inf")

for state in chain[burnin::thin]:
    partition = state[3]
    loss = binder_loss(partition, psm)
    if loss < best_loss:
        best_loss = loss
        best_partition_binder = partition

print(f"Best Binder loss: {best_loss:.2f}")
print(f"Components: {len(np.unique(best_partition_binder))}")

# ============================================================================
# STEP 5: FIND OPTIMAL PARTITION - VI DISTANCE (ALTERNATIVE)
# ============================================================================
print("Computing VI-based clustering...")

# Use first posterior sample as reference
reference = Z_post[0]

best_partition_vi = None
best_vi = float("inf")

for sample in Z_post:
    vi = vi_distance(reference, sample)
    if vi < best_vi:
        best_vi = vi
        best_partition_vi = sample

print(f"Best VI distance: {best_vi:.2f}")

# ============================================================================
# STEP 6: LOAD ORIGINAL DATA FOR VISUALIZATION
# ============================================================================
SimPoisson = np.load("data/synthetic/Muestra.npy")

# Add partition column if needed
if SimPoisson.shape[1] == 3:
    SimPoisson = np.hstack((SimPoisson, np.zeros((len(SimPoisson), 1))))

# ============================================================================
# STEP 7: SAVE CLUSTER ASSIGNMENTS
# ============================================================================
save_cluster_summary(
    best_partition_binder,
    SimPoisson,
    "results/summaries/clusters_binder.csv",
)
save_cluster_summary(
    best_partition_vi,
    SimPoisson,
    "results/summaries/clusters_vi.csv",
)

print("Cluster assignments saved:")
print("  - results/summaries/clusters_binder.csv")
print("  - results/summaries/clusters_vi.csv")

# ============================================================================
# STEP 8: VISUALIZE CLUSTERING
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binder result
ax = axes[0]
scatter = ax.scatter(
    SimPoisson[:, 0],
    SimPoisson[:, 1],
    c=best_partition_binder,
    cmap="tab20",
    s=30,
    alpha=0.7,
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Binder Clustering ({len(np.unique(best_partition_binder))} clusters)")
plt.colorbar(scatter, ax=ax)

# VI result
ax = axes[1]
scatter = ax.scatter(
    SimPoisson[:, 0],
    SimPoisson[:, 1],
    c=best_partition_vi,
    cmap="tab20",
    s=30,
    alpha=0.7,
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"VI Clustering ({len(np.unique(best_partition_vi))} clusters)")
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig("results/figures/clustering_comparison.png", dpi=100)
plt.close()

print("Clustering visualization saved to results/figures/clustering_comparison.png")

# ============================================================================
# STEP 9: BASIC POSTERIOR SUMMARY
# ============================================================================
print("\n" + "=" * 50)
print("POSTERIOR SUMMARY")
print("=" * 50)
print(f"Mean number of clusters (Binder): {len(np.unique(best_partition_binder))}")
print(f"Mean number of clusters (VI): {len(np.unique(best_partition_vi))}")

# Posterior mean of Alpha
Alpha_post_mean = np.mean(Alpha_chain[burnin:], axis=0)
print(f"Posterior mean Alpha: {Alpha_post_mean}")

# Posterior mean of gamma
gamma_post_mean = np.mean(gammaParam_chain[burnin:], axis=0)
print(f"Posterior mean gamma: {gamma_post_mean}")

print("=" * 50)
