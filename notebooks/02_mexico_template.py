"""
Template reference script for Mexico earthquake MCMC analysis.

This shows the workflow for running an MCMC chain on real earthquake data.
Based on original Mexico.py - use as a reference for implementation.
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import shapefile as shp

from src.config_templates import MEXICO_CONFIG, build_priors
from src.analysis import (
    load_and_filter_earthquakes,
    earthquakes_to_spatiotemporal,
    assign_time_partition,
)
from src.simulation import initialize_chain_state
from src.core.config import Variables, Variables2

# ============================================================================
# STEP 1: LOAD AND FILTER EARTHQUAKE DATA
# ============================================================================
config = MEXICO_CONFIG

print("Loading earthquake catalog...")
earthquakes = load_and_filter_earthquakes(
    "data/raw/Earthquakes.csv",
    config["xlim"],
    config["Xlim"],
    config["ylim"],
    config["Ylim"],
    depth_max=config["depth_max"],
    magnitude_min=config["magnitude_min"],
    year_min=config["year_min"],
    exclude_region=config["exclude_region"],
)

print(f"Loaded {len(earthquakes)} earthquakes in domain")

# ============================================================================
# STEP 2: CONVERT TO SPATIO-TEMPORAL FORMAT
# ============================================================================
min_date = datetime(config["year_min"], 1, 1)
max_date = datetime(2015, 12, 31)

SimPoisson, Tmax = earthquakes_to_spatiotemporal(
    earthquakes, min_date, max_date
)

print(f"Time span: {Tmax:.1f} days ({min_date} to {max_date})")

# ============================================================================
# STEP 3: ASSIGN TEMPORAL PARTITIONS
# ============================================================================
pT = config["pT"]
ClustTiempo = config["ClustTiempo"]
L = pT * ClustTiempo

SimPoisson = assign_time_partition(SimPoisson, pT, Tmax)

print(f"Model: pT={pT} time partitions, L={L} spatial clusters")
print(f"Events per partition: {np.bincount(SimPoisson[:, 3].astype(int))}")

# ============================================================================
# STEP 4: BUILD PRIORS AND INITIALIZE CHAIN
# ============================================================================
priors = build_priors(config)
IW = priors["IW"]
Gaus = priors["Gaus"]
GammaPrior = priors["GammaPrior"]

init_state = initialize_chain_state(
    SimPoisson,
    pT,
    L,
    config["alpha0"],
    config["xlim"],
    config["Xlim"],
    config["ylim"],
    config["Ylim"],
    IW,
    Gaus,
    GammaPrior,
)

Cadena = [init_state]

# ============================================================================
# STEP 5: SETUP GLOBAL PARAMETERS
# ============================================================================
Variables(
    (
        Tmax,
        pT,
        config["xlim"],
        config["Xlim"],
        config["ylim"],
        config["Ylim"],
        SimPoisson,
        L,
        config["alpha0"],
        IW,
        Gaus,
        GammaPrior,
    )
)
Variables2(Cadena)

# ============================================================================
# STEP 6: VISUALIZATION (OPTIONAL)
# ============================================================================
# Plot earthquakes by time partition
CutTime = np.linspace(0, Tmax, pT + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(pT):
    ax = axes[i]
    mask = SimPoisson[:, 3] == i
    ax.scatter(SimPoisson[mask, 0], SimPoisson[mask, 1], alpha=0.6)
    ax.set_xlim(config["xlim"], config["Xlim"])
    ax.set_ylim(config["ylim"], config["Ylim"])

    start_date = min_date + np.timedelta64(int(CutTime[i]), "D")
    end_date = min_date + np.timedelta64(int(CutTime[i + 1]), "D")
    ax.set_title(f"Period {i}: {start_date.year} - {end_date.year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("results/figures/earthquakes_by_period.png", dpi=100)
plt.close()

print("Visualization saved to results/figures/earthquakes_by_period.png")

# ============================================================================
# STEP 7: RUN MCMC (PLACEHOLDER)
# ============================================================================
# See original Mexico.py for full MCMC loop with all parameter updates

ChainSize = 100  # Use 10_000_000 for real runs

print(f"Ready for MCMC ({ChainSize} iterations planned)")
print("Next steps: Implement full MCMC loop in Simulado.py and save chain")

# ============================================================================
# STEP 8: SAVE PROCESSED DATA
# ============================================================================
np.savetxt("data/processed/earthquakes_mexico.csv", SimPoisson, delimiter=",")
print("Processed data saved to data/processed/earthquakes_mexico.csv")
