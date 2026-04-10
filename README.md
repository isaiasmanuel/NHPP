# NHPP Earthquake Spatial-Temporal Clustering Research

A comprehensive research framework for analyzing earthquake clustering using **Hierarchical Bayesian Non-Homogeneous Poisson Process (NHPP)** models with **Dirichlet Process Mixtures**.

## Overview

This project implements state-of-the-art Bayesian methods to:
- Model spatio-temporal earthquake clustering
- Learn cluster structure from data via Dirichlet Process Mixtures
- Handle temporal non-stationarity through time-varying mixture weights
- Perform posterior inference via MCMC sampling

**Supported datasets**: Real earthquake catalogs (Mexico Cocos subduction zone) and synthetic point patterns

---

## Project Structure

```
NHPP_Research/
├── data/
│   ├── raw/              # Original earthquake catalogs, shapefiles
│   ├── processed/        # Cleaned, filtered data
│   └── synthetic/        # Generated synthetic point patterns
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── bayesian_model.py    # Posterior, cluster assignment
│   │   └── config.py             # Global parameter management
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── data_generation.py   # Thinning, chain initialization
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── earthquake_data.py   # Loading, filtering, binning
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── clustering_analysis.py # PSM, Binder loss, VI distance
│   └── config_templates.py       # Configuration presets
├── results/
│   ├── chains/           # MCMC chain outputs (.npy)
│   ├── figures/          # Plots and visualizations
│   └── summaries/        # Cluster assignments, metrics
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Load and Preprocess Earthquake Data

```python
import numpy as np
from datetime import datetime
from src.analysis.earthquake_data import (
    load_and_filter_earthquakes,
    earthquakes_to_spatiotemporal,
    assign_time_partition
)
from src.config_templates import MEXICO_CONFIG, build_priors

# Load earthquake catalog
config = MEXICO_CONFIG
earthquakes = load_and_filter_earthquakes(
    'data/raw/Earthquakes.csv',
    config['xlim'], config['Xlim'],
    config['ylim'], config['Ylim'],
    depth_max=config['depth_max'],
    magnitude_min=config['magnitude_min'],
    year_min=config['year_min'],
    exclude_region=config['exclude_region']
)

# Convert to spatio-temporal format
min_date = datetime(config['year_min'], 1, 1)
max_date = datetime(2015, 12, 31)
SimPoisson, Tmax = earthquakes_to_spatiotemporal(
    earthquakes, min_date, max_date
)

# Assign time partitions
L = config['pT'] * config['ClustTiempo']
SimPoisson = assign_time_partition(SimPoisson, config['pT'], Tmax)

print(f"Events: {len(SimPoisson)}, Time span: {Tmax:.1f} days")
```

### 2. Initialize and Run MCMC

```python
from src.simulation.data_generation import initialize_chain_state
from src.core.config import Variables, Variables2

# Build prior distributions
priors = build_priors(config)

# Initialize MCMC state
init_state = initialize_chain_state(
    SimPoisson, config['pT'], L, config['alpha0'],
    config['xlim'], config['Xlim'],
    config['ylim'], config['Ylim'],
    priors['IW'], priors['Gaus'], priors['GammaPrior']
)

# Configure global parameters
Variables((Tmax, config['pT'], config['xlim'], config['Xlim'],
          config['ylim'], config['Ylim'], SimPoisson, L,
          config['alpha0'], priors['IW'], priors['Gaus'],
          priors['GammaPrior']))

Cadena = [init_state]
Variables2(Cadena)

# Run MCMC updates (see Simulado.py as reference for full loop)
# Save chain: np.save('results/chains/chain_mexico.npy', Cadena)
```

### 3. Post-MCMC Analysis

```python
from src.postprocessing.clustering_analysis import (
    posterior_similarity_matrix,
    binder_loss,
    save_cluster_summary
)

# Load chain
chain = np.load('results/chains/chain_mexico.npy', allow_pickle=True)

# Extract cluster assignments (skip burnin)
burnin = 1000
Z_draws = np.array([state[3] for state in chain[burnin:]])

# Compute posterior similarity matrix
psm = posterior_similarity_matrix(Z_draws)

# Find best partition
n_events = len(SimPoisson)
best_partition = None
best_loss = float('inf')

for iteration_state in chain[burnin:]:
    partition = iteration_state[3]
    loss = binder_loss(partition, psm)
    if loss < best_loss:
        best_loss = loss
        best_partition = partition

# Save results
save_cluster_summary(
    best_partition, SimPoisson,
    'results/summaries/clusters_mexico.csv'
)
```

---

## Core Modules

### `src.core.bayesian_model`
**Key functions for Bayesian inference:**
- `Post(X)` - Log posterior probability (product of 8 terms)
- `AsignarZ_Todos()` - Vectorized cluster assignment
- `NormalZ(i)` - Probabilistic cluster selection for event i
- `PostParalel()` - Parallel posterior evaluation

### `src.core.config`
**Global parameter management:**
- `Variables()` - Initialize hyperparameters
- `Variables2()` - Update MCMC chain
- `Variables3()` - Set temporal parameters

### `src.simulation.data_generation`
**Data generation and initialization:**
- `thinning_sampler()` - Generate points via acceptance/rejection
- `initialize_chain_state()` - Initialize MCMC state

### `src.analysis.earthquake_data`
**Real data handling:**
- `load_and_filter_earthquakes()` - Apply spatial/temporal filters
- `earthquakes_to_spatiotemporal()` - Convert to point process format
- `assign_time_partition()` - Temporal binning

### `src.postprocessing.clustering_analysis`
**Post-MCMC analysis:**
- `posterior_similarity_matrix()` - PSM from cluster draws
- `binder_loss()` - Loss for partition given PSM
- `vi_distance()` - Variation of Information metric
- `save_cluster_summary()` - Export results

---

## Configuration

Use template configurations or create custom ones:

```python
config = {
    'xlim': -105.5,
    'Xlim': -96.5,
    'ylim': 15.0,
    'Ylim': 20.0,
    'pT': 4,           # Time partitions
    'ClustTiempo': 3,  # Clusters per period
    'alpha0': 1,       # DP concentration
    'nu': 3,           # NIW degrees of freedom
    'Sprior': np.diag((1, 1)),
    'mu': np.array((-102, 17)),
    'lam': 0.01,
}

priors = build_priors(config)
```

See `src/config_templates.py` for predefined configurations.

---

## Model Details

### Hierarchical DP-Mixture NHPP

**Likelihood (spatial):**
$$\lambda(s_i, t_i) = \gamma_t \sum_{k=1}^K \beta_{t,k} N(\mu_k, \Sigma_k)$$

**Priors:**
- $\alpha_0 \sim \text{Gamma}(a, b)$ — DP concentration
- $\beta_t | \alpha_t \sim \text{Dirichlet}(\alpha_t / L, \ldots)$ — Mixture weights
- $\mu_k | \Sigma_k \sim N(\mu_0, \Sigma_k / \lambda)$ — Cluster centers (Normal-Inverse-Wishart)
- $\gamma_t \sim \text{Gamma}(p, k)$ — Intensity parameters

**Temporal structure:**
- $\alpha_{t+1} | \alpha_t \sim \text{Gamma}$ — Time-varying DP concentration
- $\beta_{t+1} | \alpha_{t+1}, \beta_t \sim \text{Dirichlet}(\alpha_{t+1} \beta_t)$ — Temporal coupling

---

## File Descriptions

### Original Code Files (now reorganized)

| Original | Reorganized Location | Purpose |
|----------|---------------------|---------|
| Funciones.py | src/core/*.py | Bayesian model functions |
| Simulado.py | src/simulation/, reference | MCMC loop template |
| Mexico.py | src/analysis/, reference | Data loading example |
| Clustering.R | src/postprocessing/, reference | Post-analysis example |

---

## Usage Examples

### Example 1: Synthetic Data MCMC

See `notebooks/synthetic_analysis.ipynb` (template to create)

### Example 2: Real Earthquake Analysis

See `notebooks/mexico_earthquake_analysis.ipynb` (template to create)

---

## References

Key methodologies:
- **Dirichlet Process Mixtures**: Ferguson (1973), Teh et al. (2006)
- **Point Process Models**: Cox & Isham (1980)
- **MCMC Diagnostics**: Gelman & Rubin (1992)
- **Clustering Loss Functions**: Binder (1978), Meila (2007)

---

## Contributing

For new analyses:
1. Create notebook in `notebooks/`
2. Organize code into `src/` modules
3. Save results to `results/`
4. Update this README with new methods

---

## License

[Specify license]

## Contact

Author: [Your name]  
Date: 2026-04-09

---

## Troubleshooting

**Import errors**: Ensure virtual environment is activated and `pip install -r requirements.txt` completed.

**Memory issues**: For large chains, process in batches or subsample iterations.

**Slow PSM computation**: Use sparse matrices or approximate with subset of iterations.

---

*Last updated: April 9, 2026*
