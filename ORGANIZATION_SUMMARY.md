# NHPP Baseball Hit Clustering Project - Organized

## ✅ What Was Done

Your research code has been reorganized into a **professional, modular project structure** ready for research publications and collaboration on baseball hit clustering analysis.

### 1. **Folder Structure Created**
```
NHPP/
├── data/              (raw/, processed/, synthetic/)
├── src/               (core/, simulation/, analysis/, postprocessing/)
├── results/           (chains/, figures/, summaries/)
├── notebooks/         (template scripts)
├── requirements.txt   (dependencies)
└── README.md          (full documentation)
```

### 2. **Code Refactored into Modules**

| Component | Files | Purpose |
|-----------|-------|---------|
| **Core Bayesian** | `src/core/bayesian_model.py` | Posterior, cluster assignment, diagnostics |
| **Config** | `src/core/config.py` | Global parameter management |
| **Data Generation** | `src/simulation/data_generation.py` | Thinning, chain initialization |
| **Hit Data** | `src/analysis/baseball_hit_data.py` | Load, filter, preprocess hit catalogs |
| **Data Extraction** | `src/data_extraction.py` | StatCast fetching, preparation (NEW!) |
| **Post-MCMC** | `src/postprocessing/clustering_analysis.py` | PSM, Binder loss, VI distance |

### 3. **Reference Templates Created**

Three complete workflow templates in `notebooks/`:
- `01_synthetic_template.py` — Synthetic MCMC analysis
- `02_mexico_template.py` — Real earthquake analysis
- `03_postprocessing_template.py` — Post-MCMC clustering

### 4. **Documentation**

- **README.md** — 200+ line comprehensive guide with:
  - Installation & setup
  - Quick start examples
  - API reference for all modules
  - Model details & mathematics
  - Usage examples

- **config_templates.py** — Pre-built configurations:
  - `BASEBALL_CONFIG` — MLB StatCast hit data parameters
  - `SYNTHETIC_CONFIG` — Synthetic data parameters
  - Helper function for building priors

---

## 🔄 Original Code → New Structure Mapping

### **Funciones.py** → `src/core/`
**Original**: 400+ lines of functions  
**New organization**:
- `bayesian_model.py` — All posterior/likelihood functions
  - `Post()` — Log posterior (8 components)
  - `AsignarZ_Todos()` — Vectorized cluster assignment
  - `NormalZ()` — Single-event cluster selection
  - `PostParalel()` — Parallel posterior evaluation
  - `CadenaMarginalBeta()` — Trace extraction

- `config.py` — Global parameter management
  - `Variables()` — Initialize hyperparameters
  - `Variables2()` — Update chain
  - `Variables3()` — Set temporal parameters

✨ **Improvements**: Docstrings, type hints, clearer organization

### **Simulado.py** → `notebooks/01_synthetic_template.py`
**Original**: Standalone synthetic analysis script  
**New approach**:
- Refactored into reusable functions in `src/simulation/`
- Template shows complete workflow with comments
- Can import and extend from modules

**Key functions extracted**:
- `thinning_sampler()` — Generate synthetic points
- `initialize_chain_state()` — Setup chain state

### **Mexico.py** → `notebooks/02_baseball_template.py`
**Original**: Earthquake data loading + analysis  
**New approach**:
- Utility functions in `src/analysis/baseball_hit_data.py`
- New extraction utilities in `src/data_extraction.py`
- Template script shows proper workflow
- Modular, reusable components

**Key functions extracted**:
- `load_and_filter_hits()` — Filter by region/exit velocity/launch angle
- `hits_to_spatiotemporal()` — Convert to point process format
- `assign_time_partition()` — Temporal binning
- `quick_load_season()` — One-liner data loading!

### **Clustering.R** → `notebooks/03_postprocessing_template.py`
**Original**: R-based post-MCMC clustering analysis  
**New approach**:
- Converted to Python in `src/postprocessing/`
- Complete workflow template
- No external R dependency

**Key functions created**:
- `posterior_similarity_matrix()` — Compute PSM from samples
- `binder_loss()` — Binder loss optimization
- `vi_distance()` — Variation of Information metric
- `save_cluster_summary()` — Export results to CSV

---

## 📋 How to Use the New Structure

### **For New Research**

1. **Copy a template** to start:
   ```bash
   cp notebooks/02_mexico_template.py notebooks/my_analysis.py
   ```

2. **Import modules**:
   ```python
   from src.config_templates import BASEBALL_CONFIG, build_priors
   from src.analysis import load_and_filter_hits
   from src.data_extraction import quick_load_season
   from src.postprocessing import posterior_similarity_matrix
   ```

3. **Configure**:
   ```python
   config = BASEBALL_CONFIG.copy()
   config['exit_velocity_min'] = 55  # Custom parameter
   priors = build_priors(config)
   ```

4. **Run analysis** — Everything is imported, ready to extend

---

## 📂 Data Organization

**Where to put files:**

| Data Type | Location |
|-----------|----------|
| Earthquake CSVs | `data/raw/` |
| Shapefiles (Mexico.shp) | `data/raw/` |
| Cleaned/processed data | `data/processed/` |
| Synthetic Muestra.npy | `data/synthetic/` |
| MCMC chains (.npy) | `results/chains/` |
| Plots/figures | `results/figures/` |
| Cluster assignments (.csv) | `results/summaries/` |

---

## 🚀 Next Steps for Your Research

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Also install: pip install pybaseball (for StatCast data)
   ```

2. **Quick test - Load a season**:
   ```python
   from src.data_extraction import quick_load_season
   data = quick_load_season(2023)  # Instant one-liner!
   ```

3. **Start with a template**:
   ```bash
   python notebooks/02_baseball_template.py  # Test baseball workflow
   ```

4. **Analyze results**:
   ```bash
   python notebooks/03_postprocessing_template.py
   ```

---

## 📚 Documentation Structure

- **README.md** — Start here: Overview, installation, quick start
- **config_templates.py** — All configuration options
- **Docstrings** — Every function has detailed docstrings
- **Templates** — Three complete workflow examples

---

## 🔧 Key Features

✅ **Modular** — Import only what you need  
✅ **Documented** — Comprehensive docstrings + README  
✅ **Reproducible** — Config-based setup, fixed random seeds  
✅ **Extensible** — Easy to add new models/datasets  
✅ **Professional** — Publication-ready organization  

---

## 💡 Tips

- **Use config templates** to avoid hardcoding parameters
- **Save chains to `results/chains/`** with descriptive names
- **Keep notebooks for exploration**, Python scripts for production
- **Version-control your configs** — they're key for reproducibility

---

## Original Files

Your original files are still in the workspace:
- `Clustering.R`
- `Funciones.py`
- `Mexico.py`
- `Simulado.py`
- `Muestra.npy`

These can be archived or used as reference alongside the new modular structure.

---

**Date**: April 9, 2026  
**Status**: Ready for research use ✓
