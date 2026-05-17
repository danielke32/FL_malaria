# Federated Learning for Privacy-Preserving Malaria Prediction

**Multi-Scenario Evaluation Using Ghana DHS Data**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation for the MSc thesis: **"Federated Learning for Privacy-Preserving Malaria Prediction: Multi-Scenario Evaluation Using Ghana DHS Data"**.

The project evaluates federated learning algorithms (FedAvg and FedProx) against centralized baselines for malaria prediction in children under 5 years in Ghana, using data from the Demographic and Health Surveys (DHS) and Malaria Indicator Surveys (MIS).

### Key Features

- **Privacy-preserving**: Federated learning enables collaborative model training without sharing raw patient data
- **Multi-scenario evaluation**: IID, Non-IID (regional heterogeneity), and data quality variation scenarios
- **Comprehensive comparison**: FedAvg, FedProx vs. centralized Logistic Regression and Random Forest
- **Reproducible research**: Fixed seeds, documented preprocessing, and complete pipeline automation
- **Publication-ready outputs**: Statistical analysis, figures, and tables generation

## Repository Structure

```
FL_malaria/
├── data/
│   ├── raw/                    # DHS/MIS Stata files (not included — see Data Access)
│   ├── merged/                 # Merged survey data
│   ├── cleaned/                # Preprocessed datasets
│   │   ├── train_raw.csv          # Raw training data for FL scenarios
│   │   ├── train_centralized.csv  # SMOTE-augmented centralized training
│   │   ├── val_set.csv            # Validation set (real data)
│   │   └── test_set.csv           # Test set (real data)
│   └── fl_scenarios/           # Federated learning client data
│       ├── s1_iid/             # Scenario 1: IID distribution
│       ├── s2_noniid/          # Scenario 2: Regional heterogeneity
│       ├── s3_quality/         # Scenario 3: Data quality variation
│       └── heterogeneity_metrics.json
├── results/
│   ├── centralized_results.json
│   ├── results_baseline.json
│   ├── results_grid_search.json
│   ├── comprehensive_statistical_analysis.json
│   ├── figures/                # Publication-ready figures (PNG + greyscale)
│   └── tables/                 # LaTeX and CSV tables
├── logs/                       # Pipeline execution logs
├── models/                     # Saved model checkpoints
├── validation/                 # Preprocessing validation reports
│
├── src/
│   ├── data/
│   │   ├── data_extraction.py      # Stage 1: DHS/MIS data extraction
│   │   ├── data_preprocessing.py   # Stage 2: MICE imputation & feature engineering
│   │   └── create_fl_scenarios.py  # Stage 3: FL scenario creation
│   ├── training/
│   │   ├── train_centralized.py    # Stage 4: Centralized baseline training
│   │   └── train_federated.py      # Stage 5: Federated model training
│   ├── evaluation/
│   │   ├── final_analysis.py       # Stage 6: Statistical analysis & visualization
│   │   └── sensitivity_analysis.py # Robustness check: native vs. full feature set
│   └── pipeline/
│       └── run_complete_pipeline.py  # Orchestrates the complete workflow
│
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/danielke32/FL_malaria.git
   cd FL_malaria
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv federated_ml

   # Linux/macOS
   source federated_ml/bin/activate

   # Windows
   federated_ml\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
torch>=2.0.0
statsmodels>=0.13.0
matplotlib>=3.6.0
seaborn>=0.12.0
Pillow>=9.0.0
```

## Data Access

### DHS/MIS Survey Data

This project uses Ghana Demographic and Health Survey (DHS) and Malaria Indicator Survey (MIS) data, which requires registration to access:

1. **Register** at [DHS Program](https://dhsprogram.com/data/new-user-registration.cfm)
2. **Request access** to the Ghana datasets listed below
3. **Place files** in `data/raw/`

### File Naming Convention

| Survey | PR file (household) | KR file (children) |
|--------|---------------------|-------------------|
| MIS 2016 | GHPR7BFL.DTA | GHKR7BFL.DTA |
| MIS 2019 | GHPR82FL.DTA | GHKR82FL.DTA |
| DHS 2022 | GHPR8CFL.DTA | GHKR8CFL.DTA |

### Dataset Characteristics

| Survey | Children (6–59 mo) | RDT positive | Urban | Rainy season |
|--------|-------------------|--------------|-------|--------------|
| MIS 2016 | 2,892 | 27.8% | 42.1% | 58.2% |
| MIS 2019 | 3,245 | 21.4% | 44.3% | 61.4% |
| DHS 2022 | 4,150 | 16.9% | 46.8% | 55.7% |
| **Combined** | **10,287** | **21.2%** | **44.6%** | **58.1%** |

## Usage

### Quick Start: Run Complete Pipeline

```bash
# Full pipeline — baseline experiment (default)
python src/pipeline/run_complete_pipeline.py

# With grid search hyperparameter tuning
python src/pipeline/run_complete_pipeline.py --experiment grid_search

# Ablation study (μ values only)
python src/pipeline/run_complete_pipeline.py --experiment ablation

# Skip Stage 1 if data is already extracted
python src/pipeline/run_complete_pipeline.py --skip-extraction

# Run specific stages (e.g. training + analysis only)
python src/pipeline/run_complete_pipeline.py --stages 4,5,6
```

### Step-by-Step Execution

#### Stage 1: Data Extraction
```bash
python src/data/data_extraction.py
```
Extracts and merges Ghana DHS/MIS malaria survey data from Stata files using `pandas.read_stata`.

**Outputs:**
- `data/merged/ghana_malaria_merged.csv`
- `validation/yearly_weighted_prevalence.csv`
- `validation/regional_weighted_prevalence.csv`

#### Stage 2: Data Preprocessing
```bash
python src/data/data_preprocessing.py
```
Applies MICE imputation, simulates clinical symptoms from the literature, and creates stratified train/val/test splits.

**Key steps:**
- Multiple Imputation by Chained Equations (MICE) with 5 imputations, averaged
- Literature-based symptom simulation (chills, sweating, headache, etc.)
- Stratified 60/20/20 split (train / val / test)
- SMOTE applied to the centralized training set only: 7,200 total samples at 35% positive class target

**Outputs:**
- `data/cleaned/train_raw.csv` — raw training split for FL scenario creation
- `data/cleaned/train_centralized.csv` — SMOTE-augmented training set (7,200 samples)
- `data/cleaned/val_set.csv` — validation set (real data, no oversampling)
- `data/cleaned/test_set.csv` — test set (real data, no oversampling)

#### Stage 3: FL Scenario Creation
```bash
python src/data/create_fl_scenarios.py
```
Creates three federated learning scenarios across 5 clients (mapped to Ghana regions).

| Scenario | Description | Missingness |
|----------|-------------|-------------|
| S1 (IID) | Random equal split across clients | None |
| S2 (Non-IID) | Regional assignment — unequal sizes and varying prevalence | None |
| S3 (Quality) | Same as S2 with injected missing values per client | 5–20% per client |

S3 missing rates by client: Greater Accra 5%, Ashanti 10%, Northern 20%, Upper East 15%, Western 12%.

**Outputs:**
- `data/fl_scenarios/s1_iid/client_{0-4}.csv`
- `data/fl_scenarios/s2_noniid/client_{0-4}.csv`
- `data/fl_scenarios/s3_quality/client_{0-4}.csv`
- `data/fl_scenarios/heterogeneity_metrics.json`

#### Stage 4: Centralized Training
```bash
python src/training/train_centralized.py
```
Trains baseline models using `GridSearchCV` with 5-fold stratified cross-validation.

**Models:**
- Logistic Regression (class-balanced)
- Random Forest (class-balanced)

**Primary metric:** AUC-PR (Average Precision) — appropriate for the class-imbalanced setting.

**Outputs:**
- `results/centralized_results.json`

#### Stage 5: Federated Training
```bash
# Baseline experiment (default)
python src/training/train_federated.py --experiment baseline

# Full grid search
python src/training/train_federated.py --experiment grid_search

# Ablation study (μ values only)
python src/training/train_federated.py --experiment ablation
```

**Algorithms:**
- **FedAvg**: standard federated averaging (μ = 0)
- **FedProx**: proximal-term regularization (μ ∈ {0.01, 0.1, 0.5, 1.0})

**Configuration:**
- 10 communication rounds
- 5 clients per scenario
- Batch size: 32
- Local SMOTE balancing: 35% positive target
- Decision threshold: 0.35

**Outputs:**
- `results/results_baseline.json`
- `results/results_grid_search.json` (grid search only)
- `results/results_ablation.json` (ablation only)

#### Stage 6: Final Analysis
```bash
python src/evaluation/final_analysis.py
```
Generates comprehensive statistical analysis and publication-ready visualizations.

**Figures produced** (PNG + greyscale copy for print):
- `figure_1_performance_comparison`
- `figure_2_roc_pr_curves`
- `figure_3_metric_distributions`
- `figure_4_confusion_matrices`
- `figure_5_convergence_analysis`
- `figure_6_ablation_study`
- `figure_7_statistical_summary`
- `figure_8_radar_comparison`
- `figure_9_error_analysis`

**Tables produced** (CSV + LaTeX):
- `table_1_performance_metrics`
- `table_2_statistical_tests`
- `table_3_centralized_comparison`
- `table_4_ablation_results`
- `table_5_comprehensive_stats`

**Outputs:**
- `results/comprehensive_statistical_analysis.json`
- `results/figures/` — all figures at 300 DPI
- `results/tables/` — CSV and `.tex` files

#### Sensitivity Analysis (optional)
```bash
python src/evaluation/sensitivity_analysis.py
```
Robustness check that re-runs all three FL scenarios and the centralized baseline using only the 6 native DHS features (no synthetically simulated symptoms), then compares AUC-PR and AUC-ROC against the full 12-feature results. A small performance drop confirms that synthetic features do not carry inflated outcome signal.

- **Native features (6):** `fever`, `diarrhea`, `bednet_use`, `recent_travel`, `season`, `age_group`
- **Synthetic features (6):** `chills`, `sweating`, `headache`, `bodyaches`, `nausea_vomiting`, `appetite_loss`

**Output:** `results/sensitivity_analysis.json`

## Experiment Configurations

### Baseline
```python
mu_values      = [0.0, 0.1]
learning_rates = [0.05]
local_epochs   = [10]
n_seeds        = 10
```

### Grid Search
```python
mu_values      = [0.0, 0.01, 0.1, 0.5, 1.0]
learning_rates = [0.01, 0.05]
local_epochs   = [5, 10, 15]
n_seeds        = 10
```

### Ablation Study
```python
mu_values      = [0.0, 0.01, 0.1, 0.5, 1.0]
learning_rates = [0.05]
local_epochs   = [10]
n_seeds        = 10
```

## Features

The model uses 12 features for malaria prediction:

| Feature | Type | Description |
|---------|------|-------------|
| `fever` | Binary | Recent fever (from DHS) |
| `diarrhea` | Binary | Recent diarrhea (from DHS) |
| `chills` | Binary | Simulated based on literature |
| `sweating` | Binary | Simulated based on literature |
| `headache` | Ordinal (0–3) | Simulated severity scale |
| `bodyaches` | Ordinal (0–3) | Simulated severity scale |
| `nausea_vomiting` | Binary | Simulated based on literature |
| `appetite_loss` | Binary | Simulated based on literature |
| `bednet_use` | Binary | ITN/LLIN usage |
| `recent_travel` | Binary | Travel to endemic areas |
| `season` | Binary | Rainy (1) vs. dry (0) |
| `age_group` | Ordinal (0–5) | Age category (<6 to <60 months) |

## Reproducibility

### Fixed Random Seeds
All experiments use seed 42 with comprehensive seeding:

```python
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

### Multiple Seeds for Statistical Validity
- 10 independent runs per configuration (seeds 42–51)
- Results reported as mean ± standard deviation
- Statistical tests: paired t-tests and Tukey HSD for significance

## Results Summary

### Primary Findings

1. **FedProx outperforms FedAvg** in heterogeneous scenarios (S2, S3)
2. **Optimal μ = 0.5** for FedProx in non-IID settings
3. **Federated models achieve comparable performance** to centralized baselines
4. **AUC-PR** is the most appropriate metric for the class-imbalanced malaria prediction task

### Key Metrics

| Model | Scenario | AUC-PR | AUC-ROC | F1 Score |
|-------|----------|--------|---------|----------|
| FedAvg | S1 (IID) | — | — | — |
| FedProx | S1 (IID) | — | — | — |
| FedAvg | S2 (Non-IID) | — | — | — |
| FedProx | S2 (Non-IID) | — | — | — |
| FedAvg | S3 (Data Quality) | — | — | — |
| FedProx | S3 (Data Quality) | — | — | — |
| Centralized LR | — | — | — | — |
| Centralized RF | — | — | — | — |

*(Populate after running experiments)*

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{kovor2025federated,
  title={Federated Learning for Privacy-Preserving Malaria Prediction:
         Multi-Scenario Evaluation Using Ghana DHS Data},
  author={Kovor, Kwasi Daniel},
  year={2025},
  school={Kwame Nkrumah University of Science and Technology},
  type={MSc Thesis}
}
```

## Acknowledgments

- **DHS Program** for providing access to Ghana survey data
- **WHO** and malaria literature for clinical symptom parameters
- Federated learning framework inspired by [Flower](https://flower.dev/)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Daniel Kwasi Kovor
- **Email**: dkkovor@st.knust.edu.gh / danielke32@gmail.com
- **Institution**: Kwame Nkrumah University of Science and Technology

---

**Note**: This research was conducted for academic purposes. The federated learning approach demonstrates privacy-preserving potential for healthcare applications in resource-limited settings.
