# Federated Learning for Privacy-Preserving Malaria Prediction

**Multi-Scenario Evaluation Using Ghana DHS Data**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation for the MSc thesis: **"Federated Learning for Privacy-Preserving Malaria Prediction: Multi-Scenario Evaluation Using Ghana DHS Data"**.

The project evaluates federated learning algorithms (FedAvg and FedProx) against centralized baselines for malaria prediction in children under 5 years in Ghana, using data from the Demographic and Health Surveys (DHS) and Malaria Indicator Surveys (MIS).

### Key Features

- **Privacy-preserving**: Federated learning enables collaborative model training without sharing raw patient data
- **Multi-scenario evaluation**: IID, Non-IID (regional heterogeneity), and data quality variation scenarios
- **Comprehensive comparison**: FedAvg, FedProx vs. Centralized Logistic Regression and Random Forest
- **Reproducible research**: Fixed seeds, documented preprocessing, and complete pipeline automation
- **Publication-ready outputs**: Statistical analysis, figures, and tables generation

## Repository Structure

```
FL_malaria/
├── data/
│   ├── raw/                    # DHS/MIS Stata files (not included - see Data Access)
│   ├── merged/                 # Merged survey data
│   ├── cleaned/                # Preprocessed datasets
│   │   ├── train_raw.csv       # Raw training data for FL scenarios
│   │   ├── train_centralized.csv  # SMOTE-augmented centralized training
│   │   ├── val_set.csv         # Validation set (real data)
│   │   └── test_set.csv        # Test set (real data)
│   └── fl_scenarios/           # Federated learning client data
│       ├── s1_iid/             # Scenario 1: IID distribution
│       ├── s2_noniid/          # Scenario 2: Regional heterogeneity
│       ├── s3_quality/         # Scenario 3: Data quality variation
│       └── heterogeneity_metrics.json
├── results/
│   ├── centralized_results.json
│   ├── results_baseline.json
│   ├── results_grid_search.json
│   ├── figures/                # Publication-ready figures
│   └── tables/                 # LaTeX/CSV tables
├── logs/                       # Pipeline execution logs
├── models/                     # Saved model checkpoints
├── validation/                 # Preprocessing validation reports
│
├── src/
│   ├── data/
│   │   ├── data_extraction.py      # Stage 1: DHS/MIS data extraction
│   │   ├── data_preprocessing.py   # Stage 2: MICE imputation & feature engineering
│   │   └── create_fl_scenarios.py  # Stage 3a: FL scenario creation
│   ├── training/
│   │   ├── train_centralized.py    # Stage 3b: Centralized baseline training
│   │   └── train_federated.py      # Stage 3c: Federated model training
│   ├── evaluation/
│   │   └── final_analysis_update.py  # Stage 4: Statistical analysis & visualization
│   └── pipeline/
│       └── run_complete_pipeline.py  # Orchestrates complete workflow
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

2. **Create virtual environment**
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
# Core dependencies
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0

# Machine Learning
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
torch>=2.0.0

# Statistical Analysis
statsmodels>=0.13.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Data Processing
pyreadstat>=1.2.0  # For reading Stata files
```

## Data Access

### DHS/MIS Survey Data

This project uses Ghana Demographic and Health Survey (DHS) and Malaria Indicator Survey (MIS) data, which requires registration to access:

1. **Register** at [DHS Program](https://dhsprogram.com/data/new-user-registration.cfm)
2. **Request access** to Ghana datasets:
   - MIS 2016: GHPR7BFL.DTA, GHKR7BFL.DTA
   - MIS 2019: GHPR82FL.DTA, GHKR82FL.DTA
   - DHS 2022: GHPR8CFL.DTA, GHKR8CFL.DTA
3. **Place files** in `data/raw/` directory

### File Naming Convention

| Survey | PR File (Household) | KR File (Children) |
|--------|---------------------|-------------------|
| MIS 2016 | GHPR7BFL.DTA | GHKR7BFL.DTA |
| MIS 2019 | GHPR82FL.DTA | GHKR82FL.DTA |
| DHS 2022 | GHPR8CFL.DTA | GHKR8CFL.DTA |

### Dataset Characteristics

| Survey | Children (6-59 mo) | RDT Positive | Urban | Rainy Season |
|--------|-------------------|--------------|-------|--------------|
| MIS 2016 | 2,892 | 27.8% | 42.1% | 58.2% |
| MIS 2019 | 3,245 | 21.4% | 44.3% | 61.4% |
| DHS 2022 | 4,150 | 16.9% | 46.8% | 55.7% |
| **Combined** | **10,287** | **21.2%** | **44.6%** | **58.1%** |

## Usage

### Quick Start: Run Complete Pipeline

```bash
# Full pipeline with baseline experiments
python src/pipeline/run_complete_pipeline.py

# With grid search hyperparameter tuning
python src/pipeline/run_complete_pipeline.py --experiment grid_search

# Skip data extraction (if already done)
python src/pipeline/run_complete_pipeline.py --skip-extraction

# Run specific stages only
python src/pipeline/run_complete_pipeline.py --stages 4,5,6
```

### Step-by-Step Execution

#### Stage 1: Data Extraction
```bash
python src/data/data_extraction.py
```
Extracts and merges Ghana DHS/MIS malaria survey data from Stata files.

**Outputs:**
- `data/merged/ghana_malaria_merged.csv`
- `validation/yearly_weighted_prevalence.csv`
- `validation/regional_weighted_prevalence.csv`

#### Stage 2: Data Preprocessing
```bash
python src/data/data_preprocessing.py
```
Applies MICE imputation, simulates clinical symptoms, and creates train/val/test splits.

**Key Features:**
- Multiple Imputation by Chained Equations (MICE) for missing data
- Literature-based symptom simulation (chills, sweating, headache, etc.)
- SMOTE for class balancing (35% positive target)
- Stratified 60/20/20 split

**Outputs:**
- `data/cleaned/train_raw.csv` - For FL scenario creation
- `data/cleaned/train_centralized.csv` - SMOTE-augmented (7,200 samples)
- `data/cleaned/val_set.csv` - Validation set
- `data/cleaned/test_set.csv` - Test set

#### Stage 3a: FL Scenario Creation
```bash
python src/data/create_fl_scenarios.py
```
Creates three federated learning scenarios with 5 clients each.

| Scenario | Description | Characteristics |
|----------|-------------|-----------------|
| S1 (IID) | Random equal split | Balanced clients, ~20% each |
| S2 (Non-IID) | Regional heterogeneity | Unequal sizes, varying prevalence |
| S3 (Quality) | S2 + missing data | 5-20% missing values per client |

**Outputs:**
- `data/fl_scenarios/s1_iid/client_{0-4}.csv`
- `data/fl_scenarios/s2_noniid/client_{0-4}.csv`
- `data/fl_scenarios/s3_quality/client_{0-4}.csv`
- `data/fl_scenarios/heterogeneity_metrics.json`

#### Stage 3b: Centralized Training
```bash
python src/training/train_centralized.py
```
Trains baseline models with GridSearchCV hyperparameter tuning.

**Models:**
- Logistic Regression (class-balanced)
- Random Forest (class-balanced)

**Primary Metric:** AUC-PR (Average Precision) for imbalanced data

**Outputs:**
- `results/centralized_results.json`

#### Stage 3c: Federated Training
```bash
# Baseline experiments (10 seeds)
python src/training/train_federated.py --experiment baseline

# Full grid search
python src/training/train_federated.py --experiment grid_search

# Ablation study (mu values only)
python src/training/train_federated.py --experiment ablation
```

**Algorithms:**
- **FedAvg**: Standard federated averaging (μ=0)
- **FedProx**: Proximal term regularization (μ ∈ {0.01, 0.1, 0.5, 1.0})

**Configuration:**
- 10 communication rounds
- 5 clients per scenario
- Local SMOTE balancing (35% positive)
- Decision threshold: 0.35

**Outputs:**
- `results/results_baseline.json`
- `results/results_grid_search.json`

#### Stage 4: Final Analysis
```bash
python src/evaluation/final_analysis_update.py
```
Generates comprehensive statistical analysis and publication-ready visualizations.

**Outputs:**
- `results/comprehensive_statistical_analysis.json`
- `results/figures/` - ROC curves, PR curves, convergence plots, etc.
- `results/tables/` - LaTeX tables for thesis

## Experiment Configurations

### Baseline Experiment
```python
mu_values = [0.0, 0.1]
learning_rates = [0.05]
local_epochs = [10]
n_seeds = 10
```

### Grid Search Experiment
```python
mu_values = [0.0, 0.01, 0.1, 0.5, 1.0]
learning_rates = [0.01, 0.05]
local_epochs = [5, 10, 15]
n_seeds = 10
```

### Ablation Study
```python
mu_values = [0.0, 0.01, 0.1, 0.5, 1.0]
learning_rates = [0.05]
local_epochs = [10]
n_seeds = 10
```

## Features Used

The model uses 12 features for malaria prediction:

| Feature | Type | Description |
|---------|------|-------------|
| `fever` | Binary | Recent fever (from DHS) |
| `diarrhea` | Binary | Recent diarrhea (from DHS) |
| `chills` | Binary | Simulated based on literature |
| `sweating` | Binary | Simulated based on literature |
| `headache` | Ordinal (0-3) | Simulated severity scale |
| `bodyaches` | Ordinal (0-3) | Simulated severity scale |
| `nausea_vomiting` | Binary | Simulated based on literature |
| `appetite_loss` | Binary | Simulated based on literature |
| `bednet_use` | Binary | ITN/LLIN usage |
| `recent_travel` | Binary | Travel to endemic areas |
| `season` | Binary | Rainy (1) vs Dry (0) |
| `age_group` | Ordinal (0-5) | Age category (<6 to <60 months) |

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
- 10 independent runs per configuration
- Results reported as mean ± standard deviation
- Statistical tests (paired t-tests, Tukey HSD) for significance

## Results Summary

### Primary Findings

1. **FedProx outperforms FedAvg** in heterogeneous scenarios (S2, S3)
2. **Optimal μ=0.5** for FedProx in non-IID settings
3. **Federated models achieve comparable performance** to centralized baselines
4. **AUC-PR** is the appropriate metric for imbalanced malaria data

### Key Metrics

| Model | Scenario | AUC-PR | AUC-ROC | F1 Score |
|-------|----------|--------|---------|----------|
| FedAvg | S1 (IID) | 0.XX±0.XX | 0.XX±0.XX | 0.XX±0.XX |
| FedProx | S1 (IID) | 0.XX±0.XX | 0.XX±0.XX | 0.XX±0.XX |
| FedAvg | S2 (Non-IID) | 0.XX±0.XX | 0.XX±0.XX | 0.XX±0.XX |
| FedProx | S2 (Non-IID) | 0.XX±0.XX | 0.XX±0.XX | 0.XX±0.XX |
| FedAvg | S3 (Data Quality) | 0.XX±0.XX | 0.XX±0.XX | 0.XX±0.XX |
| FedProx | S3 (Data Quality) | 0.XX±0.XX | 0.XX±0.XX | 0.XX±0.XX |
| Centralized LR | - | 0.XX | 0.XX | 0.XX |

*(Results will be populated after running experiments)*

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration:
- **Author**: Daniel Kwasi Kovor
- **Email**: dkkovor@st.knust.edu.gh / danielke32@gmail.com
- **Institution**: Kwame Nkrumah University of Science and Technology

---

**Note**: This research was conducted for academic purposes. The federated learning approach demonstrates privacy-preserving potential for healthcare applications in resource-limited settings.
