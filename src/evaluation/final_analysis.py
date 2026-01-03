# 🇬🇭 STAGE 4: Evaluation and Statistical Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    roc_auc_score
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

# Publication-quality plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'text.usetex': False
})


# Configuration
class Config:
    """Global configuration."""
    BASE_DIR = Path(__file__).parent if '__file__' in dir() else Path(".")
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = BASE_DIR / "results/figures"
    TABLES_DIR = BASE_DIR / "results/tables"
    
    # Input files
    CENTRALIZED_RESULTS = RESULTS_DIR / "centralized_results.json"
    FEDERATED_BASELINE = RESULTS_DIR / "results_baseline.json"
    FEDERATED_GRID = RESULTS_DIR / "results_grid_search.json"
    
    # Scenario names
    SCENARIO_NAMES = {
        's1_iid': 'S1: IID Baseline',
        's2_noniid': 'S2: Regional Heterogeneity',
        's3_quality': 'S3: Quality Variation'
    }
    
    # Algorithm names
    ALGO_NAMES = {
        'fedavg': 'FedAvg',
        'fedprox': 'FedProx',
        'logistic_regression': 'Centralized LR',
        'random_forest': 'Centralized RF'
    }
    
    # Colors for consistent plotting
    COLORS = {
        'FedAvg': '#1f77b4',
        'FedProx': '#ff7f0e',
        'Centralized LR': '#2ca02c',
        'Centralized RF': '#d62728',
        'fedavg': '#1f77b4',
        'fedprox': '#ff7f0e',
        'logistic_regression': '#2ca02c',
        'random_forest': '#d62728'
    }
    
    # Line styles for curves
    LINE_STYLES = {
        'FedAvg': '-',
        'FedProx': '--',
        'Centralized LR': '-.',
        'Centralized RF': ':'
    }
    
    # Statistical parameters
    N_BOOTSTRAP = 1000
    ALPHA = 0.05
    CONFIDENCE_LEVEL = 0.95
    
    # Ablation study mu values (actual values from grid search)
    MU_VALUES = [0.0, 0.01, 0.1, 0.5, 1.0]
    
    # ==========================================================================
    # OPTIMAL MU VALUES FOR FAIR COMPARISON (from ablation study)
    # ==========================================================================
    # FedAvg is always mu=0.0
    # FedProx: Using FIXED mu=0.5 across all scenarios for consistent comparison
    #   - This allows direct cross-scenario comparison
    #   - S1 (IID): mu=0.5 (note: mu=0.0 is optimal, so FedProx will underperform)
    #   - S2 (Regional Heterogeneity): mu=0.5 (optimal)
    #   - S3 (Quality Variation): mu=0.5 (optimal)
    # ==========================================================================
    OPTIMAL_FEDPROX_MU = {
        's1_iid': 0.5,       # Fixed mu=0.5 for consistent comparison
        's2_noniid': 0.5,    # Optimal from ablation study
        's3_quality': 0.5    # Optimal from ablation study
    }
    
    # Use optimal mu for fair comparison (set to False to use all mu>0 as before)
    USE_OPTIMAL_MU_COMPARISON = True
    
    @classmethod
    def create_directories(cls):
        """Create output directories."""
        cls.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cls.TABLES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created directories: {cls.FIGURES_DIR}, {cls.TABLES_DIR}")


# DATA LOADER 
class DataLoader:
    """Load and organize experimental results with validation."""
    
    def __init__(self):
        """Initialize data loader."""
        self.all_results = {}
        self.results_by_mu = {}
        self.centralized_results = {}
        self.raw_data = {}
        self.validation_report = []
    
    def load_all_data(self) -> Dict:
        """Load all experimental results with validation."""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load centralized results
        self._load_centralized()
        
        # Load federated results
        self._load_federated()
        
        # Organize results
        self._organize_all_results()
        
        # Validate data
        self._validate_data()
        
        # Print summary
        self._print_load_summary()
        
        return self.all_results
    
    def _load_centralized(self):
        """Load centralized baseline results."""
        if Config.CENTRALIZED_RESULTS.exists():
            with open(Config.CENTRALIZED_RESULTS, 'r', encoding='utf-8') as f:
                self.centralized_results = json.load(f)
            print(f"   Loaded centralized results: {Config.CENTRALIZED_RESULTS.name}")
        else:
            print(f"    Centralized results not found: {Config.CENTRALIZED_RESULTS}")
            self.validation_report.append("Missing centralized results file")
    
    def _load_federated(self):
        """Load federated learning results."""
        federated_files = [
            Config.FEDERATED_BASELINE,
            Config.FEDERATED_GRID
        ]
        
        for file_path in federated_files:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.raw_data.update(data)
                print(f"   Loaded federated results: {file_path.name}")
            else:
                print(f"    Federated results not found: {file_path}")
                self.validation_report.append(f"Missing federated file: {file_path.name}")
    
    def _organize_all_results(self):
        """Organize all results into structured format."""
        self._organize_federated_results()
        self._organize_centralized_results()
        self._organize_results_by_mu()
    
    def _organize_federated_results(self):
        """Organize federated learning results.
        
        MODIFIED: Now uses optimal mu values for fair FedAvg vs FedProx comparison.
        - FedAvg: mu = 0.0 (by definition)
        - FedProx: mu = optimal value per scenario (from ablation study)
          - S1 (IID): mu = 0.1 (baseline, since mu=0.0 is optimal)
          - S2 (Regional): mu = 0.5 (optimal)
          - S3 (Quality): mu = 0.5 (optimal)
        
        This ensures balanced sample sizes (60 vs 60) and meaningful comparison.
        """
        for config_key, runs in self.raw_data.items():
            # Parse scenario
            scenario = self._parse_scenario(config_key)
            if not scenario:
                continue
            
            # Parse mu value
            mu_val = self._parse_mu(config_key)
            
            # Determine algorithm based on configuration
            if Config.USE_OPTIMAL_MU_COMPARISON:
                # Fair comparison: FedAvg (mu=0) vs FedProx (optimal mu)
                optimal_mu = Config.OPTIMAL_FEDPROX_MU.get(scenario, 0.1)
                
                if mu_val == 0.0:
                    algo = 'fedavg'
                elif abs(mu_val - optimal_mu) < 0.001:  # Float comparison tolerance
                    algo = 'fedprox'
                else:
                    # Skip non-optimal mu values for main comparison
                    # (they're still used in ablation study via results_by_mu)
                    continue
            else:
                # Original behavior: all mu > 0 grouped as FedProx
                algo = 'fedprox' if mu_val > 0 else 'fedavg'
            
            # Initialize storage
            if scenario not in self.all_results:
                self.all_results[scenario] = {}
            
            if algo not in self.all_results[scenario]:
                self.all_results[scenario][algo] = self._init_algo_storage()
            
            # Extract metrics from each run
            for run_idx, run in enumerate(runs):
                if 'error' in run:
                    continue
                
                self._extract_run_metrics(scenario, algo, run)
    
    def _parse_scenario(self, config_key: str) -> Optional[str]:
        """Parse scenario from config key."""
        if 's1_iid' in config_key:
            return 's1_iid'
        elif 's2_noniid' in config_key:
            return 's2_noniid'
        elif 's3_quality' in config_key:
            return 's3_quality'
        return None
    
    def _parse_mu(self, config_key: str) -> float:
        """Parse mu value from config key."""
        for part in config_key.split('_'):
            if part.startswith('mu'):
                try:
                    return float(part.replace('mu', ''))
                except ValueError:
                    pass
        return 0.0
    
    def _init_algo_storage(self) -> Dict:
        """Initialize storage for algorithm results."""
        return {
            'auc': [],
            'auc_pr': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'history': [],
            'confusion_matrices': [],
            'timing': [],
            'predictions': [],
            'roc_curves': [],
            'pr_curves': []
        }
    
    def _extract_run_metrics(self, scenario: str, algo: str, run: Dict):
        """Extract metrics from a single run."""
        metrics = run.get('test_metrics', {})
        if not metrics:
            return
        
        storage = self.all_results[scenario][algo]
        
        # Core metrics
        storage['auc_pr'].append(metrics.get('auc_pr', metrics.get('pr_auc', 0.0)))
        storage['auc'].append(metrics.get('auc', metrics.get('roc_auc', 0.0)))
        storage['accuracy'].append(metrics.get('accuracy', metrics.get('acc', 0.0)))
        storage['f1'].append(metrics.get('f1', 0.0))
        storage['precision'].append(metrics.get('precision', 0.0))
        storage['recall'].append(metrics.get('recall', metrics.get('sensitivity', 0.0)))
        storage['specificity'].append(metrics.get('specificity', 0.0))
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            storage['confusion_matrices'].append(metrics['confusion_matrix'])
        
        # Timing information
        if 'timing' in run:
            storage['timing'].append(run['timing'])
        
        # Training history
        if 'history' in run:
            storage['history'].append(run['history'])
        
        # Predictions (for statistical tests)
        if 'test_true_labels' in run and 'test_prediction_scores' in run:
            pred_data = {
                'y_true': np.array(run['test_true_labels']),
                'y_pred_proba': np.array(run['test_prediction_scores']),
                'y_pred_class': np.array(run.get('test_prediction_labels', []))
            }
            storage['predictions'].append(pred_data)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(pred_data['y_true'], pred_data['y_pred_proba'])
            storage['roc_curves'].append({'fpr': fpr, 'tpr': tpr})
            
            # Calculate PR curve
            prec, rec, _ = precision_recall_curve(pred_data['y_true'], pred_data['y_pred_proba'])
            storage['pr_curves'].append({'precision': prec, 'recall': rec})
    
    def _organize_centralized_results(self):
        """Organize centralized results across scenarios."""
        for model_key in ['logistic_regression', 'random_forest']:
            if model_key not in self.centralized_results:
                continue
            
            model_data = self.centralized_results[model_key]
            metrics = model_data.get('test_metrics', {})
            
            # Add to all scenarios
            for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
                if scenario not in self.all_results:
                    self.all_results[scenario] = {}
                
                storage = self._init_algo_storage()
                
                # Core metrics (single values wrapped in lists)
                storage['auc_pr'] = [metrics.get('auc_pr', metrics.get('pr_auc', 0.0))]
                storage['auc'] = [metrics.get('auc', metrics.get('roc_auc', 0.0))]
                storage['accuracy'] = [metrics.get('accuracy', 0.0)]
                storage['f1'] = [metrics.get('f1', 0.0)]
                storage['precision'] = [metrics.get('precision', 0.0)]
                storage['recall'] = [metrics.get('recall', 0.0)]
                storage['specificity'] = [metrics.get('specificity', 0.0)]
                
                if 'confusion_matrix' in metrics:
                    storage['confusion_matrices'] = [metrics['confusion_matrix']]
                
                if 'timing' in model_data:
                    storage['timing'] = [model_data['timing']]
                
                # Predictions
                if 'test_true_labels' in model_data and 'test_prediction_scores' in model_data:
                    pred_data = {
                        'y_true': np.array(model_data['test_true_labels']),
                        'y_pred_proba': np.array(model_data['test_prediction_scores']),
                        'y_pred_class': np.array(model_data.get('test_prediction_labels', []))
                    }
                    storage['predictions'] = [pred_data]
                    
                    # ROC curve
                    fpr, tpr, _ = roc_curve(pred_data['y_true'], pred_data['y_pred_proba'])
                    storage['roc_curves'] = [{'fpr': fpr, 'tpr': tpr}]
                    
                    # PR curve
                    prec, rec, _ = precision_recall_curve(pred_data['y_true'], pred_data['y_pred_proba'])
                    storage['pr_curves'] = [{'precision': prec, 'recall': rec}]
                
                self.all_results[scenario][model_key] = storage
    
    def _organize_results_by_mu(self):
        """Organize results by mu value for ablation study."""
        for config_key, runs in self.raw_data.items():
            scenario = self._parse_scenario(config_key)
            if not scenario:
                continue
            
            mu_val = self._parse_mu(config_key)
            mu_key = f'mu_{mu_val}'
            
            if scenario not in self.results_by_mu:
                self.results_by_mu[scenario] = {}
            
            if mu_key not in self.results_by_mu[scenario]:
                self.results_by_mu[scenario][mu_key] = {
                    'mu': mu_val,
                    'auc_pr': [],
                    'auc': [],
                    'accuracy': [],
                    'f1': [],
                    'precision': [],
                    'recall': []
                }
            
            for run in runs:
                if 'error' in run:
                    continue
                
                metrics = run.get('test_metrics', {})
                if not metrics:
                    continue
                
                mu_storage = self.results_by_mu[scenario][mu_key]
                mu_storage['auc_pr'].append(metrics.get('auc_pr', 0.0))
                mu_storage['auc'].append(metrics.get('auc', 0.0))
                mu_storage['accuracy'].append(metrics.get('accuracy', 0.0))
                mu_storage['f1'].append(metrics.get('f1', 0.0))
                mu_storage['precision'].append(metrics.get('precision', 0.0))
                mu_storage['recall'].append(metrics.get('recall', 0.0))
    
    def _validate_data(self):
        """Validate loaded data."""
        print("\n  Validating data...")
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.all_results:
                self.validation_report.append(f"Missing scenario: {scenario}")
                continue
            
            for algo in ['fedavg', 'fedprox']:
                if algo not in self.all_results[scenario]:
                    self.validation_report.append(f"Missing {algo} in {scenario}")
                    continue
                
                n_runs = len(self.all_results[scenario][algo]['auc_pr'])
                if n_runs == 0:
                    self.validation_report.append(f"No runs for {algo} in {scenario}")
                elif n_runs < 3:
                    self.validation_report.append(f"Low sample size ({n_runs}) for {algo} in {scenario}")
        
        if self.validation_report:
            print("    Validation warnings:")
            for warning in self.validation_report:
                print(f"      - {warning}")
        else:
            print("   Data validation passed")
    
    def _print_load_summary(self):
        """Print summary of loaded data."""
        print("\n  Data Summary:")
        
        # Show comparison configuration
        if Config.USE_OPTIMAL_MU_COMPARISON:
            print("  [FAIR COMPARISON MODE: Using optimal mu values]")
            print("    - FedAvg: mu = 0.0")
            for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
                opt_mu = Config.OPTIMAL_FEDPROX_MU.get(scenario, 0.1)
                print(f"    - FedProx ({Config.SCENARIO_NAMES.get(scenario, scenario)}): mu = {opt_mu}")
            print()
        
        for scenario in self.all_results:
            print(f"    {Config.SCENARIO_NAMES.get(scenario, scenario)}:")
            for algo in self.all_results[scenario]:
                n_runs = len(self.all_results[scenario][algo]['auc_pr'])
                print(f"      - {Config.ALGO_NAMES.get(algo, algo)}: {n_runs} runs")
    
    def get_results_by_mu(self) -> Dict:
        """Get results organized by mu value."""
        return self.results_by_mu


# STATISTICAL TESTS 
class StatisticalAnalyzer:
    """
    PRIMARY ANALYSIS:
    - McNemar's Test: Compare paired model accuracies
    - DeLong Test: Compare AUC-ROC between models
    - ANOVA with Tukey HSD: Multi-group performance comparison
    
    SECONDARY ANALYSIS:
    - Cohen's Kappa: Inter-algorithm agreement
    - Wilcoxon Signed-Rank Test: Non-parametric model comparison
    - Bootstrap Confidence Intervals: Robust performance estimates (n=1000)
    
    COMMUNICATION EFFICIENCY:
    - Pearson Correlation: Communication rounds vs. accuracy
    - Mann-Whitney U Test: Federated vs centralized communication costs
    """
    
    def __init__(self, results: Dict, results_by_mu: Dict):
        """Initialize analyzer."""
        self.results = results
        self.results_by_mu = results_by_mu
        self.test_results = {}
    
    def run_all_tests(self) -> Dict:
        """Run all statistical tests."""
        print("\n" + "="*70)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*70)
        
        # PRIMARY ANALYSIS
        print("\n" + "="*70)
        print("PRIMARY ANALYSIS")
        print("="*70)
        
        # Per-scenario primary tests
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            self._primary_analysis_scenario(scenario)
        
        # Cross-scenario ANOVA
        self._anova_across_scenarios()
        
        # SECONDARY ANALYSIS
        print("\n" + "="*70)
        print("SECONDARY ANALYSIS")
        print("="*70)
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            self._secondary_analysis_scenario(scenario)
        
        # COMMUNICATION EFFICIENCY ANALYSIS
        print("\n" + "="*70)
        print("COMMUNICATION EFFICIENCY ANALYSIS")
        print("="*70)
        
        self._communication_efficiency_analysis()
        
        # ABLATION STUDY (ANOVA for mu parameter)
        self._ablation_study_anova()
        
        # SUMMARY TABLE
        self._print_summary_table()
        
        return self.test_results
    
    # PRIMARY ANALYSIS METHODS
    def _primary_analysis_scenario(self, scenario: str):
        """Run primary analysis for a single scenario."""
        if scenario not in self.results:
            return
        
        print(f"\n  {Config.SCENARIO_NAMES.get(scenario, scenario)}:")
        print("  " + "-"*50)
        
        if scenario not in self.test_results:
            self.test_results[scenario] = {}
        
        # Get data
        fedavg_data = self.results[scenario].get('fedavg', {})
        fedprox_data = self.results[scenario].get('fedprox', {})
        lr_data = self.results[scenario].get('logistic_regression', {})
        rf_data = self.results[scenario].get('random_forest', {})
        
        # Descriptive statistics
        self.test_results[scenario]['descriptive'] = self._descriptive_stats_full(scenario)
        
        # 1. McNemar's Test (FedAvg vs FedProx)
        print("\n    1. McNemar's Test (Paired Accuracy Comparison):")
        mcnemar_result = self._mcnemar_test_full(fedavg_data, fedprox_data, "FedAvg", "FedProx")
        self.test_results[scenario]['mcnemar_fedavg_fedprox'] = mcnemar_result
        
        # McNemar: FL vs Centralized
        if lr_data.get('predictions'):
            mcnemar_fl_lr = self._mcnemar_test_full(fedprox_data, lr_data, "FedProx", "Centralized LR")
            self.test_results[scenario]['mcnemar_fedprox_lr'] = mcnemar_fl_lr
        
        if rf_data.get('predictions'):
            mcnemar_fl_rf = self._mcnemar_test_full(fedprox_data, rf_data, "FedProx", "Centralized RF")
            self.test_results[scenario]['mcnemar_fedprox_rf'] = mcnemar_fl_rf
        
        # 2. DeLong Test (AUC-ROC Comparison)
        print("\n    2. DeLong Test (AUC-ROC Comparison):")
        delong_result = self._delong_test_full(fedavg_data, fedprox_data, "FedAvg", "FedProx")
        self.test_results[scenario]['delong_fedavg_fedprox'] = delong_result
        
        # DeLong: FL vs Centralized
        if lr_data.get('predictions'):
            delong_fl_lr = self._delong_test_full(fedprox_data, lr_data, "FedProx", "Centralized LR")
            self.test_results[scenario]['delong_fedprox_lr'] = delong_fl_lr
        
        if rf_data.get('predictions'):
            delong_fl_rf = self._delong_test_full(fedprox_data, rf_data, "FedProx", "Centralized RF")
            self.test_results[scenario]['delong_fedprox_rf'] = delong_fl_rf
        
        # 3. ANOVA with Tukey HSD (Multi-group comparison within scenario)
        print("\n    3. ANOVA with Tukey HSD (Multi-group Comparison):")
        anova_result = self._anova_tukey_within_scenario(scenario)
        self.test_results[scenario]['anova_tukey'] = anova_result
    
    def _descriptive_stats_full(self, scenario: str) -> Dict:
        stats_dict = {}
        
        for algo in ['fedavg', 'fedprox', 'logistic_regression', 'random_forest']:
            if algo not in self.results[scenario]:
                continue
            
            data = self.results[scenario][algo]
            algo_stats = {}
            
            for metric in ['auc_pr', 'auc', 'accuracy', 'f1', 'precision', 'recall']:
                values = data.get(metric, [])
                if values:
                    n = len(values)
                    mean = np.mean(values)
                    std = np.std(values, ddof=1) if n > 1 else 0
                    se = std / np.sqrt(n) if n > 0 else 0
                    
                    # 95% CI
                    if n > 1:
                        ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
                    else:
                        ci = (mean, mean)
                    
                    algo_stats[metric] = {
                        'n': n,
                        'mean': float(mean),
                        'std': float(std),
                        'se': float(se),
                        'ci_lower': float(ci[0]),
                        'ci_upper': float(ci[1]),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
            
            stats_dict[algo] = algo_stats
        
        return stats_dict
    
    def _mcnemar_test_full(self, data1: Dict, data2: Dict, name1: str, name2: str) -> Dict:
        if not data1.get('predictions') or not data2.get('predictions'):
            print(f"       Skipped: Predictions not available")
            return {'error': 'Predictions not available'}
        
        try:
            # Get first run predictions
            pred1 = data1['predictions'][0] if isinstance(data1['predictions'], list) else data1['predictions']
            pred2 = data2['predictions'][0] if isinstance(data2['predictions'], list) else data2['predictions']
            
            y_true = pred1['y_true']
            y_pred1 = pred1['y_pred_class']
            y_pred2 = pred2['y_pred_class']
            
            if len(y_pred1) == 0 or len(y_pred2) == 0:
                return {'error': 'Empty predictions'}
            
            # Build contingency table
            # a: both correct, b: only model1 correct, c: only model2 correct, d: both wrong
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)
            
            a = np.sum(correct1 & correct2)      # Both correct
            b = np.sum(correct1 & ~correct2)     # Only model1 correct
            c = np.sum(~correct1 & correct2)     # Only model2 correct
            d = np.sum(~correct1 & ~correct2)    # Both wrong
            
            n_total = a + b + c + d
            
            # McNemar's test statistic (with continuity correction)
            if (b + c) == 0:
                print(f"       {name1} vs {name2}: No disagreements between classifiers")
                return {
                    'contingency_table': {'a': int(a), 'b': int(b), 'c': int(c), 'd': int(d)},
                    'error': 'No disagreements (b + c = 0)'
                }
            
            # Chi-square statistic with continuity correction
            chi2_stat = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
            
            # Exact McNemar test (binomial) for small samples
            if (b + c) < 25:
                # Use exact binomial test
                p_exact = stats.binom_test(min(b, c), b + c, 0.5) * 2  # Two-tailed
                p_value = min(p_value, p_exact)
            
            # Effect size: Odds ratio
            odds_ratio = b / c if c > 0 else float('inf')
            
            # Significance
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            result = {
                'contingency_table': {
                    'both_correct': int(a),
                    'only_model1_correct': int(b),
                    'only_model2_correct': int(c),
                    'both_wrong': int(d)
                },
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'odds_ratio': float(odds_ratio) if odds_ratio != float('inf') else 'inf',
                'significant': bool(p_value < Config.ALPHA),
                'model1_advantage': int(b - c),
                'interpretation': f"{name1} better" if b > c else f"{name2} better" if c > b else "No difference"
            }
            
            print(f"       {name1} vs {name2}:")
            print(f"         Contingency: b={b} (only {name1}), c={c} (only {name2})")
            print(f"         Chi2 = {chi2_stat:.4f}, p = {p_value:.6f} {sig}")
            print(f"         Interpretation: {result['interpretation']}")
            
            return result
            
        except Exception as e:
            print(f"       Error: {str(e)}")
            return {'error': str(e)}
    
    def _delong_test_full(self, data1: Dict, data2: Dict, name1: str, name2: str) -> Dict:
        """
        DeLong Test for comparing two AUC-ROC values.
        Uses bootstrap variance estimation for robust comparison.
        """
        if not data1.get('predictions') or not data2.get('predictions'):
            print(f"       Skipped: Predictions not available")
            return {'error': 'Predictions not available'}
        
        try:
            pred1 = data1['predictions'][0] if isinstance(data1['predictions'], list) else data1['predictions']
            pred2 = data2['predictions'][0] if isinstance(data2['predictions'], list) else data2['predictions']
            
            y_true = pred1['y_true']
            y_score1 = pred1['y_pred_proba']
            y_score2 = pred2['y_pred_proba']
            
            # Compute AUCs
            auc1 = roc_auc_score(y_true, y_score1)
            auc2 = roc_auc_score(y_true, y_score2)
            diff = auc2 - auc1
            
            # Bootstrap for variance estimation (DeLong-like)
            np.random.seed(42)
            n_bootstrap = 1000
            n_samples = len(y_true)
            
            boot_diffs = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                
                # Ensure both classes present
                if len(np.unique(y_true[indices])) < 2:
                    continue
                
                try:
                    auc1_boot = roc_auc_score(y_true[indices], y_score1[indices])
                    auc2_boot = roc_auc_score(y_true[indices], y_score2[indices])
                    boot_diffs.append(auc2_boot - auc1_boot)
                except:
                    continue
            
            if len(boot_diffs) < 100:
                return {'error': 'Insufficient valid bootstrap samples'}
            
            # Standard error and z-statistic
            se = np.std(boot_diffs, ddof=1)
            z_stat = diff / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed
            
            # 95% CI for difference
            ci_lower = np.percentile(boot_diffs, 2.5)
            ci_upper = np.percentile(boot_diffs, 97.5)
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            result = {
                'auc_model1': float(auc1),
                'auc_model2': float(auc2),
                'difference': float(diff),
                'se': float(se),
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'significant': bool(p_value < Config.ALPHA),
                'n_bootstrap': n_bootstrap,
                'interpretation': f"{name2} better" if diff > 0 and p_value < 0.05 else 
                                 f"{name1} better" if diff < 0 and p_value < 0.05 else "No significant difference"
            }
            
            print(f"       {name1} vs {name2}:")
            print(f"         AUC: {auc1:.4f} vs {auc2:.4f} (diff = {diff:+.4f})")
            print(f"         95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
            print(f"         z = {z_stat:.4f}, p = {p_value:.6f} {sig}")
            
            return result
            
        except Exception as e:
            print(f"       Error: {str(e)}")
            return {'error': str(e)}
    
    def _anova_tukey_within_scenario(self, scenario: str) -> Dict:
        """
        One-way ANOVA with Tukey HSD post-hoc test.
        Compares all algorithms within a scenario.
        """
        groups = []
        labels = []
        all_values = []
        all_labels = []
        
        for algo in ['fedavg', 'fedprox', 'logistic_regression', 'random_forest']:
            if algo not in self.results[scenario]:
                continue
            
            auc_pr = self.results[scenario][algo].get('auc_pr', [])
            if auc_pr and len(auc_pr) > 0:
                groups.append(auc_pr)
                algo_name = Config.ALGO_NAMES.get(algo, algo)
                labels.append(algo_name)
                all_values.extend(auc_pr)
                all_labels.extend([algo_name] * len(auc_pr))
        
        if len(groups) < 2:
            print("       Skipped: Need at least 2 groups for ANOVA")
            return {'error': 'Insufficient groups'}
        
        try:
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Effect size: Eta-squared
            ss_between = sum(len(g) * (np.mean(g) - np.mean(all_values))**2 for g in groups)
            ss_total = sum((v - np.mean(all_values))**2 for v in all_values)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Omega-squared (less biased)
            k = len(groups)
            n_total = len(all_values)
            ms_within = (ss_total - ss_between) / (n_total - k) if (n_total - k) > 0 else 0
            omega_squared = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            print(f"       ANOVA: F({k-1}, {n_total-k}) = {f_stat:.4f}, p = {p_value:.6f} {sig}")
            print(f"       Effect size: eta^2 = {eta_squared:.4f}, omega^2 = {omega_squared:.4f}")
            
            result = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'df_between': k - 1,
                'df_within': n_total - k,
                'eta_squared': float(eta_squared),
                'omega_squared': float(omega_squared),
                'significant': bool(p_value < Config.ALPHA),
                'groups': {label: {'n': len(g), 'mean': np.mean(g), 'std': np.std(g)} 
                          for label, g in zip(labels, groups)}
            }
            
            # Tukey HSD post-hoc if significant
            if p_value < Config.ALPHA and len(all_values) > len(groups):
                try:
                    tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=Config.ALPHA)
                    
                    # Parse Tukey results
                    tukey_results = []
                    for i in range(len(tukey.summary().data) - 1):
                        row = tukey.summary().data[i + 1]
                        tukey_results.append({
                            'group1': str(row[0]),
                            'group2': str(row[1]),
                            'meandiff': float(row[2]),
                            'p_adj': float(row[3]),
                            'lower': float(row[4]),
                            'upper': float(row[5]),
                            'reject': bool(row[6])
                        })
                    
                    result['tukey_hsd'] = tukey_results
                    
                    print(f"\n       Tukey HSD Post-hoc:")
                    for t in tukey_results:
                        sig_t = '*' if t['reject'] else 'ns'
                        print(f"         {t['group1']} vs {t['group2']}: diff = {t['meandiff']:+.4f}, p = {t['p_adj']:.4f} {sig_t}")
                    
                except Exception as e:
                    print(f"       Tukey HSD failed: {e}")
            
            return result
            
        except Exception as e:
            print(f"       Error: {str(e)}")
            return {'error': str(e)}
    
    def _anova_across_scenarios(self):
        """ANOVA comparing FedProx performance across scenarios."""
        print("\n  Cross-Scenario ANOVA (FedProx across scenarios):")
        print("  " + "-"*50)
        
        groups = []
        labels = []
        all_values = []
        all_labels = []
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.results:
                continue
            if 'fedprox' not in self.results[scenario]:
                continue
            
            auc_pr = self.results[scenario]['fedprox'].get('auc_pr', [])
            if auc_pr:
                groups.append(auc_pr)
                scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
                labels.append(scenario_name)
                all_values.extend(auc_pr)
                all_labels.extend([scenario_name] * len(auc_pr))
        
        if len(groups) < 2:
            print("    Skipped: Insufficient scenarios")
            return
        
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Kruskal-Wallis (non-parametric alternative)
            h_stat, p_kw = stats.kruskal(*groups)
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            print(f"    ANOVA: F = {f_stat:.4f}, p = {p_value:.6f} {sig}")
            print(f"    Kruskal-Wallis: H = {h_stat:.4f}, p = {p_kw:.6f}")
            
            self.test_results['cross_scenario_anova'] = {
                'anova': {'f_statistic': float(f_stat), 'p_value': float(p_value)},
                'kruskal_wallis': {'h_statistic': float(h_stat), 'p_value': float(p_kw)},
                'significant': bool(p_value < Config.ALPHA)
            }
            
            # Tukey HSD if significant
            if p_value < Config.ALPHA:
                tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=Config.ALPHA)
                print(f"\n    Tukey HSD:")
                print(tukey)
                
        except Exception as e:
            print(f"    Error: {e}")
    
    # SECONDARY ANALYSIS METHODS
    
    def _secondary_analysis_scenario(self, scenario: str):
        """Run secondary analysis for a single scenario."""
        if scenario not in self.results:
            return
        
        print(f"\n  {Config.SCENARIO_NAMES.get(scenario, scenario)}:")
        print("  " + "-"*50)
        
        fedavg_data = self.results[scenario].get('fedavg', {})
        fedprox_data = self.results[scenario].get('fedprox', {})
        
        # 1. Cohen's Kappa (Inter-algorithm agreement)
        print("\n    1. Cohen's Kappa (Inter-algorithm Agreement):")
        kappa_result = self._cohens_kappa(fedavg_data, fedprox_data, "FedAvg", "FedProx")
        self.test_results[scenario]['cohens_kappa'] = kappa_result
        
        # 2. Wilcoxon Signed-Rank Test
        print("\n    2. Wilcoxon Signed-Rank Test (Non-parametric):")
        wilcoxon_result = self._wilcoxon_test_full(fedavg_data, fedprox_data)
        self.test_results[scenario]['wilcoxon'] = wilcoxon_result
        
        # 3. Bootstrap Confidence Intervals
        print("\n    3. Bootstrap Confidence Intervals (n=1000):")
        bootstrap_result = self._bootstrap_ci_full(fedavg_data, fedprox_data)
        self.test_results[scenario]['bootstrap_ci'] = bootstrap_result
        
        # 4. Cohen's d (Effect Size)
        print("\n    4. Cohen's d (Effect Size):")
        cohens_d_result = self._cohens_d_full(fedavg_data, fedprox_data)
        self.test_results[scenario]['cohens_d'] = cohens_d_result
        
        # 5. Paired t-test
        print("\n    5. Paired t-test:")
        ttest_result = self._paired_ttest_full(fedavg_data, fedprox_data)
        self.test_results[scenario]['paired_ttest'] = ttest_result
    
    def _cohens_kappa(self, data1: Dict, data2: Dict, name1: str, name2: str) -> Dict:
        """
        Cohen's Kappa for inter-rater (inter-algorithm) agreement.
        Measures agreement between two classifiers beyond chance.
        """
        if not data1.get('predictions') or not data2.get('predictions'):
            print(f"       Skipped: Predictions not available")
            return {'error': 'Predictions not available'}
        
        try:
            pred1 = data1['predictions'][0] if isinstance(data1['predictions'], list) else data1['predictions']
            pred2 = data2['predictions'][0] if isinstance(data2['predictions'], list) else data2['predictions']
            
            y_pred1 = pred1['y_pred_class']
            y_pred2 = pred2['y_pred_class']
            
            # Build confusion matrix between predictions
            n = len(y_pred1)
            
            # Agreement counts
            agree_0 = np.sum((y_pred1 == 0) & (y_pred2 == 0))
            agree_1 = np.sum((y_pred1 == 1) & (y_pred2 == 1))
            disagree_01 = np.sum((y_pred1 == 0) & (y_pred2 == 1))
            disagree_10 = np.sum((y_pred1 == 1) & (y_pred2 == 0))
            
            # Observed agreement
            p_o = (agree_0 + agree_1) / n
            
            # Expected agreement by chance
            p1_0 = (agree_0 + disagree_01) / n  # P(model1 predicts 0)
            p2_0 = (agree_0 + disagree_10) / n  # P(model2 predicts 0)
            p1_1 = (agree_1 + disagree_10) / n  # P(model1 predicts 1)
            p2_1 = (agree_1 + disagree_01) / n  # P(model2 predicts 1)
            
            p_e = p1_0 * p2_0 + p1_1 * p2_1
            
            # Cohen's Kappa
            kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0
            
            # Standard error (for large samples)
            se = np.sqrt(p_o * (1 - p_o) / (n * (1 - p_e)**2)) if (1 - p_e) > 0 else 0
            
            # Interpretation
            if kappa < 0:
                interpretation = "Poor (less than chance)"
            elif kappa < 0.20:
                interpretation = "Slight"
            elif kappa < 0.40:
                interpretation = "Fair"
            elif kappa < 0.60:
                interpretation = "Moderate"
            elif kappa < 0.80:
                interpretation = "Substantial"
            else:
                interpretation = "Almost Perfect"
            
            result = {
                'kappa': float(kappa),
                'se': float(se),
                'ci_lower': float(kappa - 1.96 * se),
                'ci_upper': float(kappa + 1.96 * se),
                'observed_agreement': float(p_o),
                'expected_agreement': float(p_e),
                'interpretation': interpretation,
                'confusion_matrix': {
                    'agree_0': int(agree_0),
                    'agree_1': int(agree_1),
                    'disagree_01': int(disagree_01),
                    'disagree_10': int(disagree_10)
                }
            }
            
            print(f"       {name1} vs {name2}:")
            print(f"         Kappa = {kappa:.4f} (SE = {se:.4f})")
            print(f"         95% CI: [{kappa - 1.96*se:.4f}, {kappa + 1.96*se:.4f}]")
            print(f"         Agreement: {p_o*100:.1f}% observed, {p_e*100:.1f}% expected")
            print(f"         Interpretation: {interpretation}")
            
            return result
            
        except Exception as e:
            print(f"       Error: {str(e)}")
            return {'error': str(e)}
    
    def _wilcoxon_test_full(self, data1: Dict, data2: Dict) -> Dict:
        """
        Wilcoxon Signed-Rank Test (non-parametric paired test).
        Falls back to Mann-Whitney U if samples are unpaired.
        """
        auc_pr1 = np.array(data1.get('auc_pr', []))
        auc_pr2 = np.array(data2.get('auc_pr', []))
        
        if len(auc_pr1) < 2 or len(auc_pr2) < 2:
            print("       Skipped: Insufficient samples")
            return {'error': 'Insufficient samples'}
        
        try:
            if len(auc_pr1) == len(auc_pr2):
                # Paired: Wilcoxon signed-rank
                statistic, p_value = stats.wilcoxon(auc_pr2, auc_pr1, alternative='two-sided')
                test_type = 'wilcoxon_signed_rank'
                
                # Effect size: r = Z / sqrt(N)
                n = len(auc_pr1)
                z = stats.norm.ppf(1 - p_value/2)
                r = z / np.sqrt(n)
                
            else:
                # Unpaired: Mann-Whitney U
                statistic, p_value = stats.mannwhitneyu(auc_pr2, auc_pr1, alternative='two-sided')
                test_type = 'mann_whitney_u'
                
                # Effect size: r = Z / sqrt(N)
                n1, n2 = len(auc_pr1), len(auc_pr2)
                z = stats.norm.ppf(1 - p_value/2)
                r = z / np.sqrt(n1 + n2)
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            result = {
                'test_type': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size_r': float(r),
                'significant': bool(p_value < Config.ALPHA)
            }
            
            print(f"       Test: {test_type}")
            print(f"       Statistic = {statistic:.4f}, p = {p_value:.6f} {sig}")
            print(f"       Effect size r = {r:.4f}")
            
            return result
            
        except Exception as e:
            print(f"       Error: {str(e)}")
            return {'error': str(e)}
    
    def _bootstrap_ci_full(self, data1: Dict, data2: Dict) -> Dict:
        """
        Bootstrap Confidence Intervals for performance difference.
        Uses n=1000 iterations for robust estimates.
        """
        auc_pr1 = np.array(data1.get('auc_pr', []))
        auc_pr2 = np.array(data2.get('auc_pr', []))
        
        if len(auc_pr1) == 0 or len(auc_pr2) == 0:
            print("       Skipped: No data available")
            return {'error': 'No data available'}
        
        np.random.seed(42)
        n_bootstrap = Config.N_BOOTSTRAP
        
        # Bootstrap for each metric
        metrics = ['auc_pr', 'auc', 'accuracy', 'f1']
        results = {}
        
        for metric in metrics:
            vals1 = np.array(data1.get(metric, []))
            vals2 = np.array(data2.get(metric, []))
            
            if len(vals1) == 0 or len(vals2) == 0:
                continue
            
            boot_means1 = []
            boot_means2 = []
            boot_diffs = []
            
            for _ in range(n_bootstrap):
                sample1 = np.random.choice(vals1, len(vals1), replace=True)
                sample2 = np.random.choice(vals2, len(vals2), replace=True)
                
                boot_means1.append(np.mean(sample1))
                boot_means2.append(np.mean(sample2))
                boot_diffs.append(np.mean(sample2) - np.mean(sample1))
            
            results[metric] = {
                'fedavg': {
                    'mean': float(np.mean(vals1)),
                    'bootstrap_mean': float(np.mean(boot_means1)),
                    'ci_lower': float(np.percentile(boot_means1, 2.5)),
                    'ci_upper': float(np.percentile(boot_means1, 97.5))
                },
                'fedprox': {
                    'mean': float(np.mean(vals2)),
                    'bootstrap_mean': float(np.mean(boot_means2)),
                    'ci_lower': float(np.percentile(boot_means2, 2.5)),
                    'ci_upper': float(np.percentile(boot_means2, 97.5))
                },
                'difference': {
                    'mean': float(np.mean(boot_diffs)),
                    'ci_lower': float(np.percentile(boot_diffs, 2.5)),
                    'ci_upper': float(np.percentile(boot_diffs, 97.5)),
                    'excludes_zero': bool(np.percentile(boot_diffs, 2.5) > 0 or np.percentile(boot_diffs, 97.5) < 0)
                }
            }
        
        # Print summary for AUC-PR
        if 'auc_pr' in results:
            r = results['auc_pr']
            print(f"       AUC-PR Bootstrap Results (n={n_bootstrap}):")
            print(f"         FedAvg:  {r['fedavg']['mean']:.4f} [{r['fedavg']['ci_lower']:.4f}, {r['fedavg']['ci_upper']:.4f}]")
            print(f"         FedProx: {r['fedprox']['mean']:.4f} [{r['fedprox']['ci_lower']:.4f}, {r['fedprox']['ci_upper']:.4f}]")
            print(f"         Diff:    {r['difference']['mean']:+.4f} [{r['difference']['ci_lower']:+.4f}, {r['difference']['ci_upper']:+.4f}]")
            print(f"         CI excludes zero: {r['difference']['excludes_zero']}")
        
        results['n_bootstrap'] = n_bootstrap
        return results
    
    def _cohens_d_full(self, data1: Dict, data2: Dict) -> Dict:
        """Cohen's d effect size with confidence interval."""
        auc_pr1 = np.array(data1.get('auc_pr', []))
        auc_pr2 = np.array(data2.get('auc_pr', []))
        
        if len(auc_pr1) == 0 or len(auc_pr2) == 0:
            return {'error': 'No data available'}
        
        mean1, mean2 = np.mean(auc_pr1), np.mean(auc_pr2)
        n1, n2 = len(auc_pr1), len(auc_pr2)
        
        # Pooled standard deviation
        var1 = np.var(auc_pr1, ddof=1) if n1 > 1 else 0
        var2 = np.var(auc_pr2, ddof=1) if n2 > 1 else 0
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1
        
        d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        # Standard error of d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        
        # 95% CI
        ci_lower = d - 1.96 * se_d
        ci_upper = d + 1.96 * se_d
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        result = {
            'cohens_d': float(d),
            'se': float(se_d),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'interpretation': interpretation,
            'pooled_std': float(pooled_std)
        }
        
        print(f"       Cohen's d = {d:.4f} (SE = {se_d:.4f})")
        print(f"       95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"       Interpretation: {interpretation}")
        
        return result
    
    def _paired_ttest_full(self, data1: Dict, data2: Dict) -> Dict:
        """Paired or Welch's t-test."""
        auc_pr1 = np.array(data1.get('auc_pr', []))
        auc_pr2 = np.array(data2.get('auc_pr', []))
        
        if len(auc_pr1) < 2 or len(auc_pr2) < 2:
            return {'error': 'Insufficient samples'}
        
        try:
            if len(auc_pr1) == len(auc_pr2):
                t_stat, p_value = stats.ttest_rel(auc_pr2, auc_pr1)
                test_type = 'paired'
                df = len(auc_pr1) - 1
            else:
                t_stat, p_value = stats.ttest_ind(auc_pr2, auc_pr1, equal_var=False)
                test_type = 'welch'
                # Welch-Satterthwaite df
                s1, s2 = np.var(auc_pr1, ddof=1), np.var(auc_pr2, ddof=1)
                n1, n2 = len(auc_pr1), len(auc_pr2)
                df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            result = {
                'test_type': test_type,
                't_statistic': float(t_stat),
                'df': float(df),
                'p_value': float(p_value),
                'significant': bool(p_value < Config.ALPHA)
            }
            
            print(f"       Test: {test_type} t-test")
            print(f"       t({df:.1f}) = {t_stat:.4f}, p = {p_value:.6f} {sig}")
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    # COMMUNICATION EFFICIENCY ANALYSIS
    
    def _communication_efficiency_analysis(self):
        print("\n  1. Pearson Correlation (Rounds vs. Performance):")
        print("  " + "-"*50)
        
        correlation_results = {}
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.results:
                continue
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            
            for algo in ['fedavg', 'fedprox']:
                if algo not in self.results[scenario]:
                    continue
                
                histories = self.results[scenario][algo].get('history', [])
                if not histories:
                    continue
                
                # Extract round-by-round performance from first history
                hist = histories[0]
                if not isinstance(hist, dict):
                    continue
                
                rounds = hist.get('rounds', list(range(1, 11)))
                val_auc_pr = hist.get('val_auc_pr', [])
                val_acc = hist.get('val_acc', [])
                
                if len(rounds) >= 3 and len(val_auc_pr) >= 3:
                    # Pearson correlation
                    r_auc_pr, p_auc_pr = stats.pearsonr(rounds[:len(val_auc_pr)], val_auc_pr)
                    
                    algo_name = Config.ALGO_NAMES.get(algo, algo)
                    key = f"{scenario}_{algo}"
                    
                    correlation_results[key] = {
                        'scenario': scenario_name,
                        'algorithm': algo_name,
                        'rounds_vs_auc_pr': {
                            'r': float(r_auc_pr),
                            'p_value': float(p_auc_pr),
                            'n_rounds': len(rounds)
                        }
                    }
                    
                    sig = '*' if p_auc_pr < 0.05 else 'ns'
                    print(f"    {scenario_name} - {algo_name}: r = {r_auc_pr:.4f}, p = {p_auc_pr:.4f} {sig}")
        
        self.test_results['communication_correlation'] = correlation_results
        
        # Mann-Whitney U: Federated vs Centralized timing
        print("\n  2. Mann-Whitney U Test (FL vs Centralized Timing):")
        print("  " + "-"*50)
        
        fl_times = []
        cent_times = []
        
        for scenario in self.results:
            for algo in ['fedavg', 'fedprox']:
                if algo in self.results[scenario]:
                    timing = self.results[scenario][algo].get('timing', [])
                    for t in timing:
                        if isinstance(t, dict):
                            total = t.get('total_time', t.get('training_time', 0))
                            if total > 0:
                                fl_times.append(total)
            
            for algo in ['logistic_regression', 'random_forest']:
                if algo in self.results[scenario]:
                    timing = self.results[scenario][algo].get('timing', [])
                    for t in timing:
                        if isinstance(t, dict):
                            total = t.get('total_time', t.get('training_time', 0))
                            if total > 0:
                                cent_times.append(total)
        
        if len(fl_times) >= 2 and len(cent_times) >= 1:
            try:
                u_stat, p_value = stats.mannwhitneyu(fl_times, cent_times, alternative='two-sided')
                
                self.test_results['timing_comparison'] = {
                    'federated': {
                        'n': len(fl_times),
                        'mean': float(np.mean(fl_times)),
                        'median': float(np.median(fl_times)),
                        'std': float(np.std(fl_times))
                    },
                    'centralized': {
                        'n': len(cent_times),
                        'mean': float(np.mean(cent_times)),
                        'median': float(np.median(cent_times)),
                        'std': float(np.std(cent_times))
                    },
                    'mann_whitney_u': {
                        'u_statistic': float(u_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < Config.ALPHA)
                    }
                }
                
                sig = '*' if p_value < 0.05 else 'ns'
                print(f"    Federated: {np.mean(fl_times):.2f}s (n={len(fl_times)})")
                print(f"    Centralized: {np.mean(cent_times):.2f}s (n={len(cent_times)})")
                print(f"    Mann-Whitney U = {u_stat:.2f}, p = {p_value:.4f} {sig}")
                
            except Exception as e:
                print(f"    Error: {e}")
        else:
            print("    Skipped: Insufficient timing data")
    
   
    # ABLATION STUDY
    
    def _ablation_study_anova(self):
        """Complete ablation study with ANOVA and Tukey HSD for mu parameter."""
        print("\n" + "="*70)
        print("ABLATION STUDY: Effect of mu Parameter")
        print("="*70)
        
        ablation_results = {}
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.results_by_mu:
                continue
            
            scenario_name = Config.SCENARIO_NAMES.get(scenario, scenario)
            print(f"\n  {scenario_name}:")
            print("  " + "-"*50)
            
            mu_data = self.results_by_mu[scenario]
            
            if len(mu_data) < 2:
                print("    Skipped: Need at least 2 mu values")
                continue
            
            # Prepare data
            groups = []
            labels = []
            all_values = []
            all_labels = []
            mu_summary = []
            
            for mu_key in sorted(mu_data.keys(), key=lambda x: mu_data[x]['mu']):
                data = mu_data[mu_key]
                if data['auc_pr'] and len(data['auc_pr']) > 0:
                    mu_val = data['mu']
                    groups.append(data['auc_pr'])
                    labels.append(f"mu={mu_val}")
                    all_values.extend(data['auc_pr'])
                    all_labels.extend([f"mu={mu_val}"] * len(data['auc_pr']))
                    
                    mu_summary.append({
                        'mu': mu_val,
                        'mean': float(np.mean(data['auc_pr'])),
                        'std': float(np.std(data['auc_pr'])),
                        'n': len(data['auc_pr'])
                    })
            
            if len(groups) < 2:
                continue
            
            # Print mu summary
            print("    mu values:")
            for s in mu_summary:
                print(f"      mu={s['mu']}: {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n']})")
            
            # One-way ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Effect sizes
                ss_between = sum(len(g) * (np.mean(g) - np.mean(all_values))**2 for g in groups)
                ss_total = sum((v - np.mean(all_values))**2 for v in all_values)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                print(f"\n    ANOVA: F = {f_stat:.4f}, p = {p_value:.6f} {sig}")
                print(f"    Effect size (eta^2) = {eta_squared:.4f}")
                
                # Find optimal mu
                optimal_idx = np.argmax([s['mean'] for s in mu_summary])
                optimal_mu = mu_summary[optimal_idx]['mu']
                print(f"    Optimal mu: {optimal_mu} (AUC-PR = {mu_summary[optimal_idx]['mean']:.4f})")
                
                ablation_results[scenario] = {
                    'mu_summary': mu_summary,
                    'anova': {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'eta_squared': float(eta_squared),
                        'significant': bool(p_value < Config.ALPHA)
                    },
                    'optimal_mu': float(optimal_mu)
                }
                
                # Tukey HSD if significant
                if p_value < Config.ALPHA and len(all_values) > len(groups):
                    try:
                        tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=Config.ALPHA)
                        print(f"\n    Tukey HSD Post-hoc:")
                        print(tukey)
                        ablation_results[scenario]['tukey_hsd'] = str(tukey)
                    except Exception as e:
                        print(f"    Tukey HSD failed: {e}")
                        
            except Exception as e:
                print(f"    Error: {e}")
        
        self.test_results['ablation'] = ablation_results
    
    # SUMMARY
    
    def _print_summary_table(self):
        """Print comprehensive summary table."""
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*70)
        
        print("\n  Scenario Comparisons (FedAvg vs FedProx on AUC-PR):")
        print("  " + "-"*65)
        print(f"  {'Scenario':<25} {'McNemar':<12} {'DeLong':<12} {'Wilcoxon':<12} {'Cohen d':<12}")
        print("  " + "-"*65)
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.test_results:
                continue
            
            s = self.test_results[scenario]
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            
            # McNemar p-value
            mcnemar = s.get('mcnemar_fedavg_fedprox', {})
            mcnemar_p = f"p={mcnemar.get('p_value', 'N/A'):.4f}" if 'p_value' in mcnemar else 'N/A'
            
            # DeLong p-value
            delong = s.get('delong_fedavg_fedprox', {})
            delong_p = f"p={delong.get('p_value', 'N/A'):.4f}" if 'p_value' in delong else 'N/A'
            
            # Wilcoxon p-value
            wilcoxon = s.get('wilcoxon', {})
            wilcoxon_p = f"p={wilcoxon.get('p_value', 'N/A'):.4f}" if 'p_value' in wilcoxon else 'N/A'
            
            # Cohen's d
            cohens = s.get('cohens_d', {})
            cohens_d = f"d={cohens.get('cohens_d', 'N/A'):.3f}" if 'cohens_d' in cohens else 'N/A'
            
            print(f"  {scenario_name:<25} {mcnemar_p:<12} {delong_p:<12} {wilcoxon_p:<12} {cohens_d:<12}")
        
        print("  " + "-"*65)
        print("\n  Significance: *** p<0.001, ** p<0.01, * p<0.05")


# FIGURE GENERATION (Complete)

class FigureGenerator:
    """Generate all publication-quality figures."""
    
    def __init__(self, results: Dict, stats: Dict, results_by_mu: Dict):
        """Initialize generator."""
        self.results = results
        self.stats = stats
        self.results_by_mu = results_by_mu
        self.figures_created = []

    def _load_test_data_for_analysis(self):
        test_file = Path(".") / "data" / "cleaned" / "test_set.csv"
        
        if not test_file.exists():
            print(f"      Test file not found: {test_file}")
            return None
        
        try:
            test_df = pd.read_csv(test_file)
            print(f"    ✓ Loaded test data: {len(test_df)} samples")
            return test_df
        except Exception as e:
            print(f"      Error loading test data: {e}")
            return None
        
    def generate_all_figures(self):
        """Generate all figures."""
        print("\n" + "="*60)
        print("GENERATING FIGURES")
        print("="*60)
        
        # Core figures
        self.figure_1_performance_comparison()
        self.figure_2_roc_pr_curves()
        self.figure_3_metric_distributions()
        self.figure_4_confusion_matrices()
        self.figure_5_convergence_analysis()
        self.figure_6_ablation_study()
        self.figure_7_statistical_summary()
        self.figure_8_radar_comparison()
        self.figure_9_error_analysis()  # <-- ADD THIS LINE
    
        
        print(f"\n  ✓ Generated {len(self.figures_created)} figures")
    
    def figure_1_performance_comparison(self):
        """Figure 1: Performance comparison across scenarios."""
        print("\n  Figure 1: Performance Comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        scenarios = ['s1_iid', 's2_noniid', 's3_quality']
        algorithms = ['fedavg', 'fedprox', 'logistic_regression', 'random_forest']
        
        for idx, metric in enumerate(['auc_pr', 'auc']):
            ax = axes[idx]
            x = np.arange(len(scenarios))
            width = 0.2
            
            for i, algo in enumerate(algorithms):
                means = []
                stds = []
                
                for scenario in scenarios:
                    if scenario in self.results and algo in self.results[scenario]:
                        data = self.results[scenario][algo][metric]
                        if data:
                            means.append(np.mean(data))
                            stds.append(np.std(data))
                        else:
                            means.append(0)
                            stds.append(0)
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - 1.5) * width
                algo_name = Config.ALGO_NAMES.get(algo, algo)
                color = Config.COLORS.get(algo, '#999999')
                
                bars = ax.bar(x + offset, means, width, yerr=stds,
                             label=algo_name, capsize=4, color=color, alpha=0.85)
            
            ylabel = 'AUC-PR' if metric == 'auc_pr' else 'AUC-ROC'
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_xlabel('Scenario', fontsize=13, fontweight='bold')
            ax.set_title(f'Performance Comparison ({ylabel})', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([Config.SCENARIO_NAMES[s].split(': ')[1] for s in scenarios])
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.75, 1.0])
        
        # Single legend below the figure (outside chart area)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=11, 
                   bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)  # Make room for legend below
        self._save_figure('figure_1_performance_comparison')
    
    def figure_2_roc_pr_curves(self):
        """Figure 2: ROC and PR curves."""
        print("\n  Figure 2: ROC and PR Curves...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        scenarios = ['s1_iid', 's2_noniid', 's3_quality']
        
        for col, scenario in enumerate(scenarios):
            if scenario not in self.results:
                continue
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            
            # ROC Curves (top row)
            ax_roc = axes[0, col]
            # PR Curves (bottom row)
            ax_pr = axes[1, col]
            
            for algo in ['fedavg', 'fedprox', 'logistic_regression', 'random_forest']:
                if algo not in self.results[scenario]:
                    continue
                
                algo_data = self.results[scenario][algo]
                algo_name = Config.ALGO_NAMES.get(algo, algo)
                color = Config.COLORS.get(algo, '#999999')
                linestyle = Config.LINE_STYLES.get(algo_name, '-')
                
                # ROC curve
                if algo_data['roc_curves']:
                    roc = algo_data['roc_curves'][0]
                    auc_val = np.mean(algo_data['auc']) if algo_data['auc'] else 0
                    ax_roc.plot(roc['fpr'], roc['tpr'], 
                               color=color, linestyle=linestyle, linewidth=2,
                               label=f'{algo_name} (AUC={auc_val:.3f})')
                
                # PR curve
                if algo_data['pr_curves']:
                    pr = algo_data['pr_curves'][0]
                    ap_val = np.mean(algo_data['auc_pr']) if algo_data['auc_pr'] else 0
                    ax_pr.plot(pr['recall'], pr['precision'],
                              color=color, linestyle=linestyle, linewidth=2,
                              label=f'{algo_name} (AP={ap_val:.3f})')
            
            # ROC formatting
            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC - {scenario_name}', fontweight='bold')
            ax_roc.legend(loc='lower right', fontsize=9)
            ax_roc.grid(alpha=0.3)
            
            # PR formatting
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title(f'PR - {scenario_name}', fontweight='bold')
            ax_pr.legend(loc='lower left', fontsize=9)
            ax_pr.grid(alpha=0.3)
        
        plt.suptitle('ROC and Precision-Recall Curves Across Scenarios', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('figure_2_roc_pr_curves')
    
    def figure_3_metric_distributions(self):
        """Figure 3: Box/violin plots of metric distributions."""
        print("\n  Figure 3: Metric Distributions...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics = ['auc_pr', 'auc', 'accuracy', 'f1', 'precision', 'recall']
        metric_names = ['AUC-PR', 'AUC-ROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 3, idx % 3]
            
            # Collect data
            plot_data = []
            labels = []
            colors = []
            
            for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
                if scenario not in self.results:
                    continue
                
                scenario_short = scenario.split('_')[0].upper()
                
                for algo in ['fedavg', 'fedprox']:
                    if algo not in self.results[scenario]:
                        continue
                    
                    data = self.results[scenario][algo][metric]
                    if data and len(data) > 1:
                        plot_data.append(data)
                        algo_name = Config.ALGO_NAMES.get(algo, algo)
                        labels.append(f'{scenario_short}\n{algo_name}')
                        colors.append(Config.COLORS.get(algo, '#999999'))
            
            if plot_data:
                bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
                
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(f'{metric_name} Distribution', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Metric Distributions Across Scenarios (FL Models Only)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('figure_3_metric_distributions')
    


    def figure_4_confusion_matrices(self):
        """Figure 4: Confusion matrices using median-performing seed."""
        print("\n  Figure 4: Confusion Matrices...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        plot_configs = [
            ('s1_iid', 'fedavg'), ('s1_iid', 'fedprox'),
            ('s2_noniid', 'fedavg'), ('s2_noniid', 'fedprox'),
            ('s3_quality', 'fedavg'), ('s3_quality', 'fedprox'),
            ('s1_iid', 'logistic_regression'), ('s1_iid', 'random_forest')
        ]
        
        for idx, (scenario, algo) in enumerate(plot_configs):
            ax = axes[idx // 4, idx % 4]
            
            if scenario not in self.results or algo not in self.results[scenario]:
                ax.axis('off')
                continue
            
            cms = self.results[scenario][algo]['confusion_matrices']
            if not cms:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Use median-performing seed instead of averaging
            # This ensures all confusion matrices sum to exactly 2,058
            auc_prs = self.results[scenario][algo].get('auc_pr', [])
            
            if len(cms) == 1:
                # Single run (centralized models) - use directly
                cm = np.array(cms[0])
            elif len(auc_prs) == len(cms):
                # Multiple runs - select median by AUC-PR
                median_idx = np.argsort(auc_prs)[len(auc_prs) // 2]
                cm = np.array(cms[median_idx])
            else:
                # Fallback - use first
                cm = np.array(cms[0])
        
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                    cbar=False)
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            algo_name = Config.ALGO_NAMES.get(algo, algo)
            ax.set_title(f'{algo_name}\n{scenario_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('figure_4_confusion_matrices')
        
    def figure_5_convergence_analysis(self):
        """Figure 5: Convergence analysis (training history)."""
        print("\n  Figure 5: Convergence Analysis...")
        
        # Check if history data exists
        has_history = False
        for scenario in self.results:
            for algo in self.results[scenario]:
                histories = self.results[scenario][algo]['history']
                if histories and len(histories) > 0:
                    has_history = True
                    # Debug first history structure
                    first_hist = histories[0]
                    if isinstance(first_hist, dict):
                        print(f"    History keys available: {list(first_hist.keys())}")
                    break
            if has_history:
                break
        
        if not has_history:
            print("    Warning: No training history available - creating placeholder")
            self._create_placeholder_figure('figure_5_convergence_analysis', 
                                           'Convergence Analysis\n(Training history not available)')
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for col, scenario in enumerate(['s1_iid', 's2_noniid', 's3_quality']):
            if scenario not in self.results:
                continue
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            ax_loss = axes[0, col]
            ax_metric = axes[1, col]
            
            for algo in ['fedavg', 'fedprox']:
                if algo not in self.results[scenario]:
                    continue
                
                histories = self.results[scenario][algo]['history']
                if not histories:
                    continue
                
                algo_name = Config.ALGO_NAMES.get(algo, algo)
                color = Config.COLORS.get(algo, '#999999')
                linestyle = '-' if algo == 'fedavg' else '--'
                
                # Extract training loss (use train_loss from your data)
                avg_loss = self._average_history_metric(histories, 'train_loss')
                if avg_loss is None:
                    avg_loss = self._average_history_metric(histories, 'val_loss')
                
                # Extract validation AUC-PR (use val_auc_pr from your data)
                avg_auc_pr = self._average_history_metric(histories, 'val_auc_pr')
                
                if avg_loss is not None and len(avg_loss) > 0:
                    rounds = range(1, len(avg_loss) + 1)
                    ax_loss.plot(rounds, avg_loss, label=algo_name, color=color, 
                                linewidth=2, linestyle=linestyle, marker='o', markersize=4)
                    print(f"    Plotted loss for {algo_name} in {scenario}: {len(avg_loss)} rounds")
                
                if avg_auc_pr is not None and len(avg_auc_pr) > 0:
                    rounds = range(1, len(avg_auc_pr) + 1)
                    ax_metric.plot(rounds, avg_auc_pr, label=algo_name, color=color, 
                                  linewidth=2, linestyle=linestyle, marker='o', markersize=4)
                    print(f"    Plotted AUC-PR for {algo_name} in {scenario}: {len(avg_auc_pr)} rounds")
            
            # Format loss subplot
            ax_loss.set_xlabel('Communication Round', fontsize=12)
            ax_loss.set_ylabel('Training Loss', fontsize=12)
            ax_loss.set_title(f'Training Loss - {scenario_name}', fontweight='bold')
            if ax_loss.get_legend_handles_labels()[0]:
                ax_loss.legend(loc='upper right')
            ax_loss.grid(alpha=0.3)
            ax_loss.set_xlim(left=0.5)
            
            # Format metric subplot
            ax_metric.set_xlabel('Communication Round', fontsize=12)
            ax_metric.set_ylabel('Validation AUC-PR', fontsize=12)
            ax_metric.set_title(f'Validation AUC-PR - {scenario_name}', fontweight='bold')
            if ax_metric.get_legend_handles_labels()[0]:
                ax_metric.legend(loc='lower right')
            ax_metric.grid(alpha=0.3)
            ax_metric.set_xlim(left=0.5)
        
        plt.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('figure_5_convergence_analysis')
    
    def _average_history_metric(self, histories: List, metric_key: str) -> Optional[np.ndarray]:
        metric_arrays = []
        
        for hist in histories:
            if hist is None:
                continue
            
            values = None
            
            # Primary format: Dict with metric key directly containing list
            if isinstance(hist, dict):
                # Direct key match
                if metric_key in hist:
                    val = hist[metric_key]
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        values = [float(v) for v in val]
                
                # If not found, try without prefix/suffix variations
                if values is None:
                    # Map common variations
                    key_variations = {
                        'loss': ['train_loss', 'val_loss', 'loss'],
                        'train_loss': ['train_loss', 'loss'],
                        'val_loss': ['val_loss', 'loss'],
                        'auc_pr': ['val_auc_pr', 'auc_pr', 'pr_auc'],
                        'val_auc_pr': ['val_auc_pr', 'auc_pr'],
                        'auc': ['val_auc', 'auc', 'roc_auc'],
                        'val_auc': ['val_auc', 'auc'],
                        'accuracy': ['val_acc', 'train_acc', 'accuracy', 'acc'],
                        'val_acc': ['val_acc', 'accuracy'],
                        'f1': ['val_f1', 'f1'],
                        'precision': ['val_precision', 'precision'],
                        'recall': ['val_recall', 'recall']
                    }
                    
                    variations = key_variations.get(metric_key, [metric_key])
                    for var in variations:
                        if var in hist:
                            val = hist[var]
                            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                                values = [float(v) for v in val]
                                break
            
            if values and len(values) > 0:
                metric_arrays.append(values)
        
        if not metric_arrays:
            return None
        
        # Pad to same length (use last value for padding)
        max_len = max(len(arr) for arr in metric_arrays)
        padded = []
        for arr in metric_arrays:
            if len(arr) < max_len:
                arr = list(arr) + [arr[-1]] * (max_len - len(arr))
            padded.append(arr[:max_len])
        
        return np.mean(padded, axis=0)
    
    def figure_6_ablation_study(self):
        """Figure 6: Ablation study (effect of μ)."""
        print("\n  Figure 6: Ablation Study...")
        
        if not self.results_by_mu:
            print("      No ablation data available - creating placeholder")
            self._create_placeholder_figure('figure_6_ablation_study',
                                           'Ablation Study\n(μ variation data not available)')
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for col, scenario in enumerate(['s1_iid', 's2_noniid', 's3_quality']):
            ax = axes[col]
            
            if scenario not in self.results_by_mu:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            mu_data = self.results_by_mu[scenario]
            
            # Sort by mu value
            sorted_items = sorted(mu_data.items(), key=lambda x: x[1]['mu'])
            
            mus = []
            means = []
            stds = []
            
            for mu_key, data in sorted_items:
                if data['auc_pr']:
                    mus.append(data['mu'])
                    means.append(np.mean(data['auc_pr']))
                    stds.append(np.std(data['auc_pr']))
            
            if mus:
                ax.errorbar(range(len(mus)), means, yerr=stds, 
                           marker='o', capsize=5, linewidth=2, markersize=8,
                           color='#1f77b4')
                
                ax.set_xticks(range(len(mus)))
                ax.set_xticklabels([f'{m}' for m in mus])
                
                # Highlight optimal
                opt_idx = np.argmax(means)
                ax.scatter([opt_idx], [means[opt_idx]], s=200, c='red', 
                          marker='*', zorder=5, label=f'Optimal: μ={mus[opt_idx]}')
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            ax.set_xlabel('μ (Proximal Term Weight)', fontweight='bold')
            ax.set_ylabel('AUC-PR', fontweight='bold')
            ax.set_title(f'Effect of μ - {scenario_name}', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Ablation Study: FedProx μ Parameter', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('figure_6_ablation_study')
    
    def figure_7_statistical_summary(self):
        """Figure 7: Statistical test summary heatmap."""
        print("\n  Figure 7: Statistical Summary...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Build summary matrix
        scenarios = ['s1_iid', 's2_noniid', 's3_quality']
        tests = ['t-test', 'Wilcoxon', 'DeLong', 'McNemar', "Cohen's Kappa"]
        
        p_values = np.ones((len(scenarios), len(tests))) * np.nan
        
        for i, scenario in enumerate(scenarios):
            if scenario not in self.stats:
                continue
            
            s = self.stats[scenario]
            
            # t-test
            if 'paired_ttest' in s and isinstance(s['paired_ttest'], dict) and 'p_value' in s['paired_ttest']:
                p_values[i, 0] = s['paired_ttest']['p_value']
            
            # Wilcoxon
            if 'wilcoxon' in s and isinstance(s['wilcoxon'], dict) and 'p_value' in s['wilcoxon']:
                p_values[i, 1] = s['wilcoxon']['p_value']
            
            # DeLong - check both old and new key names
            delong_key = 'delong_fedavg_fedprox' if 'delong_fedavg_fedprox' in s else 'delong'
            if delong_key in s and isinstance(s[delong_key], dict) and 'p_value' in s[delong_key]:
                p_values[i, 2] = s[delong_key]['p_value']
            
            # McNemar - check both old and new key names
            mcnemar_key = 'mcnemar_fedavg_fedprox' if 'mcnemar_fedavg_fedprox' in s else 'mcnemar'
            if mcnemar_key in s and isinstance(s[mcnemar_key], dict) and 'p_value' in s[mcnemar_key]:
                p_values[i, 3] = s[mcnemar_key]['p_value']
            
            # Cohen's Kappa (display as 1-kappa to show on same scale, higher kappa = lower "p-value" equivalent)
            if 'cohens_kappa' in s and isinstance(s['cohens_kappa'], dict) and 'kappa' in s['cohens_kappa']:
                kappa = s['cohens_kappa']['kappa']
                p_values[i, 4] = max(0, 1 - kappa)
        
        # Create heatmap
        mask = np.isnan(p_values)
        
        # Annotation: show p-values and significance
        annot = np.empty_like(p_values, dtype=object)
        for i in range(p_values.shape[0]):
            for j in range(p_values.shape[1]):
                if np.isnan(p_values[i, j]):
                    annot[i, j] = 'N/A'
                elif j == 4:  
                    kappa = 1 - p_values[i, j]
                    if kappa >= 0.8:
                        interp = 'Almost Perfect'
                    elif kappa >= 0.6:
                        interp = 'Substantial'
                    elif kappa >= 0.4:
                        interp = 'Moderate'
                    else:
                        interp = 'Fair/Poor'
                    annot[i, j] = f'k={kappa:.3f}\n{interp}'
                else:
                    p = p_values[i, j]
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    annot[i, j] = f'{p:.4f}\n{sig}'
        
        sns.heatmap(p_values, annot=annot, fmt='', cmap='RdYlGn_r',
                   xticklabels=tests, 
                   yticklabels=[Config.SCENARIO_NAMES[s].split(': ')[1] for s in scenarios],
                   ax=ax, vmin=0, vmax=0.1, mask=mask,
                   cbar_kws={'label': 'p-value (lower = more significant)'})
        
        ax.set_title('Statistical Test Results (FedAvg vs FedProx)\n*** p<0.001, ** p<0.01, * p<0.05',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure('figure_7_statistical_summary')
    
    def figure_8_radar_comparison(self):
        """Figure 8: Radar chart comparing algorithms."""
        print("\n  Figure 8: Radar Comparison...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
        
        metrics = ['AUC-PR', 'AUC-ROC', 'Accuracy', 'F1', 'Precision', 'Recall']
        metric_keys = ['auc_pr', 'auc', 'accuracy', 'f1', 'precision', 'recall']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        for col, scenario in enumerate(['s1_iid', 's2_noniid', 's3_quality']):
            ax = axes[col]
            
            if scenario not in self.results:
                continue
            
            for algo in ['fedavg', 'fedprox']:
                if algo not in self.results[scenario]:
                    continue
                
                algo_data = self.results[scenario][algo]
                algo_name = Config.ALGO_NAMES.get(algo, algo)
                color = Config.COLORS.get(algo, '#999999')
                
                values = []
                for key in metric_keys:
                    if algo_data[key]:
                        values.append(np.mean(algo_data[key]))
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the loop
                
                ax.plot(angles, values, 'o-', linewidth=2, label=algo_name, color=color)
                ax.fill(angles, values, alpha=0.25, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, size=10)
            ax.set_ylim([0.7, 1.0])
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            ax.set_title(scenario_name, fontsize=12, fontweight='bold', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.suptitle('Multi-Metric Comparison (Radar Charts)', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        self._save_figure('figure_8_radar_comparison')
    

    def figure_9_error_analysis(self):
        print("\n  Figure 9: Error Analysis (Enhanced)...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        scenario = 's2_noniid'
        algo = 'fedavg'
        
        # Load test data for detailed analysis
        test_df = self._load_test_data_for_analysis()
        
        # Get predictions using MEDIAN seed (same as figure_4)
        y_true, y_proba, y_pred = None, None, None
        if scenario in self.results and algo in self.results[scenario]:
            preds = self.results[scenario][algo].get('predictions', [])
            auc_prs = self.results[scenario][algo].get('auc_pr', [])
            
            if preds:
                if len(preds) == 1:
                    pred = preds[0]
                elif len(auc_prs) == len(preds):
                    median_idx = np.argsort(auc_prs)[len(auc_prs) // 2]
                    pred = preds[median_idx]
                    print(f"    Using median seed (index {median_idx}, AUC-PR: {auc_prs[median_idx]:.4f})")
                else:
                    pred = preds[0]
                
                y_true = np.array(pred['y_true'])
                y_proba = np.array(pred['y_pred_proba'])
                y_pred = np.array(pred.get('y_pred_class', (y_proba >= 0.35).astype(int)))
        
        # (a) Overall Error Rates
        ax = axes[0, 0]
        
        if y_true is not None and y_pred is not None:
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
            
            # Print for verification
            print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print(f"    Total: {tn+fp+fn+tp}, FNR: {fnr:.1f}%, FPR: {fpr:.1f}%")
            
            bars = ax.bar(['False Negative', 'False Positive'], [fnr, fpr],
                        color=['#E74C3C', '#3498DB'], alpha=0.8, edgecolor='black')
            
            for bar, val in zip(bars, [fnr, fpr]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Error Rate (%)', fontweight='bold')
            ax.set_title('(a) Overall Error Rates', fontweight='bold')
            ax.set_ylim(0, max(fnr, fpr) * 1.2)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(a) Overall Error Rates', fontweight='bold')
        
        # (b) Regional Error Rates
        ax = axes[0, 1]
        
        regional_analysis_done = False
        if test_df is not None and y_true is not None and y_pred is not None:
            region_col = None
            for col in ['region_id', 'region_num', 'region']:
                if col in test_df.columns:
                    region_col = col
                    break
            
            if region_col and len(test_df) == len(y_true):
                test_df = test_df.copy()
                test_df['y_true'] = y_true
                test_df['y_pred'] = y_pred
                
                region_names = {
                    1: "Western", 2: "Central", 3: "Greater Accra", 4: "Volta",
                    5: "Eastern", 6: "Ashanti", 7: "Brong Ahafo", 8: "Northern",
                    9: "Upper East", 10: "Upper West"
                }
                
                regions = []
                fn_rates = []
                fp_rates = []
                
                for region_id in sorted(test_df[region_col].dropna().unique()):
                    region_data = test_df[test_df[region_col] == region_id]
                    
                    r_tp = ((region_data['y_true'] == 1) & (region_data['y_pred'] == 1)).sum()
                    r_tn = ((region_data['y_true'] == 0) & (region_data['y_pred'] == 0)).sum()
                    r_fp = ((region_data['y_true'] == 0) & (region_data['y_pred'] == 1)).sum()
                    r_fn = ((region_data['y_true'] == 1) & (region_data['y_pred'] == 0)).sum()
                    
                    r_fnr = r_fn / (r_fn + r_tp) * 100 if (r_fn + r_tp) > 0 else 0
                    r_fpr = r_fp / (r_fp + r_tn) * 100 if (r_fp + r_tn) > 0 else 0
                    
                    region_name = region_names.get(int(region_id), f"R{int(region_id)}")
                    short_names = {
                        "Greater Accra": "Gr. Accra",
                        "Brong Ahafo": "Brong A.",
                        "Upper East": "Up. East",
                        "Upper West": "Up. West"
                    }
                    region_name = short_names.get(region_name, region_name)
                    
                    regions.append(region_name)
                    fn_rates.append(r_fnr)
                    fp_rates.append(r_fpr)
                
                if regions:
                    regional_analysis_done = True
                    x = np.arange(len(regions))
                    width = 0.35
                    
                    ax.bar(x - width/2, fn_rates, width, label='FNR', color='#E74C3C', alpha=0.8)
                    ax.bar(x + width/2, fp_rates, width, label='FPR', color='#3498DB', alpha=0.8)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=9)
                    ax.set_ylabel('Error Rate (%)', fontweight='bold')
                    ax.set_title('(b) Regional Error Rates', fontweight='bold')
                    ax.legend(loc='upper right')
                    ax.grid(axis='y', alpha=0.3)
        
        if not regional_analysis_done:
            ax.text(0.5, 0.5, 'Regional data not available\n(requires region_id in test set)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('(b) Regional Error Rates', fontweight='bold')
        
        # (c) Error by Symptom Count
        ax = axes[1, 0]
        
        symptom_analysis_done = False
        symptom_cols = ['fever', 'diarrhea', 'chills', 'sweating', 
                        'headache', 'bodyaches', 'nausea_vomiting', 'appetite_loss']
        
        if test_df is not None and y_true is not None and y_pred is not None:
            available_symptoms = [c for c in symptom_cols if c in test_df.columns]
            
            if available_symptoms and len(test_df) == len(y_true):
                test_df = test_df.copy()
                test_df['y_true'] = y_true
                test_df['y_pred'] = y_pred
                test_df['symptom_count'] = test_df[available_symptoms].sum(axis=1)
                
                test_df['is_fn'] = (test_df['y_true'] == 1) & (test_df['y_pred'] == 0)
                test_df['is_fp'] = (test_df['y_true'] == 0) & (test_df['y_pred'] == 1)
                
                symptom_counts = sorted(test_df['symptom_count'].unique())
                fn_by_count = []
                fp_by_count = []
                
                for sc in symptom_counts:
                    sc_data = test_df[test_df['symptom_count'] == sc]
                    fn_by_count.append(sc_data['is_fn'].sum())
                    fp_by_count.append(sc_data['is_fp'].sum())
                
                if any(fn_by_count) or any(fp_by_count):
                    symptom_analysis_done = True
                    x = np.arange(len(symptom_counts))
                    width = 0.35
                    
                    ax.bar(x - width/2, fn_by_count, width, label='False Negatives', color='#E74C3C', alpha=0.8)
                    ax.bar(x + width/2, fp_by_count, width, label='False Positives', color='#3498DB', alpha=0.8)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(int(s)) for s in symptom_counts])
                    ax.set_xlabel('Number of Symptoms', fontweight='bold')
                    ax.set_ylabel('Count', fontweight='bold')
                    ax.set_title('(c) Errors by Symptom Count', fontweight='bold')
                    ax.legend(loc='upper right')
                    ax.grid(axis='y', alpha=0.3)
        
        if not symptom_analysis_done:
            ax.text(0.5, 0.5, 'Symptom data not available\n(requires symptom features in test set)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('(c) Errors by Symptom Count', fontweight='bold')
        
    
        # (d) Calibration Plot
        ax = axes[1, 1]
        
        if y_true is not None and y_proba is not None:
            try:
                from sklearn.calibration import calibration_curve
                prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
                
                ax.plot(prob_pred, prob_true, 's-', color='#1f77b4',
                    linewidth=2, markersize=8, label='Model')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
                
                ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
                ax.set_ylabel('Fraction of Positives', fontweight='bold')
                ax.set_title('(d) Calibration Plot', fontweight='bold')
                ax.legend(loc='lower right')
                ax.grid(alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
            except Exception as e:
                ax.text(0.5, 0.5, f'Calibration error:\n{str(e)[:50]}',
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('(d) Calibration Plot', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Prediction probabilities not available',
                ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(d) Calibration Plot', fontweight='bold')
        
        # Final formatting
        algo_name = Config.ALGO_NAMES.get(algo, algo)
        plt.suptitle(f'Error Analysis - {algo_name} on S2 (Regional Heterogeneity)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure('figure_9_error_analysis')


    def _create_placeholder_figure(self, name: str, message: str):
        """Create a placeholder figure."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14,
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.axis('off')
        self._save_figure(name)
    
    def _save_figure(self, name: str):
        """Save figure in multiple formats including black & white print version."""
        # Save color versions
        for fmt in ['pdf', 'png']:
            filepath = Config.FIGURES_DIR / f'{name}.{fmt}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save black and white high-resolution print version (300 DPI)
        self._save_bw_figure(name)
        
        plt.close()
        self.figures_created.append(name)
        print(f"    ✓ Saved: {name} (color + B&W print versions)")
    
    def _save_bw_figure(self, name: str):
        """Save black and white high-resolution version for printing (300 DPI)."""
        import io
        from PIL import Image
        
        # Create print directory if it doesn't exist
        print_dir = Config.FIGURES_DIR / "print_bw"
        print_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current figure to buffer at high resolution
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        
        # Open with PIL and convert to grayscale
        img = Image.open(buf)
        
        # Convert to grayscale (L mode)
        img_gray = img.convert('L')
        
        # Convert back to RGB for better compatibility with some printers/viewers
        img_bw = img_gray.convert('RGB')
        
        # Save as PNG (lossless, high quality for print)
        png_path = print_dir / f'{name}_print_bw.png'
        img_bw.save(png_path, 'PNG', dpi=(300, 300))
        
        # Also save as PDF for print (vector where possible, raster content at 300 DPI)
        pdf_path = print_dir / f'{name}_print_bw.pdf'
        img_bw.save(pdf_path, 'PDF', resolution=300.0)
        
        # Also save as TIFF (preferred format for many publishers)
        tiff_path = print_dir / f'{name}_print_bw.tiff'
        img_bw.save(tiff_path, 'TIFF', dpi=(300, 300), compression='tiff_lzw')
        
        buf.close()


# TABLE GENERATION 
class TableGenerator:
    """Generate publication-quality tables."""
    
    def __init__(self, results: Dict, stats: Dict, results_by_mu: Dict):
        """Initialize generator."""
        self.results = results
        self.stats = stats
        self.results_by_mu = results_by_mu
        self.tables_created = []
    
    def generate_all_tables(self):
        """Generate all tables."""
        print("\n" + "="*60)
        print("GENERATING TABLES")
        print("="*60)
        
        self.table_1_performance_metrics()
        self.table_2_statistical_tests()
        self.table_3_centralized_comparison()
        self.table_4_ablation_results()
        self.table_5_comprehensive_stats()
        
        print(f"\n  Generated {len(self.tables_created)} tables")
    
    def _get_training_time(self, data: Dict) -> str:
            """Extract training time from data."""
            # Try timing data first
            timing_data = data.get('timing', [])
            
            if timing_data:
                times = []
                for t in timing_data:
                    if isinstance(t, dict):
                        time_val = t.get('total_time') or t.get('training_time') or t.get('elapsed_time') or t.get('time')
                        if time_val and time_val > 0:
                            times.append(time_val)
                    elif isinstance(t, (int, float)) and t > 0:
                        times.append(t)
                
                if times:
                    avg_time = np.mean(times)
                    if avg_time >= 60:
                        return f"{avg_time/60:.2f}m"  # Minutes
                    else:
                        return f"{avg_time:.1f}s"  # Seconds
            
            # Try history (round times)
            histories = data.get('history', [])
            total_times = []
            for h in histories:
                if isinstance(h, dict) and 'round_time_sec' in h:
                    round_times = h['round_time_sec']
                    if round_times:
                        total_times.append(sum(round_times))
            
            if total_times:
                avg_time = np.mean(total_times)
                if avg_time >= 60:
                    return f"{avg_time/60:.2f}m"
                else:
                    return f"{avg_time:.1f}s"
            
            return 'N/A'

    def table_1_performance_metrics(self):
        """Table 1: Comprehensive performance metrics with CIs."""
        print("\n  Table 1: Performance Metrics...")
        
        rows = []
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.results:
                continue
            
            scenario_name = Config.SCENARIO_NAMES[scenario]
            
            for algo in ['fedavg', 'fedprox', 'logistic_regression', 'random_forest']:
                if algo not in self.results[scenario]:
                    continue
                
                data = self.results[scenario][algo]
                algo_name = Config.ALGO_NAMES.get(algo, algo)
                n = len(data['auc_pr'])
                
                row = {
                    'Scenario': scenario_name,
                    'Algorithm': algo_name,
                    'N': n
                }
                
                # Add metrics with CIs
                for metric, name in [('auc_pr', 'AUC-PR'), ('auc', 'AUC-ROC'),
                                    ('accuracy', 'Accuracy'), ('f1', 'F1'),
                                    ('precision', 'Precision'), ('recall', 'Recall')]:
                    if data[metric]:
                        mean = np.mean(data[metric])
                        if n > 1:
                            ci = stats.t.interval(0.95, n-1, loc=mean, scale=stats.sem(data[metric]))
                            row[name] = f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
                        else:
                            row[name] = f"{mean:.4f}"
                    else:
                        row[name] = 'N/A'
                
                # Add training time if you want it in the table
                row['Training Time'] = self._get_training_time(data)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save with UTF-8 encoding for Windows compatibility
        df.to_csv(Config.TABLES_DIR / 'table_1_performance_metrics.csv', index=False, encoding='utf-8-sig')
        self._save_latex(df, 'table_1_performance_metrics')
        
        self.tables_created.append('table_1_performance_metrics')
        print(f"    ✓ Saved: table_1_performance_metrics")
        print("\n" + df.to_string(index=False))
    def table_2_statistical_tests(self):
        """Table 2: Comprehensive statistical test results."""
        print("\n  Table 2: Statistical Tests (Comprehensive)...")
        
        rows = []
        
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.stats:
                continue
            
            scenario_name = Config.SCENARIO_NAMES[scenario]
            s = self.stats[scenario]
            
            # Get all test results
            desc = s.get('descriptive', {})
            mcnemar = s.get('mcnemar_fedavg_fedprox', {})
            delong = s.get('delong_fedavg_fedprox', {})
            wilcoxon = s.get('wilcoxon', {})
            bootstrap = s.get('bootstrap_ci', {})
            cohens_d = s.get('cohens_d', {})
            kappa = s.get('cohens_kappa', {})
            ttest = s.get('paired_ttest', {})
            
            row = {
                'Scenario': scenario_name,
                'FedAvg AUC-PR': 'N/A',
                'FedProx AUC-PR': 'N/A',
                'Diff': 'N/A',
                "McNemar p": 'N/A',
                "DeLong p": 'N/A',
                "Wilcoxon p": 'N/A',
                "Cohen's d": 'N/A',
                "Kappa": 'N/A',
                '95% CI': 'N/A'
            }
            
            # Descriptive stats
            if desc.get('fedavg') and desc.get('fedprox'):
                fa = desc['fedavg'].get('auc_pr', {})
                fp = desc['fedprox'].get('auc_pr', {})
                if fa and fp:
                    row['FedAvg AUC-PR'] = f"{fa.get('mean', 0):.4f}+/-{fa.get('std', 0):.4f}"
                    row['FedProx AUC-PR'] = f"{fp.get('mean', 0):.4f}+/-{fp.get('std', 0):.4f}"
                    row['Diff'] = f"{fp.get('mean', 0) - fa.get('mean', 0):+.4f}"
            
            # McNemar test
            if 'p_value' in mcnemar:
                p = mcnemar['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                row["McNemar p"] = f"{p:.4f}{sig}"
            
            # DeLong test
            if 'p_value' in delong:
                p = delong['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                row["DeLong p"] = f"{p:.4f}{sig}"
            
            # Wilcoxon test
            if 'p_value' in wilcoxon:
                p = wilcoxon['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                row["Wilcoxon p"] = f"{p:.4f}{sig}"
            
            # Cohen's d
            if 'cohens_d' in cohens_d:
                d = cohens_d['cohens_d']
                interp = cohens_d.get('interpretation', '')[0] if cohens_d.get('interpretation') else ''
                row["Cohen's d"] = f"{d:.3f}({interp})"
            
            # Cohen's Kappa
            if 'kappa' in kappa:
                k = kappa['kappa']
                row["Kappa"] = f"{k:.3f}"
            
            # Bootstrap CI
            if 'auc_pr' in bootstrap:
                diff = bootstrap['auc_pr'].get('difference', {})
                if diff:
                    row['95% CI'] = f"[{diff.get('ci_lower', 0):+.4f}, {diff.get('ci_upper', 0):+.4f}]"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save with UTF-8 encoding for Windows compatibility
        df.to_csv(Config.TABLES_DIR / 'table_2_statistical_tests.csv', index=False, encoding='utf-8-sig')
        self._save_latex(df, 'table_2_statistical_tests')
        
        self.tables_created.append('table_2_statistical_tests')
        print(f"    Saved: table_2_statistical_tests")
        print("\n" + df.to_string(index=False))
    
    def table_3_centralized_comparison(self):
        """Table 3: Comparison with centralized baselines."""
        print("\n  Table 3: Centralized Comparison...")
        
        rows = []
        
        # Use s1_iid as reference (all scenarios have same centralized baseline)
        if 's1_iid' not in self.results:
            print("      No data for comparison")
            return
        
        scenario_data = self.results['s1_iid']
        
        for algo in ['fedavg', 'fedprox', 'logistic_regression', 'random_forest']:
            if algo not in scenario_data:
                continue
            
            data = scenario_data[algo]
            algo_name = Config.ALGO_NAMES.get(algo, algo)
            
            row = {
                'Model': algo_name,
                'Type': 'Federated' if algo in ['fedavg', 'fedprox'] else 'Centralized',
                'AUC-PR': f"{np.mean(data['auc_pr']):.4f}" if data['auc_pr'] else 'N/A',
                'AUC-ROC': f"{np.mean(data['auc']):.4f}" if data['auc'] else 'N/A',
                'Accuracy': f"{np.mean(data['accuracy'])*100:.2f}%" if data['accuracy'] else 'N/A',
                'F1': f"{np.mean(data['f1']):.4f}" if data['f1'] else 'N/A',
                'Privacy': 'Yes' if algo in ['fedavg', 'fedprox'] else 'No'
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save with UTF-8 encoding for Windows compatibility
        df.to_csv(Config.TABLES_DIR / 'table_3_centralized_comparison.csv', index=False, encoding='utf-8-sig')
        self._save_latex(df, 'table_3_centralized_comparison')
        
        self.tables_created.append('table_3_centralized_comparison')
        print(f"    ✓ Saved: table_3_centralized_comparison")
        print("\n" + df.to_string(index=False))
    
    def table_4_ablation_results(self):
        """Table 4: Ablation study results."""
        print("\n  Table 4: Ablation Results...")
        
        if 'ablation' not in self.stats or not self.stats['ablation']:
            print("      No ablation data available")
            return
        
        rows = []
        
        for scenario, ablation in self.stats['ablation'].items():
            scenario_name = Config.SCENARIO_NAMES[scenario]
            
            for mu_data in ablation.get('mu_summary', []):
                row = {
                    'Scenario': scenario_name,
                    'mu': mu_data['mu'],
                    'AUC-PR (Mean)': f"{mu_data['mean']:.4f}",
                    'AUC-PR (SD)': f"{mu_data['std']:.4f}",
                    'N': mu_data['n'],
                    'Optimal': '*' if mu_data['mu'] == ablation.get('optimal_mu') else ''
                }
                rows.append(row)
        
        if not rows:
            print("      No ablation results to tabulate")
            return
        
        df = pd.DataFrame(rows)
        
        # Save with UTF-8 encoding for Windows compatibility
        df.to_csv(Config.TABLES_DIR / 'table_4_ablation_results.csv', index=False, encoding='utf-8-sig')
        self._save_latex(df, 'table_4_ablation_results')
        
        self.tables_created.append('table_4_ablation_results')
        print(f"    Saved: table_4_ablation_results")
        print("\n" + df.to_string(index=False))
    
    def table_5_comprehensive_stats(self):
        """Table 5: Comprehensive statistical analysis summary."""
        print("\n  Table 5: Comprehensive Statistical Summary...")
        
        # Primary Analysis Table
        primary_rows = []
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.stats:
                continue
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            s = self.stats[scenario]
            
            # McNemar
            mcnemar = s.get('mcnemar_fedavg_fedprox', {})
            mcnemar_p = mcnemar.get('p_value', None)
            mcnemar_str = f"{mcnemar_p:.4f}" if mcnemar_p else 'N/A'
            mcnemar_interp = mcnemar.get('interpretation', 'N/A')
            
            # DeLong
            delong = s.get('delong_fedavg_fedprox', {})
            delong_p = delong.get('p_value', None)
            delong_str = f"{delong_p:.4f}" if delong_p else 'N/A'
            delong_diff = delong.get('difference', None)
            delong_diff_str = f"{delong_diff:+.4f}" if delong_diff else 'N/A'
            
            # ANOVA
            anova = s.get('anova_tukey', {})
            anova_f = anova.get('f_statistic', None)
            anova_p = anova.get('p_value', None)
            anova_str = f"F={anova_f:.2f}, p={anova_p:.4f}" if anova_f and anova_p else 'N/A'
            eta_sq = anova.get('eta_squared', None)
            eta_str = f"{eta_sq:.4f}" if eta_sq else 'N/A'
            
            primary_rows.append({
                'Scenario': scenario_name,
                'McNemar p': mcnemar_str,
                'McNemar Result': mcnemar_interp,
                'DeLong p': delong_str,
                'AUC Diff': delong_diff_str,
                'ANOVA': anova_str,
                'Eta-squared': eta_str
            })
        
        df_primary = pd.DataFrame(primary_rows)
        df_primary.to_csv(Config.TABLES_DIR / 'table_5a_primary_analysis.csv', index=False, encoding='utf-8-sig')
        self._save_latex(df_primary, 'table_5a_primary_analysis')
        
        # Secondary Analysis Table
        secondary_rows = []
        for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
            if scenario not in self.stats:
                continue
            
            scenario_name = Config.SCENARIO_NAMES[scenario].split(': ')[1]
            s = self.stats[scenario]
            
            # Cohen's Kappa
            kappa = s.get('cohens_kappa', {})
            kappa_val = kappa.get('kappa', None)
            kappa_str = f"{kappa_val:.4f}" if kappa_val else 'N/A'
            kappa_interp = kappa.get('interpretation', 'N/A')
            
            # Wilcoxon
            wilcoxon = s.get('wilcoxon', {})
            wilcoxon_p = wilcoxon.get('p_value', None)
            wilcoxon_str = f"{wilcoxon_p:.4f}" if wilcoxon_p else 'N/A'
            wilcoxon_r = wilcoxon.get('effect_size_r', None)
            wilcoxon_r_str = f"{wilcoxon_r:.4f}" if wilcoxon_r else 'N/A'
            
            # Bootstrap CI
            bootstrap = s.get('bootstrap_ci', {})
            if 'auc_pr' in bootstrap:
                diff = bootstrap['auc_pr'].get('difference', {})
                ci_lower = diff.get('ci_lower', None)
                ci_upper = diff.get('ci_upper', None)
                ci_str = f"[{ci_lower:+.4f}, {ci_upper:+.4f}]" if ci_lower is not None else 'N/A'
                excludes_zero = diff.get('excludes_zero', False)
            else:
                ci_str = 'N/A'
                excludes_zero = False
            
            # Cohen's d
            cohens_d = s.get('cohens_d', {})
            d_val = cohens_d.get('cohens_d', None)
            d_str = f"{d_val:.4f}" if d_val else 'N/A'
            d_interp = cohens_d.get('interpretation', 'N/A')
            
            secondary_rows.append({
                'Scenario': scenario_name,
                "Cohen's Kappa": kappa_str,
                'Kappa Interp': kappa_interp,
                'Wilcoxon p': wilcoxon_str,
                'Effect r': wilcoxon_r_str,
                'Bootstrap 95% CI': ci_str,
                'CI Excludes 0': 'Yes' if excludes_zero else 'No',
                "Cohen's d": d_str,
                'd Interp': d_interp
            })
        
        df_secondary = pd.DataFrame(secondary_rows)
        df_secondary.to_csv(Config.TABLES_DIR / 'table_5b_secondary_analysis.csv', index=False, encoding='utf-8-sig')
        self._save_latex(df_secondary, 'table_5b_secondary_analysis')
        
        # Communication Efficiency Table
        comm_results = self.stats.get('communication_correlation', {})
        timing_results = self.stats.get('timing_comparison', {})
        
        if comm_results or timing_results:
            comm_rows = []
            
            for key, data in comm_results.items():
                rounds_data = data.get('rounds_vs_auc_pr', {})
                comm_rows.append({
                    'Scenario': data.get('scenario', 'N/A'),
                    'Algorithm': data.get('algorithm', 'N/A'),
                    'Pearson r': f"{rounds_data.get('r', 0):.4f}",
                    'p-value': f"{rounds_data.get('p_value', 1):.4f}",
                    'N Rounds': rounds_data.get('n_rounds', 'N/A')
                })
            
            if comm_rows:
                df_comm = pd.DataFrame(comm_rows)
                df_comm.to_csv(Config.TABLES_DIR / 'table_5c_communication_efficiency.csv', index=False, encoding='utf-8-sig')
                self._save_latex(df_comm, 'table_5c_communication_efficiency')
        
        self.tables_created.append('table_5_comprehensive_stats')
        print(f"    Saved: table_5a_primary_analysis, table_5b_secondary_analysis, table_5c_communication_efficiency")
        
        print("\n  Primary Analysis:")
        print(df_primary.to_string(index=False))
        print("\n  Secondary Analysis:")
        print(df_secondary.to_string(index=False))
    
    def _save_latex(self, df: pd.DataFrame, name: str):
        """Save DataFrame as LaTeX table."""
        latex_path = Config.TABLES_DIR / f'{name}.tex'
        
        # Create LaTeX with booktabs
        latex_str = df.to_latex(index=False, escape=True,
                                column_format='l' * len(df.columns))
        
        # Add booktabs commands
        latex_str = latex_str.replace('\\toprule', '\\toprule')
        latex_str = latex_str.replace('\\midrule', '\\midrule')
        latex_str = latex_str.replace('\\bottomrule', '\\bottomrule')
        
        # Use UTF-8 encoding explicitly for Windows compatibility
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)

def print_key_findings_summary(results: Dict, stats_results: Dict):
    """Print key findings summary for thesis."""
    
    print("\n" + "="*70)
    print("📊 KEY FINDINGS SUMMARY")
    print("="*70)
    
    # Performance summary per scenario
    for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
        if scenario not in results:
            continue
        
        scenario_name = Config.SCENARIO_NAMES.get(scenario, scenario)
        print(f"\n  {scenario_name}:")
        
        # Find best FL algorithm
        best_fl_algo = None
        best_fl_auc = 0
        
        for algo in ['fedavg', 'fedprox']:
            if algo in results[scenario]:
                auc_values = results[scenario][algo].get('auc_pr', [])
                if auc_values:
                    auc = np.mean(auc_values)
                    if auc > best_fl_auc:
                        best_fl_auc = auc
                        best_fl_algo = algo
        
        if best_fl_algo:
            algo_name = Config.ALGO_NAMES.get(best_fl_algo, best_fl_algo)
            print(f"    Best FL Algorithm: {algo_name}")
            print(f"    AUC-PR: {best_fl_auc:.4f}")
            
            # Compare to centralized
            if 'logistic_regression' in results[scenario]:
                cent_auc_values = results[scenario]['logistic_regression'].get('auc_pr', [])
                if cent_auc_values:
                    cent_auc = np.mean(cent_auc_values)
                    gap = (best_fl_auc - cent_auc) / cent_auc * 100 if cent_auc > 0 else 0
                    retention = (best_fl_auc / cent_auc) * 100 if cent_auc > 0 else 0
                    print(f"    vs Centralized LR: {gap:+.2f}% ({retention:.1f}% of centralized performance)")
    
    # Statistical significance summary
    print("\n  Statistical Significance (FedAvg vs FedProx):")
    print("  " + "-"*50)
    
    for scenario in ['s1_iid', 's2_noniid', 's3_quality']:
        if scenario not in stats_results:
            continue
        
        scenario_name = Config.SCENARIO_NAMES.get(scenario, scenario).split(': ')[-1]
        s = stats_results[scenario]
        
        # DeLong test
        delong = s.get('delong_fedavg_fedprox', {})
        if 'p_value' in delong:
            p = delong['p_value']
            sig = "✓ significant" if p < 0.05 else "✗ not significant"
            print(f"    {scenario_name}: p={p:.4f} ({sig})")
    
   

# MAIN EXECUTION

def main():
    """Main execution."""
    print("="*70)
    print("COMPREHENSIVE FEDERATED LEARNING ANALYSIS v3.0")
    print("="*70)
    print("Primary Metric: AUC-PR (Better for Imbalanced Data)")
    print("Features: 8 Figures, 4 Tables, Complete Statistical Analysis")
    print("="*70)
    
    # Create directories
    Config.create_directories()
    
    # Load data
    loader = DataLoader()
    results = loader.load_all_data()
    results_by_mu = loader.get_results_by_mu()
    
    if not results:
        print("\n" + "="*70)
        print("  NO RESULTS FOUND")
        print("="*70)
        print(f"\nExpected files in {Config.RESULTS_DIR}/:")
        print(f"  - centralized_results.json")
        print(f"  - results_baseline.json")
        print(f"  - results_grid_search_s3.json")
        print("\nPlease ensure these files exist and contain valid data.")
        return
    
    # Statistical analysis
    analyzer = StatisticalAnalyzer(results, results_by_mu)
    stats_results = analyzer.run_all_tests()
    
    # Generate figures
    fig_gen = FigureGenerator(results, stats_results, results_by_mu)
    fig_gen.generate_all_figures()
    
    # Generate tables
    table_gen = TableGenerator(results, stats_results, results_by_mu)
    table_gen.generate_all_tables()
    
    # Save all statistical results
    output_file = Config.RESULTS_DIR / 'comprehensive_statistical_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json.dump(convert_numpy(stats_results), f, indent=2)
    
    print(f"\n  ✓ Statistical results saved to: {output_file}")
    
    # Final summary
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    
    # Show comparison configuration
    if Config.USE_OPTIMAL_MU_COMPARISON:
        print("\n📊 COMPARISON MODE: Fair (Optimal μ)")
        print("   FedAvg: μ = 0.0 (60 runs per scenario)")
        print("   FedProx: μ = optimal per scenario (60 runs each)")
        print("     - S1 (IID): μ = 0.1 (baseline)")
        print("     - S2 (Regional): μ = 0.5 (optimal)")
        print("     - S3 (Quality): μ = 0.5 (optimal)")
    else:
        print("\n📊 COMPARISON MODE: All μ > 0 grouped as FedProx")
    
    print(f"\n📁 Output Locations:")
    print(f"   Figures: {Config.FIGURES_DIR}/")
    print(f"   Tables:  {Config.TABLES_DIR}/")
    print(f"   Stats:   {output_file}")
    
    print(f"\n📈 Summary:")
    print(f"   - {len(fig_gen.figures_created)} figures generated")
    print(f"   - {len(table_gen.tables_created)} tables generated")
    print(f"   - {len(stats_results)} scenario analyses completed")
    
    if 'ablation' in stats_results and stats_results['ablation']:
        print(f"\n🔬 Ablation Study:")
        for scenario, abl in stats_results['ablation'].items():
            if 'optimal_mu' in abl:
                print(f"   - {Config.SCENARIO_NAMES[scenario]}: Optimal μ = {abl['optimal_mu']}")
    
    # Print key findings summary
    print_key_findings_summary(results, stats_results)

    print("\n" + "="*70)
    print(" All outputs generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()