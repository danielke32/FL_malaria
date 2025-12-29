# 🇬🇭 STAGE 2: Ghana Malaria DHS/MIS Data Preprocessing and Augmentation Pipeline

import pandas as pd
import numpy as np
import os
import warnings
import random
import json
from datetime import datetime
from typing import Dict, Tuple, List

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from scipy.special import expit
from scipy import stats

warnings.filterwarnings('ignore')

# --- Configuration ---
class Config:
    """Centralized configuration."""
    
    # Paths
    MERGED_DATA = "data/merged"
    CLEANED_DATA = "data/cleaned"
    VALIDATION_DIR = "validation/preprocessing"
    INPUT_FILE = f"{MERGED_DATA}/ghana_malaria_merged.csv"
    
    # Output files
    TRAIN_RAW_OUTPUT = f"{CLEANED_DATA}/train_raw.csv"
    TRAIN_CENTRALIZED_OUTPUT = f"{CLEANED_DATA}/train_centralized.csv"
    VAL_OUTPUT = f"{CLEANED_DATA}/val_set.csv"
    TEST_OUTPUT = f"{CLEANED_DATA}/test_set.csv"
    VALIDATION_REPORT = f"{VALIDATION_DIR}/preprocessing_report.json"
    
    # Reproducibility
    SEED = 42
    
    # Data split
    VAL_TEST_RATIO = 0.40  # 40% for val+test (real data)
    TRAIN_RATIO = 0.60     # 60% for train
    
   
    # SMOTE CONFIGURATION 
    # Both centralized and federated use the same target ratio
    SMOTE_TARGET_POSITIVE_RATIO = 0.35  # 35% positive class target
    SMOTE_K_NEIGHBORS = 5
    TARGET_TRAIN_SIZE = 7200  # Total training samples for centralized
    
    # MICE Configuration
    MICE_MAX_ITER = 50
    MICE_TOL = 1e-3
    MICE_N_IMPUTATIONS = 5


# REGION MAPPING (ALL 10 GHANA REGIONS)
REGION_TO_ID_MAPPING = {
    "Western": 1,
    "Central": 2,
    "Greater Accra": 3,
    "Volta": 4,
    "Eastern": 5,
    "Ashanti": 6,
    "Brong Ahafo": 7,
    "Northern": 8,
    "Upper East": 9,
    "Upper West": 10
}


# LITERATURE-BASED SYMPTOM PARAMETERS
SYMPTOM_LITERATURE = {
    "chills": {
        "intercept": -1.2,
        "malaria_log_or": np.log(4.5),
        "fever_log_or": np.log(3.2),
        "noise_sd": 0.4,
        "expected_prev_malaria_pos": 0.65,
        "expected_prev_malaria_neg": 0.15,
        "source": "Luxemburger et al. 1998"
    },
    "sweating": {
        "intercept": -1.5,
        "malaria_log_or": np.log(3.8),
        "fever_log_or": np.log(2.8),
        "noise_sd": 0.4,
        "expected_prev_malaria_pos": 0.55,
        "expected_prev_malaria_neg": 0.12,
        "source": "Tangpukdee et al. 2009"
    },
    "nausea_vomiting": {
        "intercept": -1.8,
        "malaria_log_or": np.log(2.5),
        "fever_log_or": np.log(1.8),
        "noise_sd": 0.5,
        "expected_prev_malaria_pos": 0.40,
        "expected_prev_malaria_neg": 0.10,
        "source": "WHO Guidelines 2023"
    },
    "appetite_loss": {
        "intercept": -1.4,
        "malaria_log_or": np.log(3.0),
        "fever_log_or": np.log(2.2),
        "noise_sd": 0.4,
        "expected_prev_malaria_pos": 0.58,
        "expected_prev_malaria_neg": 0.18,
        "source": "Asante et al. 2026"
    },
    "recent_travel": {
        "intercept": -2.5,
        "malaria_log_or": np.log(1.8),
        "fever_log_or": np.log(1.0),
        "noise_sd": 0.6,
        "expected_prev_malaria_pos": 0.20,
        "expected_prev_malaria_neg": 0.08,
        "source": "Ghana MIS 2019"
    }
}

SEVERITY_LITERATURE = {
    "headache": {
        "malaria_positive_probs": [0.08, 0.22, 0.42, 0.28],
        "malaria_negative_probs": [0.55, 0.30, 0.12, 0.03],
        "source": "WHO 2014; Koram et al. 2003"
    },
    "bodyaches": {
        "malaria_positive_probs": [0.10, 0.25, 0.40, 0.25],
        "malaria_negative_probs": [0.60, 0.28, 0.10, 0.02],
        "source": "Tangpukdee et al. 2009"
    }
}

# UTILITY FUNCTIONS
def set_all_seeds(seed: int):
    """Set all random seeds."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def calculate_odds_ratio(df: pd.DataFrame, symptom_col: str, outcome_col: str) -> Dict:
    """Calculate odds ratio with 95% CI."""
    contingency = pd.crosstab(df[symptom_col], df[outcome_col])
    
    if contingency.shape != (2, 2):
        return {"or": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    
    a, b = contingency.iloc[1, 1] + 0.5, contingency.iloc[1, 0] + 0.5
    c, d = contingency.iloc[0, 1] + 0.5, contingency.iloc[0, 0] + 0.5
    
    odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)
    
    return {"or": round(odds_ratio, 3), "ci_lower": round(ci_lower, 3), "ci_upper": round(ci_upper, 3)}


# MICE IMPUTATION
class EnhancedMICEImputer:
    """MICE imputation without using outcome variable."""
    
    def __init__(self, max_iter: int = 50, tol: float = 1e-3, n_imputations: int = 5, seed: int = 42):
        self.max_iter = max_iter
        self.tol = tol
        self.n_imputations = n_imputations
        self.seed = seed
        
    def fit_transform(self, df: pd.DataFrame, target_cols: List[str], reference_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
        print(f"   ⚙️ MICE Imputation (predictors: {reference_cols})")
        
        impute_df = df[reference_cols + target_cols].copy()
        
        # Encode categoricals
        for col in impute_df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            mask = impute_df[col].notnull()
            impute_df.loc[mask, col] = le.fit_transform(impute_df.loc[mask, col].astype(str))
            impute_df[col] = pd.to_numeric(impute_df[col], errors='coerce')
        
        # Track missing before
        missing_before = {col: int(impute_df[col].isna().sum()) for col in target_cols}
        
        # Run multiple imputations
        imputed_values = {col: [] for col in target_cols}
        
        for i in range(self.n_imputations):
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.seed + i,
                skip_complete=True
            )
            
            imputed = imputer.fit_transform(impute_df)
            imputed_df = pd.DataFrame(imputed, columns=impute_df.columns, index=impute_df.index)
            
            for col in target_cols:
                imputed_values[col].append(imputed_df[col].values)
        
        # Average imputations
        result_df = df.copy()
        for col in target_cols:
            avg_values = np.mean(imputed_values[col], axis=0)
            result_df[col] = np.round(avg_values).astype(int)
        
        # Validation stats
        validation_stats = {
            "missing_before": missing_before,
            "missing_after": {col: int(result_df[col].isna().sum()) for col in target_cols},
            "n_imputations": self.n_imputations,
            "max_iter": self.max_iter
        }
        
        for col in target_cols:
            print(f"      {col}: {missing_before[col]:,} missing → {validation_stats['missing_after'][col]} remaining")
        
        return result_df, validation_stats

# SYMPTOM SIMULATION
class LiteratureBasedSimulator:
    """Simulate symptoms using literature-derived parameters."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def simulate_binary_symptom(self, df: pd.DataFrame, symptom_name: str, params: Dict) -> pd.Series:
        """Simulate binary symptom using logistic model."""
        malaria = df['rdt_test'].values
        fever = df['fever'].fillna(0).values
        
        log_odds = (params['intercept'] + 
                   params['malaria_log_or'] * malaria + 
                   params['fever_log_or'] * fever +
                   np.random.normal(0, params['noise_sd'], len(df)))
        
        prob = expit(log_odds)
        return pd.Series((np.random.random(len(df)) < prob).astype(int), index=df.index)
    
    def simulate_severity_symptom(self, df: pd.DataFrame, symptom_name: str, params: Dict) -> pd.Series:
        """Simulate ordinal severity (0-3) based on malaria status."""
        result = np.zeros(len(df), dtype=int)
        
        for idx, row in df.iterrows():
            if row['rdt_test'] == 1:
                probs = params['malaria_positive_probs']
            else:
                probs = params['malaria_negative_probs']
            result[df.index.get_loc(idx)] = np.random.choice([0, 1, 2, 3], p=probs)
        
        return pd.Series(result, index=df.index)
    
    def simulate_all_symptoms(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Simulate all symptoms and return validation statistics."""
        result_df = df.copy()
        validation_stats = {}
        
        # Binary symptoms
        for symptom, params in SYMPTOM_LITERATURE.items():
            result_df[symptom] = self.simulate_binary_symptom(result_df, symptom, params)
            
            # Validate
            pos_prev = result_df.loc[result_df['rdt_test'] == 1, symptom].mean()
            neg_prev = result_df.loc[result_df['rdt_test'] == 0, symptom].mean()
            or_result = calculate_odds_ratio(result_df, symptom, 'rdt_test')
            
            validation_stats[symptom] = {
                "prevalence_positive": round(pos_prev, 3),
                "prevalence_negative": round(neg_prev, 3),
                "expected_positive": params['expected_prev_malaria_pos'],
                "expected_negative": params['expected_prev_malaria_neg'],
                "odds_ratio": or_result
            }
            
            print(f"   {symptom:18s}: pos={pos_prev:.1%} neg={neg_prev:.1%} OR={or_result['or']:.2f}")
        
        # Severity symptoms
        for symptom, params in SEVERITY_LITERATURE.items():
            result_df[symptom] = self.simulate_severity_symptom(result_df, symptom, params)
            
            pos_mean = result_df.loc[result_df['rdt_test'] == 1, symptom].mean()
            neg_mean = result_df.loc[result_df['rdt_test'] == 0, symptom].mean()
            
            validation_stats[symptom] = {
                "mean_severity_positive": round(pos_mean, 2),
                "mean_severity_negative": round(neg_mean, 2)
            }
            
            print(f"   {symptom:18s}: pos_mean={pos_mean:.2f} neg_mean={neg_mean:.2f}")
        
        return result_df, validation_stats

# CENTRALIZED SMOTE (FOR BASELINE MODEL) - 35% OF 7,200 SAMPLES
def apply_centralized_smote(X_train: pd.DataFrame, y_train: pd.Series, 
                            target_total_size: int = 7200,
                            target_positive_ratio: float = 0.35,
                            seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    
    print(f"\n Centralized SMOTE (for baseline global model)...")
    print(f"   Current: {len(y_train):,} samples ({y_train.mean():.1%} positive)")
    print(f"   Target: {target_total_size:,} samples with {target_positive_ratio:.0%} positive")
    
    # Define categorical features
    categorical_features = []
    for col in ['region_id', 'season', 'age_group']:
        if col in X_train.columns:
            categorical_features.append(X_train.columns.get_loc(col))
    
    # Current class distribution
    current_pos = int((y_train == 1).sum())
    current_neg = int((y_train == 0).sum())
    current_ratio = current_pos / len(y_train) if len(y_train) > 0 else 0.5
    
    # Calculate target counts for specified total size and ratio
    target_pos = int(target_total_size * target_positive_ratio)  # 7200 * 0.35 = 2520
    target_neg = target_total_size - target_pos                   # 7200 - 2520 = 4680
    
    print(f"   Target: {target_pos:,} positive + {target_neg:,} negative = {target_total_size:,} total")
    
    try:
        X_aug = X_train.astype(float)
        y_aug = y_train.astype(int)
        
        # Step 1: Oversample positive class (minority) to target
        if target_pos > current_pos and current_pos > 1:
            k_neighbors = min(Config.SMOTE_K_NEIGHBORS, current_pos - 1)
            smote = SMOTENC(
                categorical_features=categorical_features,
                sampling_strategy={1: target_pos},
                random_state=seed,
                k_neighbors=k_neighbors
            )
            X_aug, y_aug = smote.fit_resample(X_aug, y_aug)
            print(f"   SMOTE (positive): {current_pos} → {target_pos} (k={k_neighbors})")
        
        # Step 2: Oversample negative class if needed to reach target ??
        current_neg_after = int((y_aug == 0).sum())
        if target_neg > current_neg_after:
            ros = RandomOverSampler(
                sampling_strategy={0: target_neg}, 
                random_state=seed + 100
            )
            X_aug, y_aug = ros.fit_resample(X_aug, y_aug)
            print(f"   RandomOverSample (negative): {current_neg_after} → {target_neg}")
        
        # Round categorical features
        for col_idx in categorical_features:
            col_name = X_train.columns[col_idx]
            X_aug[col_name] = X_aug[col_name].round().astype(int)
        
    except Exception as e:
        print(f"   ⚠️ SMOTE failed: {e}")
        X_aug, y_aug = X_train.astype(float), y_train.astype(int)
    
    X_aug_df = pd.DataFrame(X_aug, columns=X_train.columns)
    y_aug_series = pd.Series(y_aug, name='malaria_positive')
    
    final_pos = int((y_aug_series == 1).sum())
    final_neg = int((y_aug_series == 0).sum())
    final_total = len(y_aug_series)
    final_ratio = final_pos / final_total
    
    stats_dict = {
        "target_total_size": target_total_size,
        "target_positive_ratio": target_positive_ratio,
        "before": {
            "total": len(y_train), 
            "positive": current_pos, 
            "negative": current_neg,
            "positive_ratio": round(current_ratio, 4)
        },
        "after": {
            "total": final_total, 
            "positive": final_pos, 
            "negative": final_neg,
            "positive_ratio": round(final_ratio, 4)
        },
        "synthetic_positive_added": final_pos - current_pos,
        "synthetic_negative_added": final_neg - current_neg
    }
    
    print(f"   Result: {final_total:,} samples ({final_ratio:.1%} positive)")
    print(f"   Synthetic added: +{final_pos - current_pos:,} positive, +{final_neg - current_neg:,} negative")
    
    return X_aug_df, y_aug_series, stats_dict


# MAIN PIPELINE
def main():
    print("=" * 70)
    print("🇬🇭 GHANA MALARIA PREPROCESSING (FAIR COMPARISON VERSION)")
    print("=" * 70)
    print(f"Purpose: Prepare data for Federated Learning experiments")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Random Seed: {Config.SEED}")
    print(f"\n IMPORTANT: Using {Config.SMOTE_TARGET_POSITIVE_RATIO:.0%} positive class target")
    print(f"    for BOTH centralized and federated training (fair comparison)")
    print("=" * 70)
    
    set_all_seeds(Config.SEED)
    os.makedirs(Config.CLEANED_DATA, exist_ok=True)
    os.makedirs(Config.VALIDATION_DIR, exist_ok=True)
    
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": Config.SEED,
            "smote_target_positive_ratio": Config.SMOTE_TARGET_POSITIVE_RATIO,
            "smote_k_neighbors": Config.SMOTE_K_NEIGHBORS
        }
    }
    
    # ===== LOAD DATA =====
    print("\n Loading data...")
    ghana = pd.read_csv(Config.INPUT_FILE)
    print(f"   Loaded: {len(ghana):,} records")
    
    # ===== BASIC PREPROCESSING =====
    print("\n Basic preprocessing...")
    ghana["rdt_test"] = pd.to_numeric(ghana["rdt_test"], errors="coerce").fillna(0).astype(int)
    ghana["region_id"] = ghana["region_label"].map(REGION_TO_ID_MAPPING)
    ghana = ghana[ghana["region_id"].notna()].copy()
    ghana["region_id"] = ghana["region_id"].astype(int)
    ghana["season_enc"] = LabelEncoder().fit_transform(ghana["season"].astype(str))
    
    # ===== MICE IMPUTATION =====
    print("\n MICE Imputation (NO outcome in predictors)...")
    ghana["fever"] = pd.to_numeric(ghana["fever"], errors="coerce")
    ghana["diarrhea"] = pd.to_numeric(ghana["diarrhea"], errors="coerce")
    
    mice_imputer = EnhancedMICEImputer(
        max_iter=Config.MICE_MAX_ITER,
        tol=Config.MICE_TOL,
        n_imputations=Config.MICE_N_IMPUTATIONS,
        seed=Config.SEED
    )
    
    reference_cols = ["age_months", "region_id", "season_enc"]
    ghana, mice_diagnostics = mice_imputer.fit_transform(ghana, ["fever", "diarrhea"], reference_cols)
    validation_report["mice"] = mice_diagnostics
    
    # ===== SYMPTOM SIMULATION =====
    print("\n Symptom Simulation...")
    simulator = LiteratureBasedSimulator(seed=Config.SEED)
    ghana, simulation_stats = simulator.simulate_all_symptoms(ghana)
    validation_report["simulation"] = simulation_stats
    
    # ===== FEATURE ENGINEERING =====
    print("\n Feature Engineering...")
    ghana["bednet_use"] = pd.to_numeric(ghana.get("bed_net_use_num"), errors="coerce").fillna(0).astype(int)
    ghana["residence_type"] = (ghana["residence_label"] == "Urban").astype(int)
    ghana["season_binary"] = (ghana["season"] == "Rainy").astype(int)
    age_group_map = {"<6": 0, "Below 12": 1, "Below 24": 2, "Below 36": 3, "Below 48": 4, "Below 60": 5}
    ghana["age_group_enc"] = ghana["age_group"].map(age_group_map).fillna(0).astype(int)
    
    # ===== FINAL FEATURE SELECTION =====
    final_features = [
        "survey_year", "age_months", "age_group_enc", "region_id", "residence_type",
        "season_binary", "bednet_use", "recent_travel",
        "fever", "diarrhea", "chills", "sweating", "headache", "bodyaches", 
        "nausea_vomiting", "appetite_loss", "rdt_test"
    ]
    
    for col in final_features:
        if col not in ghana.columns:
            ghana[col] = 0
    
    df_clean = ghana[final_features].copy()
    df_clean.rename(columns={"age_group_enc": "age_group", "season_binary": "season", 
                              "rdt_test": "malaria_positive"}, inplace=True)
    df_clean.fillna(0, inplace=True)
    
    print(f"   Final: {len(df_clean):,} records, {len(df_clean.columns)} features")
    print(f"   Malaria prevalence: {df_clean['malaria_positive'].mean():.2%}")
    
    # ===== TRAIN/VAL/TEST SPLIT =====
    print("\n Train/Val/Test Split (40% for Val+Test)...")
    
    X = df_clean.drop(columns=['malaria_positive'])
    y = df_clean['malaria_positive']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=Config.VAL_TEST_RATIO, stratify=y, random_state=Config.SEED
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=Config.SEED
    )
    
    print(f"\n   Train (raw):  {len(y_train):,} samples ({y_train.mean():.1%} positive)")
    print(f"   Val (real):   {len(y_val):,} samples ({y_val.mean():.1%} positive)")
    print(f"   Test (real):  {len(y_test):,} samples ({y_test.mean():.1%} positive)")
    
    validation_report["splits"] = {
        "train_raw": len(y_train),
        "val": len(y_val),
        "test": len(y_test)
    }
    
    # ===== SAVE RAW TRAINING (FOR FL SCENARIOS) =====
    print("\n💾 Saving files...")
    
    # Save RAW training (for FL scenario creation)
    train_raw = pd.concat([X_train, y_train], axis=1)
    train_raw.to_csv(Config.TRAIN_RAW_OUTPUT, index=False)
    print(f"    {Config.TRAIN_RAW_OUTPUT} ({len(train_raw):,} samples - RAW)")
    
    # Save Val/Test
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    val_df.to_csv(Config.VAL_OUTPUT, index=False)
    test_df.to_csv(Config.TEST_OUTPUT, index=False)
    print(f"    {Config.VAL_OUTPUT} ({len(val_df):,} samples)")
    print(f"    {Config.TEST_OUTPUT} ({len(test_df):,} samples)")
    
    # ===== CENTRALIZED BASELINE (GLOBAL SMOTE - 7,200 samples @ 35%) =====
    print("\n Creating Centralized Baseline (7,200 samples @ 35% positive)...")
    
    # Target: 7,200 samples total with 35% positive class
    TARGET_TRAIN_SIZE = 7200
    
    X_centralized, y_centralized, smote_stats = apply_centralized_smote(
        X_train, y_train, 
        target_total_size=TARGET_TRAIN_SIZE,
        target_positive_ratio=Config.SMOTE_TARGET_POSITIVE_RATIO,
        seed=Config.SEED
    )
    
    train_centralized = pd.concat([X_centralized, y_centralized], axis=1)
    train_centralized.to_csv(Config.TRAIN_CENTRALIZED_OUTPUT, index=False)
    print(f"    {Config.TRAIN_CENTRALIZED_OUTPUT} ({len(train_centralized):,} samples - AUGMENTED)")
    
    validation_report["centralized_smote"] = smote_stats
    
    # ===== SAVE REPORT =====
    with open(Config.VALIDATION_REPORT, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    print(f"    {Config.VALIDATION_REPORT}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print(" PREPROCESSING COMPLETE (FAIR COMPARISON VERSION)")
    print("=" * 70)
    print(f"""
Summary:
├── Raw Data Split (from {len(df_clean):,} samples):
│   ├── Train (raw):  {len(y_train):,} samples → for FL scenarios
│   ├── Val (real):   {len(y_val):,} samples → evaluation only
│   └── Test (real):  {len(y_test):,} samples → evaluation only
│
├── Centralized Baseline:
│   └── Train (augmented): {len(y_centralized):,} samples ({y_centralized.mean():.1%} positive)
│
├── SMOTE Configuration (STANDARDIZED FOR FAIR COMPARISON):
│   ├── Target total size: {Config.TARGET_TRAIN_SIZE:,} samples
│   ├── Target positive ratio: {Config.SMOTE_TARGET_POSITIVE_RATIO:.0%}
│   ├── k-neighbors: {Config.SMOTE_K_NEIGHBORS}
│   └── Both centralized and FL use identical targets
│
└── Final Dataset: {len(y_centralized) + len(y_val) + len(y_test):,} total samples
    (Train: {len(y_centralized):,} + Val: {len(y_val):,} + Test: {len(y_test):,} = 12,000)

Output Files:
├── {Config.TRAIN_RAW_OUTPUT}
│   └── Use this for: create_fl_scenarios.py
├── {Config.TRAIN_CENTRALIZED_OUTPUT}
│   └── Use this for: centralized baseline training ({len(y_centralized):,} samples @ {y_centralized.mean():.0%} pos)
├── {Config.VAL_OUTPUT}
└── {Config.TEST_OUTPUT}

Next Steps:
1. Run: python create_fl_scenarios.py (splits train_raw.csv into S1/S2/S3)
2. Train centralized baseline: python train_centralized.py
3. Train FL models: python train_federated.py --experiment baseline / grid_search
    """)

if __name__ == "__main__":
    main()