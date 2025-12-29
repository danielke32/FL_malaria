# 🇬🇭 STAGE 3b: Centralized Training (Logistic Regression and Random Forest)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score  # NEW: For AUC-PR
)
from pathlib import Path
import json
import time

# Configuration
class Config:
    """Configuration for centralized training."""
    BASE_DIR = Path(".")
    DATA_DIR = BASE_DIR / "data" / "cleaned"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data files
    TRAIN_FILE = DATA_DIR / "train_centralized.csv"
    VAL_FILE = DATA_DIR / "val_set.csv"
    TEST_FILE = DATA_DIR / "test_set.csv"
    
    # GridSearchCV settings
    CV_FOLDS = 5
    SCORING = 'average_precision' 
    N_JOBS = -1
    RANDOM_STATE = 42
    
    # Classification threshold (aligned with federated learning)
    CLASSIFICATION_THRESHOLD = 0.35
    
    # Hyperparameter grids
    LR_PARAM_GRID = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }
    
    RF_PARAM_GRID = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    
    # Features and target
    FEATURES = [
        'fever', 'diarrhea',
        'chills', 'sweating', 'headache', 'bodyaches', 'nausea_vomiting', 'appetite_loss',
        'bednet_use', 'recent_travel', 'season', 'age_group'
    ]
    TARGET = 'malaria_positive'


# DATA LOADING

def load_data():
    """Load and preprocess data."""
    print("\nLoading data...")
    print(f"    Using AUGMENTED training data: {Config.TRAIN_FILE}")
    print(f"      (35% positive class - matches FL local SMOTE)")
    
    # Load datasets
    train_df = pd.read_csv(Config.TRAIN_FILE)
    val_df = pd.read_csv(Config.VAL_FILE)
    test_df = pd.read_csv(Config.TEST_FILE)
    
    train_pos_ratio = train_df[Config.TARGET].mean()
    
    print(f"\n  Train: {len(train_df)} samples ({train_pos_ratio:.1%} positive)")
    print(f"  Val:   {len(val_df)} samples ({val_df[Config.TARGET].mean():.1%} positive)")
    print(f"  Test:  {len(test_df)} samples ({test_df[Config.TARGET].mean():.1%} positive)")
    
    # Extract features and labels
    X_train = np.nan_to_num(train_df[Config.FEATURES].values, nan=0)
    y_train = train_df[Config.TARGET].values
    
    X_val = np.nan_to_num(val_df[Config.FEATURES].values, nan=0)
    y_val = val_df[Config.TARGET].values
    
    X_test = np.nan_to_num(test_df[Config.FEATURES].values, nan=0)
    y_test = test_df[Config.TARGET].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# MODEL TRAINING WITH GRIDSEARCHCV
def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Logistic Regression with GridSearchCV using AUC-PR."""
    print("\n" + "="*60)
    print("Training Logistic Regression with GridSearchCV (AUC-PR)...")
    print("="*60)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    # Base estimator
    base_model = LogisticRegression(random_state=Config.RANDOM_STATE, class_weight='balanced')
    
    # GridSearchCV
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=Config.LR_PARAM_GRID,
        cv=cv,
        scoring=Config.SCORING,  
        n_jobs=Config.N_JOBS,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Best model
    best_model = grid_search.best_estimator_
    
    print(f"\n  Best Parameters: {grid_search.best_params_}")
    print(f"  Best CV Score (AUC-PR): {grid_search.best_score_:.4f}")
    print(f"  Grid Search Time: {search_time:.2f}s")
    
    # Validation predictions 
    val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_pred_class = (val_pred_proba >= Config.CLASSIFICATION_THRESHOLD).astype(int)
    
    val_auc_pr = average_precision_score(y_val, val_pred_proba)
    val_auc_roc = roc_auc_score(y_val, val_pred_proba)
    val_acc = accuracy_score(y_val, val_pred_class)
    
    print(f"  Val AUC-PR: {val_auc_pr:.4f}, AUC-ROC: {val_auc_roc:.4f}, Acc: {val_acc:.4f}")
    
    # Test predictions 
    test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred_class = (test_pred_proba >= Config.CLASSIFICATION_THRESHOLD).astype(int)
    
    # Compute metrics
    results = compute_metrics(
        y_test, test_pred_class, test_pred_proba,
        search_time, 'Logistic Regression',
        grid_search
    )
    
    return results


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "="*60)
    print("Training Random Forest with GridSearchCV (AUC-PR)...")
    print("="*60)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    # Base estimator
    base_model = RandomForestClassifier(random_state=Config.RANDOM_STATE, class_weight='balanced', n_jobs=Config.N_JOBS)
    
    # GridSearchCV
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=Config.RF_PARAM_GRID,
        cv=cv,
        scoring=Config.SCORING,  # Uses 'average_precision' (AUC-PR)
        n_jobs=Config.N_JOBS,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # Best model
    best_model = grid_search.best_estimator_
    
    print(f"\n  Best Parameters: {grid_search.best_params_}")
    print(f"  Best CV Score (AUC-PR): {grid_search.best_score_:.4f}")
    print(f"  Grid Search Time: {search_time:.2f}s")
    
    # Validation predictions (using custom threshold)
    val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_pred_class = (val_pred_proba >= Config.CLASSIFICATION_THRESHOLD).astype(int)
    
    val_auc_pr = average_precision_score(y_val, val_pred_proba)
    val_auc_roc = roc_auc_score(y_val, val_pred_proba)
    val_acc = accuracy_score(y_val, val_pred_class)
    
    print(f"  Val AUC-PR: {val_auc_pr:.4f}, AUC-ROC: {val_auc_roc:.4f}, Acc: {val_acc:.4f}")
    
    # Test predictions (using custom threshold)
    test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred_class = (test_pred_proba >= Config.CLASSIFICATION_THRESHOLD).astype(int)
    
    # Compute metrics
    results = compute_metrics(
        y_test, test_pred_class, test_pred_proba,
        search_time, 'Random Forest',
        grid_search
    )
    
    return results


# METRICS COMPUTATION
def compute_metrics(y_true, y_pred_class, y_pred_proba, train_time, model_name, grid_search=None):

    # PRIMARY METRICS 
    auc_pr = average_precision_score(y_true, y_pred_proba)  # using the auc-pr as primary metric
    auc_roc = roc_auc_score(y_true, y_pred_proba)          # secondary metrics
    

    # THRESHOLD-BASED METRICS
    acc = accuracy_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class, zero_division=0)
    recall = recall_score(y_true, y_pred_class, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_class)
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred_class, average=None, zero_division=0
    )
    
    results = {
        'model': model_name,
        
        'test_metrics': {
            # PRIMARY METRICS
            'auc_pr': float(auc_pr),    # PRIMARY - for imbalanced data
            'auc': float(auc_roc),       # SECONDARY - for comparison
            
            # Classification threshold (aligned with FL)
            'classification_threshold': Config.CLASSIFICATION_THRESHOLD,
            
            # Core metrics
            'accuracy': float(acc),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            
            # Confusion matrix
            'confusion_matrix': cm.tolist(),
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1]),
            
            # Per-class metrics
            'precision_per_class': prec_per_class.tolist(),
            'recall_per_class': rec_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support.tolist(),
            
            # Predictions for statistical tests
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred_class': y_pred_class.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'n_samples': int(len(y_true))
            },
        },
        
        'timing': {
            'train_time_sec': float(train_time)
        },
        
        # Top-level predictions for compatibility
        'test_true_labels': y_true.tolist(),
        'test_prediction_scores': y_pred_proba.tolist(),
        'test_prediction_labels': y_pred_class.tolist()
    }
    
    # Add GridSearchCV results
    if grid_search is not None:
        results['hyperparameter_tuning'] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'cv_folds': Config.CV_FOLDS,
            'scoring_metric': Config.SCORING,
            'n_candidates': len(grid_search.cv_results_['params']),
            
            # CV results summary
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': grid_search.cv_results_['mean_train_score'].tolist(),
                'std_train_score': grid_search.cv_results_['std_train_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']],
                'rank_test_score': grid_search.cv_results_['rank_test_score'].tolist()
            }
        }
    
    print(f"  Test - AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
    
    return results


# MAIN

def main():
    """Main training pipeline."""
    print("="*80)
    print("CENTRALIZED TRAINING WITH AUC-PR (MATCHES FEDERATED LEARNING)")
    print("="*80)
    print("\n  UPDATES:")
    print("    1. Uses AUC-PR as primary metric (better for imbalanced data)")
    print("    2. GridSearchCV optimizes for AUC-PR (matches FL validation)")
    print("    3. Still reports AUC-ROC for comparison with literature")
    print("    4. Uses SMOTE-augmented data (35% positive - matches FL)")
    print(f"    5. Classification threshold: {Config.CLASSIFICATION_THRESHOLD} (aligned with FL)")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Train models
    results = {}
    
    # Logistic Regression with GridSearchCV
    results['logistic_regression'] = train_logistic_regression(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Random Forest with GridSearchCV
    results['random_forest'] = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Add metadata
    results['_metadata'] = {
        'training_data': str(Config.TRAIN_FILE),
        'smote_target_positive_ratio': 0.35,
        'classification_threshold': Config.CLASSIFICATION_THRESHOLD,
        'primary_metric': 'AUC-PR (average_precision)',
        'secondary_metric': 'AUC-ROC',
        'fair_comparison_note': 'Centralized uses same 35% positive target, 0.35 threshold, and AUC-PR metric as FL'
    }
    
    # Save results
    output_file = Config.RESULTS_DIR / "centralized_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (Primary Metric: AUC-PR)")
    print("="*80)
    
    for model_key in ['logistic_regression', 'random_forest']:
        model_results = results[model_key]
        print(f"\n{model_results['model']}:")
        print(f"  Best Params: {model_results['hyperparameter_tuning']['best_params']}")
        print(f"  Best CV AUC-PR: {model_results['hyperparameter_tuning']['best_cv_score']:.4f}")
        print(f"  Test AUC-PR:    {model_results['test_metrics']['auc_pr']:.4f}")
        print(f"  Test AUC-ROC:   {model_results['test_metrics']['auc']:.4f}")
        print(f"  Test Acc:       {model_results['test_metrics']['accuracy']:.4f}")
        print(f"  Test F1:        {model_results['test_metrics']['f1']:.4f}")
    
    print("\n" + "="*80)
    print(" Fair comparison: Both centralized and FL use AUC-PR metric and 0.35 threshold")
    print("="*80)


if __name__ == "__main__":
    main()