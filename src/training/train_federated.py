# 🇬🇭 STAGE 3c: Federated Training (FedAvg and FedProx)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import json
import argparse
from pathlib import Path
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, precision_recall_fscore_support,
    average_precision_score  # NEW: For AUC-PR
)
from imblearn.over_sampling import SMOTE
import itertools
from dataclasses import dataclass
import random
import os
import time


# DETERMINISTIC SEED SETUP

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# CONFIGURATION
@dataclass
class ExperimentConfig:
    """Experiment configuration with sensible defaults."""
    experiment_type: str
    mu_values: List[float]
    learning_rates: List[float]
    local_epochs: List[int]
    n_seeds: int

class Config:
    """Global configuration."""
    BASE_DIR = Path(".")
    DATA_DIR = BASE_DIR / "data"
    SCENARIOS_DIR = DATA_DIR / "fl_scenarios"
    RESULTS_DIR = BASE_DIR / "results"
    MODELS_DIR = BASE_DIR / "models"
    
    SEED = 42
    N_CLIENTS = 5
    N_ROUNDS = 10
    BATCH_SIZE = 32
    
    USE_LOCAL_SMOTE = True
    SMOTE_K_NEIGHBORS = 5
    TARGET_POSITIVE_RATIO = 0.35
    
    # Decision threshold for binary classification
    DECISION_THRESHOLD = 0.35
    
    SCENARIOS = ['s1_iid', 's2_noniid', 's3_quality']
    FEATURES = [
        'fever', 'diarrhea', 'chills', 'sweating', 
        'headache', 'bodyaches', 'nausea_vomiting', 'appetite_loss',
        'bednet_use', 'recent_travel', 'season', 'age_group'
    ]
    TARGET = 'malaria_positive'
    
    EXPERIMENTS = {
        'baseline': ExperimentConfig(
            experiment_type='baseline',
            mu_values=[0.0, 0.1],
            learning_rates=[0.05],
            local_epochs=[10],
            n_seeds=10
        ),
        'grid_search': ExperimentConfig(
            experiment_type='grid_search',
            mu_values=[0.0, 0.01, 0.1, 0.5, 1.0], 
            learning_rates=[0.01, 0.05], 
            local_epochs=[5, 10, 15], 
            n_seeds=10 
        ),
        'ablation': ExperimentConfig(
            experiment_type='ablation',
            mu_values=[0.0, 0.01, 0.1, 0.5, 1.0],
            learning_rates=[0.05], 
            local_epochs=[10], 
            n_seeds=10 
        )
    }
    
    @classmethod
    def create_directories(cls):
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# MODEL (Logistic regression as single-layer neural network)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))


# SMOTE (local smote)

def apply_local_smote(X, y, use_smote=True, target_ratio=0.35, k=5, seed=42):
    """Apply SMOTE to balance local client data."""
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    
    stats = {
        "before": {"total": len(y), "positive": n_pos, "negative": n_neg},
        "smote_applied": False
    }
    
    # Early exits
    if not use_smote:
        stats["reason"] = "disabled"
        return X, y, stats
    if n_pos < 2 or n_neg < 2:
        stats["reason"] = "insufficient_samples"
        return X, y, stats
    if n_pos / len(y) >= target_ratio:
        stats["reason"] = "already_balanced"
        return X, y, stats
    
    # Apply SMOTE
    target_pos = max(int(target_ratio * n_neg / (1 - target_ratio)), n_pos + 1)
    actual_k = min(k, n_pos - 1)
    
    if actual_k < 1:
        stats["reason"] = "k_too_small"
        return X, y, stats
    
    try:
        smote = SMOTE(sampling_strategy={1: target_pos}, random_state=seed, k_neighbors=actual_k)
        X_res, y_res = smote.fit_resample(X, y)
        stats["smote_applied"] = True
        stats["after"] = {"total": len(y_res), "positive": int(y_res.sum()), "negative": int((y_res == 0).sum())}
        stats["synthetic_added"] = stats["after"]["positive"] - n_pos
        return X_res.astype(np.float32), y_res.astype(np.float32), stats
    except Exception as e:
        stats["reason"] = f"error: {str(e)}"
        return X, y, stats

# CLIENT

class FederatedClient:
    
    def __init__(self, client_id, X_train, y_train, X_val, y_val, mu, lr, epochs, batch_size, use_smote, seed):
        self.client_id = client_id
        self.mu = mu
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        
        client_id_int = int(client_id)
        
        # Apply SMOTE
        X_train, y_train, self.smote_stats = apply_local_smote(
            X_train, y_train, use_smote, seed=seed + client_id_int
        )
        
        # Create deterministic generator
        self.generator = torch.Generator()
        self.generator.manual_seed(seed + client_id_int)
        
        # Store data
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            generator=self.generator,
            drop_last=False
        )
        
        val_dataset = TensorDataset(self.X_val, self.y_val)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = LogisticRegressionModel(X_train.shape[1], seed=seed + client_id_int)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    
    def fit(self, global_params):
        start_time = time.time()
        
        # Load global model
        self.model.load_state_dict(OrderedDict(zip(
            self.model.state_dict().keys(),
            [torch.tensor(p, dtype=torch.float32) for p in global_params]
        )))
        
        global_params_tensors = [torch.tensor(p, dtype=torch.float32) for p in global_params]
        
        self.model.train()
        
        # Track training metrics per epoch
        epoch_train_losses = []
        epoch_train_accs = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for X_batch, y_batch in self.train_loader:
    
                # Reset gradients
                self.optimizer.zero_grad()

                # Model forward pass
                outputs = self.model(X_batch)
                
                outputs = outputs.view(-1)
                y_batch = y_batch.view(-1)

                # Calculate loss 
                bce_loss = self.criterion(outputs, y_batch)
                
                # Proximal term
                prox_term = 0.0
                if self.mu > 0:
                    for param, global_param in zip(self.model.parameters(), global_params_tensors):
                        prox_term += ((param - global_param) ** 2).sum()
                    prox_term = (self.mu / 2) * prox_term
                
                loss = bce_loss + prox_term
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += bce_loss.item() * len(y_batch)
                preds = (outputs >= Config.DECISION_THRESHOLD).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
            
            # Store epoch metrics
            avg_epoch_loss = epoch_loss / total
            epoch_acc = correct / total
            epoch_train_losses.append(avg_epoch_loss)
            epoch_train_accs.append(epoch_acc)
        
        # Get final local parameters
        local_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
        
        # Compute proximal distance
        prox_distance = 0.0
        if self.mu > 0:
            for local_p, global_p in zip(local_params, global_params):
                prox_distance += np.sum((local_p - global_p) ** 2)
            prox_distance = np.sqrt(prox_distance)
        
        elapsed_time = time.time() - start_time
        
        metrics = {
            'prox': float(prox_distance),
            'train_time_sec': elapsed_time,
            'final_train_loss': epoch_train_losses[-1],
            'final_train_acc': epoch_train_accs[-1],
            'epoch_losses': epoch_train_losses,
            'epoch_accs': epoch_train_accs
        }
        
        return local_params, len(self.y_train), metrics
    
    def evaluate(self, global_params):
        # Load global model
        self.model.load_state_dict(OrderedDict(zip(
            self.model.state_dict().keys(),
            [torch.tensor(p, dtype=torch.float32) for p in global_params]
        )))
        
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * len(y_batch)
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / len(all_labels)
        
        # METRICS (Primary for model selection)
        
        # AUC-ROC: Standard metric
        if len(np.unique(all_labels)) > 1:
            auc_roc = roc_auc_score(all_labels, all_probs)
        else:
            auc_roc = 0.5  
        
        # AUC-PR: Better for imbalanced data (focuses on positive class)
        if len(np.unique(all_labels)) > 1:
            auc_pr = average_precision_score(all_labels, all_probs)
        else:
            auc_pr = all_labels.mean()  # Baseline for single class
        
        # METRICS (Secondary for interpretation)
        
        # Use fixed threshold consistently
        preds = (all_probs >= Config.DECISION_THRESHOLD).astype(float)
        accuracy = (preds == all_labels).mean()
        
        # Calculate F1, precision, recall at fixed threshold
        if preds.sum() > 0:
            f1 = f1_score(all_labels, preds, zero_division=0)
            precision = precision_score(all_labels, preds, zero_division=0)
            recall = recall_score(all_labels, preds, zero_division=0)
        else:
            # No positive predictions - metrics are 0
            f1 = 0.0
            precision = 0.0
            recall = 0.0
      
        # DIAGNOSTIC INFO
        pos_ratio = all_labels.mean()
        pred_ratio = preds.mean()
        
        metrics = {
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            
            # Primary metrics 
            'auc': float(auc_roc),        
            'auc_pr': float(auc_pr),      
            
            # Secondary metrics 
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            
            # Diagnostics
            'n_samples': int(len(all_labels)),
            'n_positive': int(all_labels.sum()),
            'positive_ratio': float(pos_ratio),
            'prediction_ratio': float(pred_ratio),
            'n_positive_preds': int(preds.sum()),
            
            # Model calibration
            'mean_prob_pos_class': float(all_probs[all_labels == 1].mean()) if all_labels.sum() > 0 else 0.0,
            'mean_prob_neg_class': float(all_probs[all_labels == 0].mean()) if (all_labels == 0).sum() > 0 else 0.0,
        }
        
        return avg_loss, accuracy, metrics

# FEDERATED TRAINING
def train_federated_model(scenario: str, mu: float, lr: float, epochs: int, seed: int) -> Dict:
    set_all_seeds(seed)
    
    # Load scenario data
    scenario_dir = Config.SCENARIOS_DIR / scenario
    if not scenario_dir.exists():
        print(f"\n  ERROR: Scenario directory not found: {scenario_dir}")
        return {
            'error': f'Scenario directory not found: {scenario_dir}',
            'test_metrics': {'auc': 0.0, 'auc_pr': 0.0, 'accuracy': 0.0, 'f1': 0.0},
            'timing': {'total_train_time_sec': 0.0, 'avg_round_time_sec': 0.0, 'inference_time_ms_mean': 0.0},
            'history': {},
            'seed': seed, 'mu': mu, 'lr': lr, 'epochs': epochs
        }
    
    # Load test set and scaler
    test_df = pd.read_csv(Config.DATA_DIR / "cleaned" / "test_set.csv")
    scaler = StandardScaler()
    
    # Initialize clients
    clients = []
    client_files = sorted(scenario_dir.glob("client_*.csv"))
    
    for i, cf in enumerate(client_files):
        cid = int(cf.stem.split('_')[1])
        df = pd.read_csv(cf)
        
        if len(df) == 0:
            continue
        
        #  1: SHUFFLE BEFORE SPLIT (deterministic)
        df_shuffled = df.sample(frac=1, random_state=seed + cid).reset_index(drop=True)
        
        
        #  2: ADAPTIVE SPLIT RATIO based on dataset size
        if len(df_shuffled) < 50:
            split_ratio = 0.8
            min_val_samples = 5
        else:
            split_ratio = 0.7
            min_val_samples = 10
        
        n_train = int(split_ratio * len(df_shuffled))
        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train:].copy()
        
        # 3: CHECK CLASS BALANCE IN VALIDATION SET
        val_pos = val_df[Config.TARGET].sum()
        val_neg = len(val_df) - val_pos
        
        # Skip client if validation set doesn't have BOTH classes
        if len(train_df) < 10 or len(val_df) < min_val_samples or val_pos == 0 or val_neg == 0:
            continue
        
        # Prepare features
        X_train = np.nan_to_num(train_df[Config.FEATURES].values, nan=0)
        y_train = train_df[Config.TARGET].values
        X_val = np.nan_to_num(val_df[Config.FEATURES].values, nan=0)
        y_val = val_df[Config.TARGET].values
        
        # Fit scaler on first client's training data
        if i == 0:
            scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Create client
        clients.append(FederatedClient(
            cid, X_train, y_train, X_val, y_val,
            mu, lr, epochs, Config.BATCH_SIZE, Config.USE_LOCAL_SMOTE, seed
        ))
    
    if not clients:
        print(f"\n  WARNING: No valid clients created for scenario '{scenario}'")
        return {
            'test_metrics': {'auc': 0.5, 'auc_pr': 0.5, 'accuracy': 0.5, 'f1': 0.5},
            'timing': {'total_train_time_sec': 0.0, 'avg_round_time_sec': 0.0, 'inference_time_ms_mean': 0.0},
            'history': {},
            'seed': seed, 'mu': mu, 'lr': lr, 'epochs': epochs
        }
    
    # Initialize global model
    input_size = clients[0].model.linear.in_features
    global_model = LogisticRegressionModel(input_size, seed=seed)
    global_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
    
    # Training history - EXPANDED with AUC-PR
    history = {
        'rounds': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],         # AUC-ROC (SECONDARY metric)
        'val_auc_pr': [],      # AUC-PR (PRIMARY metric)
        'val_f1': [],
        'val_precision': [],   
        'val_recall': [],      
        'avg_prox': [],
        'round_time_sec': []
    }
    
    # Federated training loop
    for r in range(Config.N_ROUNDS):
        round_start = time.time()
        
        params_list, samples_list, metrics_list = [], [], []
        
        # Client updates
        for c in clients:
            p, n, m = c.fit(global_params)
            params_list.append(p)
            samples_list.append(n)
            metrics_list.append(m)
        
        # FedAvg aggregation
        total_samples = sum(samples_list)
        global_params = [
            sum(params_list[i][j] * (samples_list[i] / total_samples) for i in range(len(clients)))
            for j in range(len(params_list[0]))
        ]
        
        # Aggregate training metrics
        avg_train_loss = np.mean([m['final_train_loss'] for m in metrics_list])
        avg_train_acc = np.mean([m['final_train_acc'] for m in metrics_list])
        avg_prox = np.mean([m['prox'] for m in metrics_list])
        
      
        # 4: WEIGHTED VALIDATION METRICS (by validation set size)
        val_metrics_list = []
        for c in clients:
            loss, acc, m = c.evaluate(global_params)
            val_metrics_list.append(m)
        
        # Weighted average by validation set size
        total_val_samples = sum(m['n_samples'] for m in val_metrics_list)
        
        def weighted_avg(metric_name):
            return sum(m[metric_name] * m['n_samples'] for m in val_metrics_list) / total_val_samples
        
        round_time = time.time() - round_start
        
        # Record all metrics
        history['rounds'].append(r + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['train_acc'].append(float(avg_train_acc))
        history['val_loss'].append(float(weighted_avg('loss')))
        history['val_acc'].append(float(weighted_avg('accuracy')))
        history['val_auc'].append(float(weighted_avg('auc')))
        history['val_auc_pr'].append(float(weighted_avg('auc_pr')))  # PRIMARY!
        history['val_f1'].append(float(weighted_avg('f1')))
        history['val_precision'].append(float(weighted_avg('precision')))
        history['val_recall'].append(float(weighted_avg('recall')))
        history['avg_prox'].append(float(avg_prox))
        history['round_time_sec'].append(float(round_time))
        

        # 5: PRINT VALIDATION METRICS (AUC-PR as primary)
        if r % 2 == 0 or r == Config.N_ROUNDS - 1:
            print(f"    Round {r+1}/{Config.N_ROUNDS}: "
                  f"Val AUC-PR={weighted_avg('auc_pr'):.4f}, "
                  f"AUC-ROC={weighted_avg('auc'):.4f}, "
                  f"F1={weighted_avg('f1'):.4f}")
    
    # Test evaluation with COMPREHENSIVE metrics
    device = next(iter(global_model.parameters())).device
    global_model.load_state_dict(OrderedDict(zip(
        global_model.state_dict().keys(),
        [torch.tensor(p).float().to(device) for p in global_params]
    )))
    
    X_test = np.nan_to_num(test_df[Config.FEATURES].values, nan=0)
    X_test = scaler.transform(X_test)
    y_test = test_df[Config.TARGET].values
    
    global_model.eval()
    
    # Measure inference time
    inference_times = []
    with torch.no_grad():
        _ = global_model(torch.FloatTensor(X_test[:10]).to(device))
        
        n_inference_samples = min(1000, len(X_test))
        for i in range(n_inference_samples):
            start = time.time()
            _ = global_model(torch.FloatTensor(X_test[i:i+1]).to(device))
            inference_times.append((time.time() - start) * 1000)
        
        y_pred_proba = global_model(torch.FloatTensor(X_test).to(device)).cpu().numpy().flatten()
    
    y_pred_class = (y_pred_proba >= Config.DECISION_THRESHOLD).astype(int)
    
    # Core metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    auc_pr = average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else y_test.mean()
    acc = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, zero_division=0)
    recall = recall_score(y_test, y_pred_class, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_class)
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred_class, average=None, zero_division=0
    )
    
    total_train_time = sum(history['round_time_sec'])
    
    return {
        'history': history,
        'test_metrics': {
            # Core metrics
            'auc': float(auc_roc),           # AUC-ROC
            'auc_pr': float(auc_pr),         # AUC-PR (PRIMARY)
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
                'y_true': y_test.tolist(),
                'y_pred_class': y_pred_class.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'n_samples': int(len(y_test))
            },
        },
        
        # Top-level predictions (for compatibility)
        'test_true_labels': y_test.tolist(),
        'test_prediction_scores': y_pred_proba.tolist(),
        'test_prediction_labels': y_pred_class.tolist(),
        'regional_confusion_matrices': [],
        
        # Timing metrics
        'timing': {
            'total_train_time_sec': float(total_train_time),
            'avg_round_time_sec': float(np.mean(history['round_time_sec'])),
            'inference_time_ms_mean': float(np.mean(inference_times)),
            'inference_time_ms_std': float(np.std(inference_times))
        },
        
        'seed': seed,
        'mu': mu,
        'lr': lr,
        'epochs': epochs,
        'smote_summary': [{"client_id": c.client_id, **c.smote_stats} for c in clients],
        'n_clients_used': len(clients)
    }

# MAIN

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Experiments - AUC-PR Version')
    parser.add_argument('--experiment', type=str, default='baseline',
                       choices=['baseline', 'grid_search', 'ablation'],
                       help='Experiment type to run')
    args = parser.parse_args()
    
    set_all_seeds(Config.SEED)
    Config.create_directories()
    exp_config = Config.EXPERIMENTS[args.experiment]
    
    # Print header
    print("=" * 80)
    print(f"FEDERATED LEARNING EXPERIMENT: {args.experiment.upper()} [AUC-PR VERSION]")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  μ values: {exp_config.mu_values}")
    print(f"  Learning rates: {exp_config.learning_rates}")
    print(f"  Local epochs: {exp_config.local_epochs}")
    print(f"  Seeds: {exp_config.n_seeds}")
    print(f"  Rounds: {Config.N_ROUNDS}")
    print(f"  Decision threshold: {Config.DECISION_THRESHOLD}")
    print(f"  Local SMOTE: {Config.USE_LOCAL_SMOTE}")
    print(f"  Deterministic: ✓ (all seeds set)")
    print(f"\nEnhancements v2.0:")
    print(f"  ✓ AUC-PR as primary metric (imbalance-aware)")
    print(f"  ✓ Threshold-free validation")
    print(f"  ✓ Per-round training dynamics")
    print(f"  ✓ Confusion matrices")
    print(f"  ✓ Per-class metrics")
    print(f"  ✓ Comprehensive timing")
    print(f"  ✓ Fixed 0.5 AUC issue")
    print("=" * 80)
    
    # Generate all hyperparameter combinations
    combos = list(itertools.product(
        Config.SCENARIOS,
        exp_config.mu_values,
        exp_config.learning_rates,
        exp_config.local_epochs
    ))
    
    total_runs = len(combos) * exp_config.n_seeds
    print(f"\nTotal runs: {total_runs}")
    print(f"  {len(Config.SCENARIOS)} scenarios × {len(exp_config.mu_values)} μ × "
          f"{len(exp_config.learning_rates)} LR × {len(exp_config.local_epochs)} epochs × "
          f"{exp_config.n_seeds} seeds\n")
    
    # Run all experiments
    results = {}
    counter = 0
    
    for scenario, mu, lr, epochs in combos:
        key = f"{scenario}_mu{mu}_lr{lr}_ep{epochs}"
        results[key] = []
        
        algo_name = "FedAvg" if mu == 0 else f"FedProx(μ={mu})"
        print(f"\n{scenario.upper()} | {algo_name} | LR={lr} | Epochs={epochs}")
        
        for i in range(exp_config.n_seeds):
            counter += 1
            seed = Config.SEED + i
            print(f"  [{counter}/{total_runs}] Seed {seed}...", end=" ", flush=True)
            
            result = train_federated_model(scenario, mu, lr, epochs, seed)
            results[key].append(result)
            
            # Check if training was successful
            if 'error' in result:
                print(f"ERROR: {result['error']}")
            elif 'test_metrics' in result:
                print(f"AUC-PR: {result['test_metrics']['auc_pr']:.4f}, "
                      f"AUC-ROC: {result['test_metrics']['auc']:.4f}")
            else:
                print(f"WARNING: Unexpected result format")
        
        # Summary for this configuration
        successful_runs = [r for r in results[key] if 'test_metrics' in r and 'timing' in r and 'error' not in r]
        
        if successful_runs:
            auc_prs = [r['test_metrics']['auc_pr'] for r in successful_runs]
            auc_rocs = [r['test_metrics']['auc'] for r in successful_runs]
            f1s = [r['test_metrics']['f1'] for r in successful_runs]
            times = [r['timing']['total_train_time_sec'] for r in successful_runs]
            print(f"  → Summary ({len(successful_runs)}/{exp_config.n_seeds} successful): "
                  f"AUC-PR={np.mean(auc_prs):.4f}±{np.std(auc_prs):.4f}, "
                  f"AUC-ROC={np.mean(auc_rocs):.4f}±{np.std(auc_rocs):.4f}, "
                  f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}, "
                  f"Time={np.mean(times):.1f}±{np.std(times):.1f}s")
        else:
            print(f"  → ERROR: No successful runs for this configuration!")
    
    # Save results
    output_file = Config.RESULTS_DIR / f"results_{args.experiment}.json"
    
    def json_serializer(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=json_serializer)
    
    print(f"\n{'='*80}")
    print(f" Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print comparative summary
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY (PRIMARY METRIC: AUC-PR)")
    print("=" * 80)
    
    for scenario in Config.SCENARIOS:
        print(f"\n{scenario.upper()}:")
        
        for lr in exp_config.learning_rates:
            for epochs in exp_config.local_epochs:
                if len(exp_config.learning_rates) > 1 or len(exp_config.local_epochs) > 1:
                    print(f"\n  LR={lr}, Epochs={epochs}:")
                
                for mu in exp_config.mu_values:
                    key = f"{scenario}_mu{mu}_lr{lr}_ep{epochs}"
                    if key in results:
                        successful_runs = [r for r in results[key] if 'test_metrics' in r and 'error' not in r]
                        if successful_runs:
                            auc_prs = [r['test_metrics']['auc_pr'] for r in successful_runs]
                            auc_rocs = [r['test_metrics']['auc'] for r in successful_runs]
                            algo = "FedAvg" if mu == 0 else f"FedProx(μ={mu})"
                            indent = "    " if len(exp_config.learning_rates) > 1 or len(exp_config.local_epochs) > 1 else "  "
                            print(f"{indent}{algo:16s} | AUC-PR={np.mean(auc_prs):.4f}±{np.std(auc_prs):.4f} | "
                                  f"AUC-ROC={np.mean(auc_rocs):.4f}±{np.std(auc_rocs):.4f}")
                
                # Show FedProx improvement
                if 0.0 in exp_config.mu_values and any(m > 0 for m in exp_config.mu_values):
                    fedavg_key = f"{scenario}_mu0.0_lr{lr}_ep{epochs}"
                    fedprox_keys = [f"{scenario}_mu{m}_lr{lr}_ep{epochs}" 
                                   for m in exp_config.mu_values if m > 0]
                    
                    if fedavg_key in results and fedprox_keys:
                        fedavg_runs = [r for r in results[fedavg_key] if 'test_metrics' in r and 'error' not in r]
                        if fedavg_runs:
                            fedavg_auc_prs = [r['test_metrics']['auc_pr'] for r in fedavg_runs]
                            for pk in fedprox_keys:
                                if pk in results:
                                    fedprox_runs = [r for r in results[pk] if 'test_metrics' in r and 'error' not in r]
                                    if fedprox_runs:
                                        fedprox_auc_prs = [r['test_metrics']['auc_pr'] for r in fedprox_runs]
                                        improvement = np.mean(fedprox_auc_prs) - np.mean(fedavg_auc_prs)
                                        mu_val = float(pk.split('_mu')[1].split('_')[0])
                                        indent = "    " if len(exp_config.learning_rates) > 1 or len(exp_config.local_epochs) > 1 else "  "
                                        print(f"{indent}→ FedProx(μ={mu_val}) improvement: {improvement:+.4f} AUC-PR")
    
    print("\n" + "=" * 80)
    print("Note: AUC-PR is the primary metric for imbalanced data.")
    print("Higher AUC-PR = better performance on positive class.")
    print("=" * 80)

if __name__ == "__main__":
    main()