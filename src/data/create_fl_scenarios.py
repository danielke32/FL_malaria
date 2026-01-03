# 🇬🇭 STAGE 3a: Creation of Scenarios S1, S2 and S3

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
from scipy.spatial.distance import jensenshannon
from imblearn.over_sampling import SMOTENC, RandomOverSampler

# Configuration
SEED = 42
np.random.seed(SEED)

# Paths
BASE_DIR = Path(".")
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
SCENARIOS_DIR = BASE_DIR / "data" / "fl_scenarios"

# FL Configuration
N_CLIENTS = 5
TARGET_TOTAL_PER_SCENARIO = 7200
TARGET_PER_CLIENT_S1 = TARGET_TOTAL_PER_SCENARIO // N_CLIENTS  # 1,440 each

# Client labels (5 regions)
CLIENT_LABELS = {
    0: "Greater Accra",
    1: "Ashanti",
    2: "Northern",
    3: "Upper East",
    4: "Western"
}

# S2: Regional mapping (region_id → client_id)
S2_REGION_TO_CLIENT = {
    3: 0,   # Greater Accra → Client 0 (low prevalence)
    6: 1,   # Ashanti → Client 1 (medium prevalence)
    8: 2,   # Northern → Client 2 (high prevalence)
    9: 3,   # Upper East → Client 3 (medium-high )
    1: 4    # Western → Client 4 (medium-high)
}

# S2: Target sizes after SMOTE (unequal distribution)
S2_CLIENT_TARGET_SIZES = {
    0: 1800,  # Greater Accra (25%)
    1: 1440,  # Ashanti (20%)
    2: 1440,  # Northern (20%)
    3: 1260,  # Upper East (17.5%)
    4: 1260   # Western (17.5%)
}

# S3: Missingness rates per client
S3_MISSING_RATES = {
    0: 0.05,   # Greater Accra: 5%
    1: 0.10,   # Ashanti: 10%
    2: 0.20,   # Northern: 20%
    3: 0.15,   # Upper East: 15%
    4: 0.12    # Western: 12%
}

print("=" * 70)
print("FEDERATED LEARNING SCENARIO CREATION")
print("=" * 70)
print("\nOutput: Per-client CSV files for each scenario")
print("  - S1 (IID): Random equal split + local SMOTE")
print("  - S2 (Non-IID): Region-based unequal split + local SMOTE")
print("  - S3 (Quality): S2 + missing data injection")
print()


# LOAD DATA
print("-" * 70)
print("Loading training data...")
print("-" * 70)

train_raw_file = CLEANED_DIR / "train_raw.csv"
if not train_raw_file.exists():
    print(f" ERROR: File not found: {train_raw_file}")
    print("   Please ensure you have the preprocessed data.")
    exit(1)

train_raw = pd.read_csv(train_raw_file)
print(f"     Loaded {len(train_raw):,} samples")
print(f"     Prevalence: {train_raw['malaria_positive'].mean():.2%}")
print(f"     Columns: {train_raw.columns.tolist()}")

# HELPER FUNCTIONS
def apply_local_smote(X_local: pd.DataFrame, y_local: pd.Series, 
                      target_size: int, client_id: int, seed: int = 42) -> pd.DataFrame:
    
    # Identify categorical features
    categorical_features = []
    for col in ['region_id', 'season', 'age_group']:
        if col in X_local.columns:
            categorical_features.append(X_local.columns.get_loc(col))
    
    current_size = len(y_local)
    current_pos = int((y_local == 1).sum())
    current_neg = int((y_local == 0).sum())
    current_ratio = current_pos / current_size if current_size > 0 else 0.5
    
    # Calculate target counts maintaining current ratio
    target_pos = int(target_size * current_ratio)
    target_neg = target_size - target_pos
    
    try:
        X_aug = X_local.astype(float)
        y_aug = y_local.astype(int)
        
        # Oversample positive class if needed
        if target_pos > current_pos and current_pos >= 2:
            k_neighbors = min(5, current_pos - 1)
            if k_neighbors > 0:
                smote = SMOTENC(
                    categorical_features=categorical_features,
                    sampling_strategy={1: target_pos},
                    random_state=seed + client_id,
                    k_neighbors=k_neighbors
                )
                X_aug, y_aug = smote.fit_resample(X_aug, y_aug)
        
        # Oversample negative class if needed
        current_neg_after = int((y_aug == 0).sum())
        if target_neg > current_neg_after:
            ros = RandomOverSampler(
                sampling_strategy={0: target_neg}, 
                random_state=seed + client_id + 100
            )
            X_aug, y_aug = ros.fit_resample(X_aug, y_aug)
        
        # Round categorical features
        for col_idx in categorical_features:
            col_name = X_local.columns[col_idx]
            X_aug[col_name] = X_aug[col_name].round().astype(int)
        
    except Exception as e:
        print(f"          SMOTE failed: {e}")
        print(f"          Using original data")
        X_aug, y_aug = X_local.astype(float), y_local.astype(int)
    
    # Create result DataFrame
    result = pd.DataFrame(X_aug, columns=X_local.columns)
    result['malaria_positive'] = y_aug
    
    return result


def inject_missing_data(df: pd.DataFrame, missing_rate: float, seed_offset: int = 0) -> pd.DataFrame:
    """Inject missing values into feature columns (not target)."""
    df_missing = df.copy()
    
    # Features to inject missingness (exclude target and IDs)
    feature_cols = [c for c in df.columns 
                    if c not in ['malaria_positive', 'client_id', 'region_id']]
    
    if len(feature_cols) == 0:
        return df_missing
    
    n_features = len(feature_cols)
    n_samples = len(df_missing)
    total_values = n_samples * n_features
    n_to_remove = int(total_values * missing_rate)
    
    # Randomly select cells to make missing
    np.random.seed(SEED + seed_offset)
    for _ in range(n_to_remove):
        rand_row = np.random.randint(0, n_samples)
        rand_col = np.random.choice(feature_cols)
        df_missing.at[rand_row, rand_col] = np.nan
    
    return df_missing


def compute_heterogeneity_metrics(client_dfs: List[pd.DataFrame], scenario_name: str) -> Dict:
    
    # Combine all clients to get global stats
    all_data = pd.concat(client_dfs, ignore_index=True)
    global_prev = all_data['malaria_positive'].mean()
    
    client_stats = []
    for i, df in enumerate(client_dfs):
        prev = df['malaria_positive'].mean()
        n = len(df)
        client_stats.append({
            'client_id': i,
            'client_name': CLIENT_LABELS.get(i, f"Client {i}"),
            'n_samples': int(n),
            'prevalence': float(prev),
            'distribution': [float(1-prev), float(prev)]
        })
    
    # Calculate metrics
    prevalences = [s['prevalence'] for s in client_stats]
    sizes = [s['n_samples'] for s in client_stats]
    
    cv = np.std(prevalences) / np.mean(prevalences) if np.mean(prevalences) > 0 else 0
    max_ratio = max(prevalences) / min(prevalences) if min(prevalences) > 0 else float('inf')
    size_ratio = max(sizes) / min(sizes) if min(sizes) > 0 else 1.0
    
    # Jensen-Shannon divergence
    global_dist = [1-global_prev, global_prev]
    js_divs = [jensenshannon(s['distribution'], global_dist)**2 for s in client_stats]
    avg_js = np.mean(js_divs)
    
    return {
        'scenario': scenario_name,
        'n_clients': len(client_dfs),
        'total_samples': int(len(all_data)),
        'global_prevalence': float(global_prev),
        'coefficient_variation': float(cv),
        'max_prevalence_ratio': float(max_ratio),
        'size_ratio': float(size_ratio),
        'avg_js_divergence': float(avg_js),
        'client_stats': client_stats
    }


def save_client_files(client_dfs: List[pd.DataFrame], scenario_dir: Path, scenario_name: str):
    """Save client DataFrames to individual CSV files."""
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    for client_id, df in enumerate(client_dfs):
        # Remove client_id column if it exists
        if 'client_id' in df.columns:
            df = df.drop(columns=['client_id'])
        
        output_file = scenario_dir / f"client_{client_id}.csv"
        df.to_csv(output_file, index=False)
        
        prev = df['malaria_positive'].mean()
        
        # Check for missing data
        feature_cols = [c for c in df.columns if c != 'malaria_positive']
        if len(feature_cols) > 0:
            missing_pct = df[feature_cols].isna().sum().sum() / (len(df) * len(feature_cols)) * 100
        else:
            missing_pct = 0
        
        print(f"    Client {client_id}: {len(df):,} samples, "
              f"{prev:.1%} positive, {missing_pct:.1f}% missing → {output_file.name}")
    
    print(f"   Saved {len(client_dfs)} client files to {scenario_dir}")


def print_regional_prevalence_table(scenarios_data: Dict[str, List[pd.DataFrame]]):
    
    print("\n" + "=" * 90)
    print("REGIONAL MALARIA PREVALENCE ACROSS SCENARIOS")
    print("=" * 90)
    
    print(f"\n{'Region/Client':<20} {'Samples':<12} {'S1 (IID)':<15} {'S2 (Non-IID)':<15} {'S3 (Quality)':<15}")
    print("-" * 90)
    
    # Calculate prevalence for each client in each scenario
    for client_id in range(N_CLIENTS):
        region_name = CLIENT_LABELS[client_id]
        
        s1_df = scenarios_data['s1'][client_id]
        s2_df = scenarios_data['s2'][client_id]
        s3_df = scenarios_data['s3'][client_id]
        
        s1_prev = s1_df['malaria_positive'].mean()
        s2_prev = s2_df['malaria_positive'].mean()
        s3_prev = s3_df['malaria_positive'].mean()
        
        # Use S2 sample size (since S3 is same as S2)
        n_samples = len(s2_df)
        
        print(f"{region_name:<20} {n_samples:<12,} {s1_prev:<14.1%} {s2_prev:<14.1%} {s3_prev:<14.1%}")
    
    print("-" * 90)
    
    # Calculate global averages
    s1_global = pd.concat(scenarios_data['s1'])['malaria_positive'].mean()
    s2_global = pd.concat(scenarios_data['s2'])['malaria_positive'].mean()
    s3_global = pd.concat(scenarios_data['s3'])['malaria_positive'].mean()
    
    total_samples_s1 = sum(len(df) for df in scenarios_data['s1'])
    total_samples_s2 = sum(len(df) for df in scenarios_data['s2'])
    total_samples_s3 = sum(len(df) for df in scenarios_data['s3'])
    
    print(f"{'Global Average':<20} {'':<12} {s1_global:<14.1%} {s2_global:<14.1%} {s3_global:<14.1%}")
    print(f"{'Total Samples':<20} {'':<12} {total_samples_s1:<14,} {total_samples_s2:<14,} {total_samples_s3:<14,}")
    print("=" * 90)
    
    # Calculate and display heterogeneity metrics
    print("\nHETEROGENEITY METRICS:")
    print("-" * 90)
    
    for scenario_name, scenario_label in [('s1', 'S1 (IID)'), ('s2', 'S2 (Non-IID)'), ('s3', 'S3 (Quality)')]:
        dfs = scenarios_data[scenario_name]
        prevalences = [df['malaria_positive'].mean() for df in dfs]
        
        min_prev = min(prevalences)
        max_prev = max(prevalences)
        cv = np.std(prevalences) / np.mean(prevalences) if np.mean(prevalences) > 0 else 0
        prev_range = max_prev - min_prev
        
        print(f"{scenario_label:<15} Range: {min_prev:.1%} - {max_prev:.1%}  |  "
              f"CV: {cv:.4f}  |  Spread: {prev_range:.1%}")
    
    print("=" * 90)


# SCENARIO 1: IID (Equal Split)
print("\n" + "-" * 70)
print("S1: IID Baseline (Equal Random Split)")
print("-" * 70)

# Shuffle and split evenly
train_shuffled = train_raw.sample(frac=1, random_state=SEED).reset_index(drop=True)

s1_client_dfs = []
for client_id in range(N_CLIENTS):
    # Calculate slice for this client
    start_idx = client_id * len(train_shuffled) // N_CLIENTS
    end_idx = (client_id + 1) * len(train_shuffled) // N_CLIENTS if client_id < N_CLIENTS - 1 else len(train_shuffled)
    
    client_raw = train_shuffled.iloc[start_idx:end_idx].copy()
    raw_prev = client_raw['malaria_positive'].mean()
    
    print(f"  Client {client_id}: {len(client_raw):,} raw samples ({raw_prev:.1%}) → {TARGET_PER_CLIENT_S1:,} target")
    
    # Apply SMOTE
    X_client = client_raw.drop(columns=['malaria_positive'])
    y_client = client_raw['malaria_positive']
    
    client_augmented = apply_local_smote(X_client, y_client, TARGET_PER_CLIENT_S1, client_id, SEED)
    s1_client_dfs.append(client_augmented)
    
    final_prev = client_augmented['malaria_positive'].mean()
    print(f"      Result: {len(client_augmented):,} samples, {final_prev:.1%} prevalence")

total_s1 = sum(len(df) for df in s1_client_dfs)
print(f"\n  Total: {total_s1:,} samples")

# Save S1 client files
s1_dir = SCENARIOS_DIR / "s1_iid"
save_client_files(s1_client_dfs, s1_dir, "s1_iid")

# Compute heterogeneity
hetero_s1 = compute_heterogeneity_metrics(s1_client_dfs, "s1_iid")


# SCENARIO 2: NON-IID (Regional Heterogeneity)
print("\n" + "-" * 70)
print("S2: Non-IID (Regional Heterogeneity)")
print("-" * 70)

s2_client_dfs = []

for client_id in range(N_CLIENTS):
    # Get corresponding region
    region_id = [k for k, v in S2_REGION_TO_CLIENT.items() if v == client_id][0]
    region_data = train_raw[train_raw['region_id'] == region_id].copy()
    
    target_size = S2_CLIENT_TARGET_SIZES[client_id]
    
    # Calculate how much raw data we need (estimate SMOTE will ~2x the data)
    raw_needed = int(target_size / 2.0)
    available = len(region_data)
    
    # Sample with replacement if needed
    if raw_needed > available:
        print(f"    Client {client_id} ({CLIENT_LABELS[client_id]}): "
              f"Need {raw_needed}, have {available} - sampling with replacement")
        sampled_data = region_data.sample(n=raw_needed, replace=True, random_state=SEED + client_id)
    else:
        sampled_data = region_data.sample(n=min(raw_needed, available), random_state=SEED + client_id)
    
    raw_prev = sampled_data['malaria_positive'].mean()
    print(f"  Client {client_id} ({CLIENT_LABELS[client_id]:15s}, Region {region_id}): "
          f"{len(sampled_data):,} sampled ({raw_prev:.1%}) → {target_size:,} target")
    
    # Apply SMOTE
    X_client = sampled_data.drop(columns=['malaria_positive'])
    y_client = sampled_data['malaria_positive']
    
    client_augmented = apply_local_smote(X_client, y_client, target_size, client_id, SEED)
    s2_client_dfs.append(client_augmented)
    
    final_prev = client_augmented['malaria_positive'].mean()
    print(f"      Result: {len(client_augmented):,} samples, {final_prev:.1%} prevalence")

total_s2 = sum(len(df) for df in s2_client_dfs)
print(f"\n  Total: {total_s2:,} samples")

# Calculate heterogeneity
prevalences = [df['malaria_positive'].mean() for df in s2_client_dfs]
sizes = [len(df) for df in s2_client_dfs]
prev_cv = np.std(prevalences) / np.mean(prevalences) if np.mean(prevalences) > 0 else 0
size_ratio = max(sizes) / min(sizes) if min(sizes) > 0 else 1.0
print(f"  Prevalence: {min(prevalences):.1%} to {max(prevalences):.1%} (CV={prev_cv:.3f})")
print(f"  Size: {min(sizes):,} to {max(sizes):,} (ratio={size_ratio:.2f}×)")

# Save S2 client files
s2_dir = SCENARIOS_DIR / "s2_noniid"
save_client_files(s2_client_dfs, s2_dir, "s2_noniid")

# Compute heterogeneity
hetero_s2 = compute_heterogeneity_metrics(s2_client_dfs, "s2_noniid")


# SCENARIO 3: DATA QUALITY (S2 + Missingness)
print("\n" + "-" * 70)
print("S3: Data Quality (S2 + Missing Data)")
print("-" * 70)

s3_client_dfs = []

for client_id in range(N_CLIENTS):
    # Start with S2 client data
    client_df = s2_client_dfs[client_id].copy()
    missing_rate = S3_MISSING_RATES[client_id]
    
    # Inject missing values
    client_df_missing = inject_missing_data(client_df, missing_rate, seed_offset=client_id * 100)
    
    # Calculate actual missing rate
    feature_cols = [c for c in client_df_missing.columns 
                    if c not in ['malaria_positive', 'region_id']]
    actual_missing = client_df_missing[feature_cols].isna().sum().sum() / (len(client_df_missing) * len(feature_cols)) * 100
    
    print(f"  Client {client_id} ({CLIENT_LABELS[client_id]:15s}): "
          f"{missing_rate*100:.0f}% target → {actual_missing:.1f}% actual")
    
    s3_client_dfs.append(client_df_missing)

total_s3 = sum(len(df) for df in s3_client_dfs)
print(f"\n  Total: {total_s3:,} samples")

# Save S3 client files
s3_dir = SCENARIOS_DIR / "s3_quality"
save_client_files(s3_client_dfs, s3_dir, "s3_quality")

# Compute heterogeneity
hetero_s3 = compute_heterogeneity_metrics(s3_client_dfs, "s3_quality")


# PRINT REGIONAL PREVALENCE TABLE
scenarios_data = {
    's1': s1_client_dfs,
    's2': s2_client_dfs,
    's3': s3_client_dfs
}

print_regional_prevalence_table(scenarios_data)


# SAVE METRICS
heterogeneity_metrics = {
    's1_iid': hetero_s1,
    's2_noniid': hetero_s2,
    's3_quality': hetero_s3
}

# Add regional prevalence summary to metadata
regional_prevalence = {}
for scenario_name, scenario_label in [('s1_iid', 's1'), ('s2_noniid', 's2'), ('s3_quality', 's3')]:
    regional_prevalence[scenario_name] = {}
    for client_id in range(N_CLIENTS):
        region_name = CLIENT_LABELS[client_id]
        df = scenarios_data[scenario_label][client_id]
        regional_prevalence[scenario_name][region_name] = {
            'client_id': client_id,
            'n_samples': len(df),
            'prevalence': float(df['malaria_positive'].mean()),
            'n_positive': int(df['malaria_positive'].sum()),
            'n_negative': int((df['malaria_positive'] == 0).sum())
        }

metadata = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "seed": SEED,
    "n_clients": N_CLIENTS,
    "target_total_per_scenario": TARGET_TOTAL_PER_SCENARIO,
    "s2_region_mapping": S2_REGION_TO_CLIENT,
    "s2_target_sizes": S2_CLIENT_TARGET_SIZES,
    "s3_missing_rates": S3_MISSING_RATES,
    "regional_prevalence": regional_prevalence
}

output_data = {
    "metadata": metadata,
    "heterogeneity_metrics": heterogeneity_metrics
}

with open(SCENARIOS_DIR / 'heterogeneity_metrics.json', 'w') as f:
    json.dump(output_data, f, indent=2)


# SUMMARY
print("\n" + "=" * 70)
print(" SCENARIO CREATION COMPLETE")
print("=" * 70)

print("\n┌──────────────────────────────────────────────────────────┐")
print("│ VERIFICATION                                                │")
print("├──────────────────────────────────────────────────────────┤")
print(f"│ S1 Total: {total_s1:,} samples ({len(s1_client_dfs)} clients)                   │")
print(f"│ S2 Total: {total_s2:,} samples ({len(s2_client_dfs)} clients)                   │")
print(f"│ S3 Total: {total_s3:,} samples ({len(s3_client_dfs)} clients)                   │")

if abs(total_s1 - TARGET_TOTAL_PER_SCENARIO) < 100:
    print(f"│  S1 close to target ({TARGET_TOTAL_PER_SCENARIO:,})                          │")
else:
    print(f"│   S1 differs from target ({TARGET_TOTAL_PER_SCENARIO:,})                     │")

print("└──────────────────────────────────────────────────────────┘")

print("\n┌──────────────────────────────────────────────────────────┐")
print("│ HETEROGENEITY COMPARISON                                    │")
print("├──────────────────────────────────────────────────────────┤")
print(f"│ S1 (IID):     CV={hetero_s1['coefficient_variation']:.4f}, "
      f"Max/Min={hetero_s1['max_prevalence_ratio']:.2f}×  │")
print(f"│ S2 (Non-IID): CV={hetero_s2['coefficient_variation']:.4f}, "
      f"Max/Min={hetero_s2['max_prevalence_ratio']:.2f}×  │")
print(f"│ S3 (Quality): CV={hetero_s3['coefficient_variation']:.4f}, "
      f"Max/Min={hetero_s3['max_prevalence_ratio']:.2f}×  │")
print("└──────────────────────────────────────────────────────────┘")

print(f"""
 Files Saved:

data/fl_scenarios/
├── s1_iid/
│   ├── client_0.csv ({len(s1_client_dfs[0]):,} samples)
│   ├── client_1.csv ({len(s1_client_dfs[1]):,} samples)
│   ├── client_2.csv ({len(s1_client_dfs[2]):,} samples)
│   ├── client_3.csv ({len(s1_client_dfs[3]):,} samples)
│   └── client_4.csv ({len(s1_client_dfs[4]):,} samples)
│
├── s2_noniid/
│   ├── client_0.csv ({len(s2_client_dfs[0]):,} samples)
│   ├── client_1.csv ({len(s2_client_dfs[1]):,} samples)
│   ├── client_2.csv ({len(s2_client_dfs[2]):,} samples)
│   ├── client_3.csv ({len(s2_client_dfs[3]):,} samples)
│   └── client_4.csv ({len(s2_client_dfs[4]):,} samples)
│
├── s3_quality/
│   ├── client_0.csv ({len(s3_client_dfs[0]):,} samples)
│   ├── client_1.csv ({len(s3_client_dfs[1]):,} samples)
│   ├── client_2.csv ({len(s3_client_dfs[2]):,} samples)
│   ├── client_3.csv ({len(s3_client_dfs[3]):,} samples)
│   └── client_4.csv ({len(s3_client_dfs[4]):,} samples)
│
└── heterogeneity_metrics.json

 Next Stage: 3b
  python train_federated.py --experiment baseline
  python train_federated.py --experiment grid_search
""")