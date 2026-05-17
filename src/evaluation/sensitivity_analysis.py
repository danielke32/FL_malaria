# =============================================================================
# SENSITIVITY ANALYSIS (PIPELINE-CONSISTENT VERSION)
# -----------------------------------------------------------------------------
# Purpose:
#   Evaluate whether the six synthetic symptom variables introduce circularity
#   by rerunning the EXACT federated learning pipeline used in
#   train_federated.py while changing ONLY the feature set.
#
# IMPORTANT:
#   This script IMPORTS and REUSES the original federated AND centralized
#   training functions to ensure full methodological consistency.
#
# Key Principle:
#   EVERYTHING remains identical to the original experiments:
#       ✓ same SMOTE
#       ✓ same scaler logic (fitted via load_data())
#       ✓ same client splitting
#       ✓ same FedAvg/FedProx implementation
#       ✓ same seeds (42–51, n=10)
#       ✓ same thresholds (0.35)
#       ✓ same training rounds (10)
#       ✓ same validation strategy
#       ✓ same LR hyperparameter search (GridSearchCV, 5-fold, AUC-PR)
#       ✓ LR only for centralized baseline (no RF)
#
#   ONLY the feature set changes between Native-5 and Full-12 runs.
#
# Full-12 results MUST reproduce the published centralized and federated
# numbers exactly — this validates the pipeline and confirms the only
# variable is the feature set.
# =============================================================================

import json
import numpy as np
from pathlib import Path
from copy import deepcopy
import sys

# =============================================================================
# PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# =============================================================================
# IMPORT ORIGINAL PIPELINES
# =============================================================================

from src.training.train_federated import (
    Config,
    train_federated_model,
    set_all_seeds
)

import src.training.train_centralized as cent_module

# =============================================================================
# FEATURE SETS
# =============================================================================

NATIVE_FEATURES = [
    'fever',
    'diarrhea',
    'bednet_use',
    'season',
    'age_group'
]

SYNTHETIC_FEATURES = [
    'chills',
    'sweating',
    'headache',
    'bodyaches',
    'nausea_vomiting',
    'appetite_loss',
    'recent_travel'
]

FULL_FEATURES = NATIVE_FEATURES + SYNTHETIC_FEATURES

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = Config.SCENARIOS

# Matches Config.EXPERIMENTS['baseline'] → n_seeds=10, lr=0.05, epochs=10
SEEDS = list(range(Config.SEED, Config.SEED + 10))

MU_VALUES = {
    "FedAvg": 0.0,
    "FedProx": 0.5
}

# =============================================================================
# CENTRALIZED BASELINE
# -----------------------------------------------------------------------------
# Uses train_centralized.py's OWN load_data() and train_logistic_regression()
# verbatim — GridSearchCV over C, 5-fold StratifiedKFold, AUC-PR scoring,
# StandardScaler, class_weight='balanced', threshold=0.35.
# LR only — no RF (matches manuscript).
# Config.FEATURES is temporarily replaced so load_data() selects the correct
# columns; it is restored in the finally block regardless of outcome.
# =============================================================================

def run_centralized(feature_set_name, feature_cols):
    """
    Run the centralized LR baseline using the IDENTICAL pipeline from
    train_centralized.py. Only the feature set is changed.

    Full-12 output must match the published centralized LR result exactly.
    """

    # Swap feature list on the centralized Config
    original_features = deepcopy(cent_module.Config.FEATURES)
    cent_module.Config.FEATURES = feature_cols

    try:
        # load_data() reads train_centralized.csv, val_set.csv, test_set.csv,
        # applies np.nan_to_num, and fits StandardScaler on X_train only —
        # exactly as in the original centralized training run.
        X_train, y_train, X_val, y_val, X_test, y_test = cent_module.load_data()

        # train_logistic_regression() runs GridSearchCV (C in [0.001..100],
        # l2, lbfgs, 5-fold, scoring='average_precision') then evaluates on
        # X_test with threshold=0.35 — identical to the published baseline.
        results = cent_module.train_logistic_regression(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

    finally:
        # Always restore the original feature list
        cent_module.Config.FEATURES = original_features

    auc_pr  = results['test_metrics']['auc_pr']
    auc_roc = results['test_metrics']['auc']

    print(
        f"  [{feature_set_name}] "
        f"AUC-PR={auc_pr:.4f} | "
        f"AUC-ROC={auc_roc:.4f}"
    )

    return {
        "auc_pr":  float(auc_pr),
        "auc_roc": float(auc_roc)
    }

# =============================================================================
# FEDERATED RUNNER
# -----------------------------------------------------------------------------
# Uses train_federated.py's OWN train_federated_model() verbatim.
# lr=0.05, epochs=10 match Config.EXPERIMENTS['baseline'].
# 10 seeds (42–51) match n_seeds=10 in the baseline experiment config.
# mu=0.0 → FedAvg, mu=0.5 → FedProx (matches published grid search best).
# Config.FEATURES is temporarily replaced so data loading inside
# train_federated_model() selects the correct columns.
# =============================================================================

def run_federated_experiment(feature_name, feature_cols):
    """
    Run federated experiments using the IDENTICAL pipeline from
    train_federated.py. Only the feature set is changed.

    Full-12 mean±SD must reproduce the published federated results.
    """

    print("\n" + "=" * 70)
    print(f"{feature_name.upper()} FEATURE SET")
    print("=" * 70)

    # Swap feature list on the federated Config
    original_features = deepcopy(Config.FEATURES)
    Config.FEATURES = feature_cols

    results = {}

    try:

        for scenario in SCENARIOS:

            print(f"\n{scenario.upper()}")
            results[scenario] = {}

            for algo_name, mu in MU_VALUES.items():

                print(f"\n  {algo_name}  (mu={mu})")

                auc_prs  = []
                auc_rocs = []
                f1s      = []

                for seed in SEEDS:

                    set_all_seeds(seed)

                    result = train_federated_model(
                        scenario=scenario,
                        mu=mu,
                        lr=0.05,      # matches Config.EXPERIMENTS['baseline']
                        epochs=10,    # matches Config.EXPERIMENTS['baseline']
                        seed=seed
                    )

                    auc_pr  = result['test_metrics']['auc_pr']
                    auc_roc = result['test_metrics']['auc']
                    f1      = result['test_metrics']['f1']

                    auc_prs.append(auc_pr)
                    auc_rocs.append(auc_roc)
                    f1s.append(f1)

                    print(
                        f"    Seed {seed} | "
                        f"AUC-PR={auc_pr:.4f} | "
                        f"AUC-ROC={auc_roc:.4f}"
                    )

                summary = {
                    "auc_pr_mean":  float(np.mean(auc_prs)),
                    "auc_pr_sd":    float(np.std(auc_prs)),
                    "auc_roc_mean": float(np.mean(auc_rocs)),
                    "auc_roc_sd":   float(np.std(auc_rocs)),
                    "f1_mean":      float(np.mean(f1s)),
                    "f1_sd":        float(np.std(f1s))
                }

                results[scenario][algo_name] = summary

                print(
                    f"\n    SUMMARY | "
                    f"AUC-PR={summary['auc_pr_mean']:.4f}±{summary['auc_pr_sd']:.4f} | "
                    f"AUC-ROC={summary['auc_roc_mean']:.4f}±{summary['auc_roc_sd']:.4f}"
                )

    finally:
        # Always restore the original feature list
        Config.FEATURES = original_features

    return results

# =============================================================================
# SUMMARY TABLES
# =============================================================================

def print_summary(cent_native, cent_full, native_results, full_results):

    for metric_key, metric_label, sd_key in [
        ("auc_pr_mean",  "AUC-PR",  "auc_pr_sd"),
        ("auc_roc_mean", "AUC-ROC", "auc_roc_sd"),
    ]:
        print("\n" + "=" * 70)
        print(f"SENSITIVITY RESULTS — {metric_label} (mean ± SD)")
        print("=" * 70)

        header = (
            f"{'Scenario':<28}"
            f"{'Algorithm':<12}"
            f"{'Native-5':>18}"
            f"{'Full-12':>18}"
            f"{'Δ ' + metric_label:>14}"
        )
        print(header)
        print("-" * 70)

        delta_values = []

        for scenario in SCENARIOS:
            for algo in MU_VALUES.keys():

                nat = native_results[scenario][algo]
                ful = full_results[scenario][algo]

                delta = ful[metric_key] - nat[metric_key]
                delta_values.append(delta)

                nat_str = f"{nat[metric_key]:.4f}±{nat[sd_key]:.4f}"
                ful_str = f"{ful[metric_key]:.4f}±{ful[sd_key]:.4f}"

                print(
                    f"{scenario:<28}"
                    f"{algo:<12}"
                    f"{nat_str:>18}"
                    f"{ful_str:>18}"
                    f"{delta:>+14.4f}"
                )

        # Centralized LR row — single value, no SD
        cent_delta = cent_full[metric_key.replace("_mean", "")] \
                   - cent_native[metric_key.replace("_mean", "")]
        cent_nat_str = f"{cent_native[metric_key.replace('_mean', '')]:.4f}"
        cent_ful_str = f"{cent_full[metric_key.replace('_mean', '')]:.4f}"

        print("-" * 70)
        print(
            f"{'Centralized LR':<28}"
            f"{'—':<12}"
            f"{cent_nat_str:>18}"
            f"{cent_ful_str:>18}"
            f"{cent_delta:>+14.4f}"
        )
        print("=" * 70)

        mean_d = np.mean(delta_values)
        max_d  = np.max(delta_values)

        print(f"{metric_label} — Mean gain from synthetic features: {mean_d:+.4f} | Max: {max_d:+.4f}")

        if max_d < 0.05:
            print(
                f"Interpretation: Max Δ {metric_label} < 0.05 — "
                f"synthetic features add minimal signal; "
                f"no evidence of circularity on this metric."
            )
        else:
            print(
                f"Interpretation: Max Δ {metric_label} = {max_d:+.4f} — "
                f"synthetic features contribute meaningful signal. "
                f"Additional justification for simulation methodology is recommended."
            )

# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("Native DHS Features (5) vs Full Feature Set (12)")
    print("Primary metric: AUC-PR | Secondary metric: AUC-ROC")
    print("=" * 70)

    print(f"\nNative features: {NATIVE_FEATURES}")
    print(f"Full features:   {FULL_FEATURES}")
    print(
        f"Seeds: {len(SEEDS)} ({SEEDS[0]}–{SEEDS[-1]}) | "
        f"Rounds: {Config.N_ROUNDS} | "
        f"FedProx μ=0.5"
    )

    # =========================================================================
    # CENTRALIZED BASELINE
    # Runs train_centralized.py's GridSearchCV LR pipeline — LR only, no RF.
    # Full-12 result must match published centralized LR AUC-PR/AUC-ROC.
    # =========================================================================

    print("\n--- Centralized Baseline ---")

    cent_native = run_centralized("Native-only", NATIVE_FEATURES)
    cent_full   = run_centralized("Full feature", FULL_FEATURES)

    # =========================================================================
    # FEDERATED EXPERIMENTS
    # Runs train_federated.py's pipeline for both feature sets.
    # Full-12 mean±SD must match published federated results.
    # =========================================================================

    native_results = run_federated_experiment("Native", NATIVE_FEATURES)
    full_results   = run_federated_experiment("Full",   FULL_FEATURES)

    # =========================================================================
    # SUMMARY TABLES
    # =========================================================================

    print_summary(cent_native, cent_full, native_results, full_results)

    # =========================================================================
    # SAVE
    # Build the JSON shape expected by SensitivityAnalysisSection in
    # final_analysis_c_sensitivity.py, which requires:
    #   - "config"            : n_seeds, n_rounds, fedprox_mu
    #   - "centralized"       : native{}, full{}, delta_auc_pr, delta_auc_roc
    #   - "summary_rows"      : one dict per (scenario × algorithm)
    #   - "mean/max_delta_*"  : aggregates over all rows
    #   - "interpretation"    : human-readable verdict
    #   - raw federated dicts kept for traceability
    # =========================================================================

    # Build summary_rows — one entry per (scenario, algorithm)
    summary_rows = []
    for scenario in SCENARIOS:
        for algo_name in MU_VALUES.keys():
            nat = native_results[scenario][algo_name]
            ful = full_results[scenario][algo_name]
            summary_rows.append({
                "scenario":          scenario,
                "algorithm":         algo_name,
                # AUC-PR
                "native_auc_pr":     nat["auc_pr_mean"],
                "native_auc_pr_sd":  nat["auc_pr_sd"],
                "full_auc_pr":       ful["auc_pr_mean"],
                "full_auc_pr_sd":    ful["auc_pr_sd"],
                "delta_auc_pr":      ful["auc_pr_mean"] - nat["auc_pr_mean"],
                # AUC-ROC
                "native_auc_roc":    nat["auc_roc_mean"],
                "native_auc_roc_sd": nat["auc_roc_sd"],
                "full_auc_roc":      ful["auc_roc_mean"],
                "full_auc_roc_sd":   ful["auc_roc_sd"],
                "delta_auc_roc":     ful["auc_roc_mean"] - nat["auc_roc_mean"],
            })

    # Aggregates
    all_delta_pr  = [r["delta_auc_pr"]  for r in summary_rows]
    all_delta_roc = [r["delta_auc_roc"] for r in summary_rows]

    mean_delta_pr  = float(np.mean(all_delta_pr))
    max_delta_pr   = float(np.max(all_delta_pr))
    mean_delta_roc = float(np.mean(all_delta_roc))
    max_delta_roc  = float(np.max(all_delta_roc))

    # Interpretation verdict
    if max_delta_pr < 0.05 and max_delta_roc < 0.05:
        interpretation = (
            f"Max Δ AUC-PR={max_delta_pr:+.4f}, Max Δ AUC-ROC={max_delta_roc:+.4f} — "
            "both below 0.05 threshold. Synthetic features add minimal signal; "
            "no evidence of circularity."
        )
    else:
        interpretation = (
            f"Max Δ AUC-PR={max_delta_pr:+.4f}, Max Δ AUC-ROC={max_delta_roc:+.4f} — "
            "synthetic features contribute meaningful signal on at least one metric. "
            "Additional justification for the simulation methodology is recommended."
        )

    output = {
        # ── metadata ──────────────────────────────────────────────────────────
        "native_features":    NATIVE_FEATURES,
        "synthetic_features": SYNTHETIC_FEATURES,
        "full_features":      FULL_FEATURES,
        "seeds":              SEEDS,

        # ── config block required by SensitivityAnalysisSection ───────────────
        "config": {
            "n_seeds":    len(SEEDS),
            "n_rounds":   Config.N_ROUNDS,
            "fedprox_mu": MU_VALUES["FedProx"],   # 0.5
        },

        # ── centralized baseline (with delta fields) ──────────────────────────
        "centralized": {
            "native":        cent_native,
            "full":          cent_full,
            "delta_auc_pr":  float(cent_full["auc_pr"]  - cent_native["auc_pr"]),
            "delta_auc_roc": float(cent_full["auc_roc"] - cent_native["auc_roc"]),
        },

        # ── per-(scenario, algorithm) rows ─────────────────────────────────────
        "summary_rows": summary_rows,

        # ── aggregates ─────────────────────────────────────────────────────────
        "mean_delta_auc_pr":  mean_delta_pr,
        "max_delta_auc_pr":   max_delta_pr,
        "mean_delta_auc_roc": mean_delta_roc,
        "max_delta_auc_roc":  max_delta_roc,

        # ── interpretation ─────────────────────────────────────────────────────
        "interpretation": interpretation,

        # ── raw federated results kept for traceability ────────────────────────
        "federated_native": native_results,
        "federated_full":   full_results,
    }

    out_path = RESULTS_DIR / "sensitivity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved → {out_path}")
    print("\nUsage for manuscript:")
    print("  • Report both AUC-PR and AUC-ROC tables in S1 Text.")
    print("  • Cite mean/max delta for both metrics in Response to Reviewers.")
    print("  • Full-12 column should reproduce your published centralized and federated numbers.")
    print("  • If both max Δ < 0.05: frame as supporting evidence against circularity.")

# =============================================================================

if __name__ == "__main__":
    main()