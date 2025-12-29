
# 🇬🇭 STAGE 1: Ghana Malaria DHS/MIS Data Extraction Pipeline

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
RAW_DATA = "data/raw"
MERGED_DATA = "data/merged"
RESULTS_DIR = 'validation'

# Ensure directories exist
os.makedirs(MERGED_DATA, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# File Paths
MIS16_FILES = {"pr": f"{RAW_DATA}/GHPR7BFL.DTA", "kr": f"{RAW_DATA}/GHKR7BFL.DTA"}
MIS19_FILES = {"pr": f"{RAW_DATA}/GHPR82FL.DTA", "kr": f"{RAW_DATA}/GHKR82FL.DTA"}
DHS22_FILES = {"pr": f"{RAW_DATA}/GHPR8CFL.DTA", "kr": f"{RAW_DATA}/GHKR8CFL.DTA"}

# Variables of interest
PR_VARS = [
    "hv001", "hv002", "hvidx", "hv005", "hv006", "hv042",
    "hv103", "hv104", "hc1", "hv024", "hv025", "hml12",
    "hml32", "hml35"
]

KR_VARS_BASE = ["v001", "v002", "b16", "h22"]
KR_VARS_DHS = KR_VARS_BASE + ["h11"]

# --- Helper Functions ---
def load_data(file_paths, survey_name, pr_vars_needed=None, kr_vars_needed=None):
    """Loads Stata datasets safely, handling label errors."""
    print(f"Loading {survey_name} datasets...")
    
    REPAIR_MAPS = {
        "hv025": {1: "Urban", 2: "Rural"},
        "hv024": {1: "Western", 2: "Central", 3: "Greater Accra", 4: "Volta", 5: "Eastern", 6: "Ashanti", 7: "Brong Ahafo", 8: "Northern", 9: "Upper East", 10: "Upper West"},
        "hml12": {0: "Did not sleep under a net", 1: "Only treated (ITN) nets", 2: "Both treated (ITN) and untreated nets", 3: "Only untreated nets"},
        "h22": {0: "No", 1: "Yes", 8: "Don't know", 9: "Missing"},
        "h11": {0: "No", 1: "Yes, last 24 hours", 2: "Yes, last two weeks", 8: "Don't know"},
        "hml35": {0: "Negative", 1: "Positive"},
        "hml32": {0: "Negative", 1: "Positive"}
    }

    def safe_read(path, var_list):
        if not path: return None
        try:
            return pd.read_stata(path, columns=var_list)
        except (ValueError, Exception) as e:
            if "not unique" in str(e) or "categoricals" in str(e):
                print(f"   ⚠️  Label error in {os.path.basename(path)}. Repairing...")
                df = pd.read_stata(path, columns=var_list, convert_categoricals=False)
                for col, mapping in REPAIR_MAPS.items():
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].map(mapping).fillna(df[col])
                return df
            else:
                print(f"   Critical Error reading {path}: {e}")
                return None

    pr = safe_read(file_paths["pr"], pr_vars_needed)
    kr = safe_read(file_paths["kr"], kr_vars_needed)

    if pr is not None and kr is not None:
        print(f"   PR: {len(pr):,} | KR: {len(kr):,}")
        return pr, kr
    else:
        return None, None

def clean_and_recode(pr_df, kr_df, survey_type, survey_year, kr_vars_list):
    print(f"\n⚙️  Cleaning {survey_year} {survey_type} data...")
    df = pr_df[PR_VARS].copy()
    kr_subset = kr_df[kr_vars_list].copy().rename(columns={'v001': 'hv001', 'v002': 'hv002', 'b16': 'hvidx'})
    
    df = df.merge(kr_subset, on=['hv001', 'hv002', 'hvidx'], how='left')

    df['survey_type'] = survey_type
    df['survey_year'] = survey_year
    df["weight"] = pd.to_numeric(df["hv005"], errors="coerce") / 1_000_000
    df['age_months'] = pd.to_numeric(df['hc1'], errors='coerce')

    # Age Group
    df["age_group"] = pd.cut(df["age_months"], bins=[0, 6, 12, 24, 36, 48, 61], labels=["<6", "Below 12", "Below 24", "Below 36", "Below 48", "Below 60"], right=False, include_lowest=True)

    # Residence
    df["residence_label"] = df["hv025"].astype(str).str.strip().str.title()
    df["residence_num"] = df["residence_label"].map({"Urban": 1, "Rural": 2})

    # Fever (H22)
    h22_mapping = {"0": 0, "no": 0, "1": 1, "yes": 1, "8": 8, "don't know": 8}
    df["h22_num"] = df["h22"].astype(str).str.lower().map(h22_mapping).astype("Int64")
    df["fever"] = df["h22_num"].map({1: 1, 0: 0}).astype("Int64")
    df["had_fever"] = df["h22_num"].map({0: "No", 1: "Yes", 8: "Don't know"})

    # Diarrhea (H11 - DHS only)
    if 'h11' in df.columns:
        h11_mapping = {"0": 0, "no": 0, "1": 1, "yes, last 24 hours": 1, "2": 2, "yes, last two weeks": 2, "8": 8}
        df["h11_num"] = df["h11"].astype(str).str.lower().map(h11_mapping).astype("Int64")
        df["diarrhea"] = df["h11_num"].map({1: 1, 2: 1, 0: 0}).astype("Int64")
        df["had_diarrhea"] = df["diarrhea"].map({1: "Yes", 0: "No"})
    else:
        df["diarrhea"] = np.nan
        df["had_diarrhea"] = np.nan

    # Region Collapse
    region_collapse = {
        "Western": "Western", "Central": "Central", "Greater Accra": "Greater Accra", "Volta": "Volta",
        "Eastern": "Eastern", "Ashanti": "Ashanti", "Western North": "Western", "Ahafo": "Brong Ahafo",
        "Bono": "Brong Ahafo", "Bono East": "Brong Ahafo", "Oti": "Volta", "Northern": "Northern",
        "Savannah": "Northern", "North East": "Northern", "Upper East": "Upper East", "Upper West": "Upper West"
    }
    df["region_label"] = df["hv024"].astype(str).str.strip().str.title()
    if survey_year == "2022": df["region_label"] = df["region_label"].map(region_collapse)
    
    region_mapping_10 = {"Western": 1, "Central": 2, "Greater Accra": 3, "Volta": 4, "Eastern": 5, "Ashanti": 6, "Brong Ahafo": 7, "Northern": 8, "Upper East": 9, "Upper West": 10}
    df["region_num"] = df["region_label"].map(region_mapping_10)

    # Bed Net
    net_map = {"did not sleep under a net": 0, "only treated (itn) nets": 1, "both treated (itn) and untreated nets": 2, "only untreated nets": 3}
    df["bed_net_use_code"] = df["hml12"].astype(str).str.lower().str.strip().map(net_map)
    df["bed_net_use_label"] = df["bed_net_use_code"].replace({0: "did not sleep under a net", 1: "only treated (itn) nets", 2: "both treated (itn) and untreated nets", 3: "only untreated nets"})
    df["bed_net_use_num"] = df["bed_net_use_code"].isin([1, 2]).astype(float)

    # Testing & Malaria
    df["malaria"] = df["hml35"].astype(str).str.lower().str.strip().map({"negative": "Negative", "positive": "Positive"})
    df["rdt_test"] = df["malaria"].map({"Positive": 1, "Negative": 0}).astype("Int64")
    
    df["microscopy_test"] = df["hml32"].astype(str).str.lower().str.strip().map({"positive": 1, "negative": 0}).astype("Int64")

    # Filtering
    selection_map = {"1": 1, "selected": 1, "0": 0, "not selected": 0}
    df["selected"] = df["hv042"].astype(str).str.lower().map(selection_map)
    df["slept"] = df["hv103"].astype(str).str.lower().map({"yes": 1, "no": 0})

    df_rdt = df[
        (df["selected"] == 1) & (df["slept"] == 1) &
        (df["age_months"] >= 6) & (df["age_months"] <= 59) &
        (df["rdt_test"].isin([0, 1]))
    ].copy()

    # Season
    df_rdt["season"] = np.where(pd.to_numeric(df_rdt["hv006"], errors='coerce').isin([4,5,6,7,8,9,10]), "Rainy", "Dry")

    keep_cols = ["survey_type", "survey_year", "age_group", "age_months", "region_label", "region_num", "residence_label", "residence_num", "bed_net_use_label", "bed_net_use_num", "had_fever", "fever", "had_diarrhea", "diarrhea", "malaria", "rdt_test", "microscopy_test", "season", "weight"]
    return df_rdt.reindex(columns=keep_cols)

def weighted_prevalence(data, positive_col, weight_col='sample_weight'):
    if weight_col not in data.columns or positive_col not in data.columns or data[weight_col].isnull().all():
        return 0.0
    valid_data = data.dropna(subset=[positive_col, weight_col])
    if valid_data.empty: return 0.0
    return (valid_data.loc[valid_data[positive_col] == 1, weight_col].sum() / valid_data[weight_col].sum()) * 100

def combine_and_analyze(dfs_to_merge):
    print("\n🔄 Merging Datasets...")
    valid_dfs = [d for d in dfs_to_merge if d is not None]
    if not valid_dfs: return

    ghana_merged = pd.concat(valid_dfs, ignore_index=True)
    ghana_merged["sample_weight"] = pd.to_numeric(ghana_merged.get("weight"), errors="coerce")
    ghana_merged["rdt_test"] = pd.to_numeric(ghana_merged.get("rdt_test"), errors="coerce")
    ghana_merged["microscopy_test"] = pd.to_numeric(ghana_merged.get("microscopy_test"), errors='coerce')
    ghana_merged["fever"] = pd.to_numeric(ghana_merged.get("fever"), errors='coerce') # Ensure numeric
    
    ghana_merged.dropna(subset=['sample_weight', 'survey_year', 'region_label'], inplace=True)

    print(f"Total Records: {len(ghana_merged):,}")

    # Prevalence Calculation (RDT/Microscopy/Fever)
    def calc_prev(df, group_col):
        return df.groupby(group_col).apply(lambda g: pd.Series({
            'rdt_prev': weighted_prevalence(g, 'rdt_test'),
            'micro_prev': weighted_prevalence(g, 'microscopy_test'),
            'fever_prev': weighted_prevalence(g, 'fever')
        }))

    yearly_prev = calc_prev(ghana_merged, 'survey_year')
    regional_prev = calc_prev(ghana_merged, 'region_label')
    
    print("\n Regional Prevalence (Long-term):")
    print(regional_prev.round(2))
    
    yearly_prev.to_csv(f"{RESULTS_DIR}/yearly_weighted_prevalence.csv")
    regional_prev.to_csv(f"{RESULTS_DIR}/regional_weighted_prevalence.csv")

    # Risk Classification & Export
    reg_risk = regional_prev[['rdt_prev']].rename(columns={'rdt_prev': 'reg_prev_pct'}).reset_index()
    ghana = ghana_merged.merge(reg_risk, on="region_label", how="left")
    ghana["region_risk"] = np.where(ghana["reg_prev_pct"] >= ghana["reg_prev_pct"].median(), "High risk", "Low risk")

    final_cols = ["survey_type", "survey_year", "age_group", "age_months", "region_label", "region_num", "residence_label", "residence_num", "bed_net_use_label", "bed_net_use_num", "had_fever", "fever", "had_diarrhea", "diarrhea", "malaria", "rdt_test", "microscopy_test", "season", "weight", "sample_weight", "reg_prev_pct", "region_risk"]
    
    ghana_final = ghana.reindex(columns=final_cols)
    ghana_final.to_csv(f"{MERGED_DATA}/ghana_malaria_merged.csv", index=False)
    print(f"\n Saved: {MERGED_DATA}/ghana_malaria_merged.csv")

# Execution
data_frames = []
pr_mis16, kr_mis16 = load_data(MIS16_FILES, "MIS 2016", PR_VARS, KR_VARS_BASE)
if pr_mis16 is not None: data_frames.append(clean_and_recode(pr_mis16, kr_mis16, "MIS", "2016", KR_VARS_BASE))

pr_mis19, kr_mis19 = load_data(MIS19_FILES, "MIS 2019", PR_VARS, KR_VARS_BASE)
if pr_mis19 is not None: data_frames.append(clean_and_recode(pr_mis19, kr_mis19, "MIS", "2019", KR_VARS_BASE))

pr_dhs22, kr_dhs22 = load_data(DHS22_FILES, "DHS 2022", PR_VARS, KR_VARS_DHS)
if pr_dhs22 is not None: data_frames.append(clean_and_recode(pr_dhs22, kr_dhs22, "DHS", "2022", KR_VARS_DHS))

combine_and_analyze(data_frames)