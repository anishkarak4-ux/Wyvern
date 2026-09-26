"""
production_config.py - Frozen Production Configuration for Wyvern Entity Resolution Pipeline
Amazon ML Challenge 2026

Single source of truth for all production settings across Stage 5.5 and Stage 6 inference.
DO NOT modify frozen parameters without formal validation and review.
"""

import os
from typing import Dict, Any, List

# Base directories
BASE_DIR = r"C:\Users\KIIT\Desktop\Wyvern"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Model paths
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_final_stage5.json")
FINAL_CONFIG_PATH = os.path.join(MODELS_DIR, "final_model_config.json")
BASELINE_STAGE4_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_stage4_baseline.json")
STAGE5_VALIDATION_RESULTS_PATH = os.path.join(MODELS_DIR, "stage5_entity_validation_results.json")

# Stage 3 feature data path
STAGE3_FEATURES_PATH = os.path.join(PROCESSED_DIR, "stage3_sample_features.npz")

# -----------------------------------------------------------------------------
# 1. FROZEN BLOCKING CONFIGURATION (Stages 2.5 - 5.5)
# -----------------------------------------------------------------------------
BLOCKING_RULES = [
    {
        "name": "rule_stripped_name",
        "keys": ["country", "stripped_name"],
        "bucket_cap": None,  # Uncapped
        "description": "Exact match on country and normalized alphanumeric name (no spaces/punctuation)",
    },
    {
        "name": "rule_first_2_tokens",
        "keys": ["country", "first_2_tokens"],
        "bucket_cap": 200,
        "description": "Exact match on country and first two name tokens (cap <= 200)",
    },
    {
        "name": "rule_name_pfx8",
        "keys": ["country", "name_pfx8"],
        "bucket_cap": 200,
        "description": "Exact match on country and first 8 characters of stripped name (cap <= 200)",
    },
    {
        "name": "rule_addr_num_initial",
        "keys": ["country", "address_num_ge_3", "name_initial"],
        "bucket_cap": 100,
        "description": "Exact match on country, address numeric >= 3 digits, and first letter of name (cap <= 100)",
    },
]

# -----------------------------------------------------------------------------
# 2. FROZEN XGBOOST HYPERPARAMETERS (Stage 4 baseline)
# -----------------------------------------------------------------------------
XGB_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": -1,
}

# -----------------------------------------------------------------------------
# 3. FROZEN DECISION THRESHOLD (Stage 5 Macro F0.5 Optimum)
# -----------------------------------------------------------------------------
DECISION_THRESHOLD = 0.91

# -----------------------------------------------------------------------------
# 4. OFFICIAL VALIDATION EVIDENCE (Stage 5 Held-Out Results)
# -----------------------------------------------------------------------------
STAGE5_VALIDATION_METRICS = {
    "validation_entity_count": 10002,
    "validation_singletons": 559,
    "validation_gt_matches": 34677,
    "selected_threshold": 0.91,
    "macro_f05": 0.7785301848362192,
    "macro_precision": 0.8609841888925043,
    "macro_recall": 0.6496591843247512,
    "micro_precision": 0.8760461703224783,
    "micro_recall": 0.6369063067739423,
    "micro_f05": 0.8148552622840741,
    "blocking_recall_ceiling_pct": 71.68440176485855,
    "singleton_zero_prediction_rate": 0.8676207513416816,
}

# -----------------------------------------------------------------------------
# 5. FROZEN FEATURE ORDERING (Exactly 46 Features from pair_features.py)
# -----------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # --- Name Lexical & Token Features (1-15) ---
    "name_exact",
    "name_stripped_exact",
    "name_len_diff",
    "name_len_ratio",
    "name_stripped_len_diff",
    "name_jaro_winkler",
    "name_levenshtein_norm",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_first_two_tokens_match",
    "name_prefix_8_match",
    "name_first_word_match",
    "name_num_shared_tokens",
    "name_token_jaccard",

    # --- Address Lexical, Token & Numeric Features (16-28) ---
    "addr_exact",
    "addr_len_diff",
    "addr_jaro_winkler",
    "addr_levenshtein_norm",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_num_shared_tokens",
    "addr_token_jaccard",
    "shared_numeric_count",
    "shared_numeric_ratio",
    "exact_numeric_match",
    "has_postal_code_pair",
    "postal_code_match",

    # --- Cross-Field Interaction Features (29-34) ---
    "name_addr_sim_product",
    "name_exact_and_addr_sim",
    "name_sim_and_shared_num",
    "both_fields_agree_strong",
    "name_only_agrees",
    "addr_only_agrees",

    # --- Metadata & Blocking Provenance Features (35-46) ---
    "same_country",
    "is_source3",
    "s1_missing_name",
    "t_missing_name",
    "s1_missing_addr",
    "t_missing_addr",
    "both_have_addr",
    "blocking_rule_stripped",
    "blocking_rule_tok2",
    "blocking_rule_pfx8",
    "blocking_rule_addr_num",
    "num_blocking_rules",
]

assert len(FEATURE_NAMES) == 46, f"Expected 46 features, got {len(FEATURE_NAMES)}"
