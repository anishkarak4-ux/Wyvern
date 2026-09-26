# Amazon ML Challenge 2026: Stage 5.5 Final Model Refit & Production Freeze Report

## 1. Executive Summary & Objective

In Stage 5.5, we performed the **final production model refit and configuration freeze** for the Wyvern Business Entity Resolution pipeline.

Having completed model selection, threshold optimization, and safeguard analysis in Stage 5, the objective of Stage 5.5 is to:
1. Combine all available labeled Stage 3 sample data ($X_{\text{train}}$ and $X_{\text{val}}$) into a single unified training matrix $X_{\text{final}}$.
2. Refit the frozen Stage 4 XGBoost architecture for the full 1,000 boosting rounds without changing any hyperparameters or features.
3. Lock and freeze the single global decision threshold at **$\tau = 0.91$** (as determined by official Macro Entity-Level $F_{0.5}$ optimization in Stage 5).
4. Establish a single source of truth for production configuration ([`src/config/production_config.py`](file:///C:/Users/KIIT/Desktop/Wyvern/src/config/production_config.py)) to ensure seamless, error-free Stage 6 inference.
5. Strictly guarantee zero access or contamination from test datasets.

---

## 2. Frozen Production Configuration

The following parameters and configurations are permanently frozen for production:

### 2.1 Candidate Blocking Rules (Empirically Validated in Stage 2.5):
1. **Rule 1 (`stripped_name`):** Exact match on `(country, stripped_name)` — **Uncapped**.
2. **Rule 2 (`first_2_tokens`):** Exact match on `(country, first_2_tokens)` — **Bucket Cap $\le 200$**.
3. **Rule 3 (`name_pfx8`):** Exact match on `(country, name_pfx8)` — **Bucket Cap $\le 200$**.
4. **Rule 4 (`addr_num_initial`):** Exact match on `(country, address_num_ge_3, name_initial)` — **Bucket Cap $\le 100$**.

### 2.2 Feature Vector Architecture (Stage 3):
- Exactly **46 deterministic features** defined in [`src/features/pair_features.py`](file:///C:/Users/KIIT/Desktop/Wyvern/src/features/pair_features.py).
- Features span:
  - Name lexical, token, and string similarity (15 features)
  - Address lexical, token, and numeric agreement (13 features)
  - Cross-field interactions and agreement flags (6 features)
  - Metadata, missingness indicators, and blocking rule provenance (12 features)

### 2.3 XGBoost Hyperparameters (Stage 4 Baseline):
```python
XGB_PARAMS = {
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
```

### 2.4 Decision Threshold:
- **$\tau = 0.91$ (Global)**.
- Any candidate pair $(s_1, t)$ with predicted probability $\hat{P} \ge 0.91$ is accepted as a match.

---

## 3. Final Labeled Training Dataset Profile

The training dataset was constructed by concatenating the disjoint train and validation partitions from [`data/processed/stage3_sample_features.npz`](file:///C:/Users/KIIT/Desktop/Wyvern/data/processed/stage3_sample_features.npz):

| Partition | Total Candidate Pairs | Positive Matches ($y=1$) | Negative Pairs ($y=0$) | Positive % |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 3 Train** | 471,697 | 99,051 | 372,646 | 21.00% |
| **Stage 3 Validation** | 117,823 | 24,858 | 92,965 | 21.09% |
| **Combined Final ($X_{\text{final}}$)** | **589,520** | **123,909** | **465,611** | **21.02%** |

### Data Quality & Integrity Checks:
- **Matrix Dimensions:** 589,520 pairs $\times$ 46 features.
- **NaN / Inf Checks:** Exactly **0** NaNs, **0** Infs.
- **Feature Ordering:** Exactly identical to `FEATURE_NAMES` in `production_config.py`.

---

## 4. Official Model Selection & Validation Evidence

In accordance with strict ML engineering standards:
> [!IMPORTANT]
> The final model was refitted on 100% of the labeled sample ($X_{\text{train}} \cup X_{\text{val}}$) after all model architecture, hyperparameter, and threshold choices were completed.
> Therefore, no new training metrics are claimed as validation performance.
> The held-out Stage 5 evaluation remains the official, unbiased model-selection benchmark.

### Official Held-Out Benchmark (Stage 5 Results across 10,002 Validation Entities):
- **Validation Macro Entity-Level $F_{0.5}$:** **0.77853**
- **Validation Macro Precision:** **0.86098** (86.10%)
- **Validation Macro Recall:** **0.64966** (64.97%)
- **Validation Micro Precision:** **0.87605** (87.60%)
- **Validation Micro Recall:** **0.63691** (63.69%)
- **Validation Micro $F_{0.5}$:** **0.81486** (81.49%)
- **Singleton Zero-Prediction Accuracy:** **86.76%** (485 / 559 singletons correctly predicted with 0 matches)
- **Candidate Blocking Recall Ceiling:** **71.68%** ($24,858 / 34,677$ matches recovered by blocking)

---

## 5. Production Artifacts Inventory

1. [`src/config/production_config.py`](file:///C:/Users/KIIT/Desktop/Wyvern/src/config/production_config.py): Single source of truth containing all frozen blocking rules, hyperparameters, threshold, feature definitions, and validation metadata.
2. [`src/models/refit_final_model.py`](file:///C:/Users/KIIT/Desktop/Wyvern/src/models/refit_final_model.py): Production script for deterministic final model refitting, sanity checks, and metadata export.
3. [`data/models/xgb_final_stage5.json`](file:///C:/Users/KIIT/Desktop/Wyvern/data/models/xgb_final_stage5.json): Final refitted production XGBoost model serialized to standard portable JSON.
4. [`data/models/final_model_config.json`](file:///C:/Users/KIIT/Desktop/Wyvern/data/models/final_model_config.json): Complete machine-readable production configuration with training provenance and sanity check results.
5. [`data/models/xgb_stage4_baseline.json`](file:///C:/Users/KIIT/Desktop/Wyvern/data/models/xgb_stage4_baseline.json): Original Stage 4 baseline model (preserved intact).
6. [`data/models/stage5_entity_validation_results.json`](file:///C:/Users/KIIT/Desktop/Wyvern/data/models/stage5_entity_validation_results.json): Full Stage 5 validation evaluation metrics (preserved intact).

---

## 6. Smoke Test Verification Results

To verify inference reliability without touching test data, a smoke test was executed on a disjoint slice of 1,000 pairs from the production feature matrix:

- **Sample Size:** 1,000 pairs (185 true positives, 815 negative pairs).
- **Probability Output Range:** All values strictly bounded within $[0.0, 1.0]$. Exactly **0** NaNs.
- **Mean Predicted Probability on Positives:** **0.9725**
- **Mean Predicted Probability on Negatives:** **0.0104**
- **Discrimination Verification:** High positive confidence and aggressive negative rejection confirm proper calibration and model integrity.

---

## 7. Computational Performance & System Resources

- **Model Training Duration:** **21.99 seconds** (1,000 boosting rounds with `tree_method='hist'`).
- **Total Pipeline Runtime:** **23.01 seconds** (including data loading, concatenation, training, serialization, and sanity checks).
- **Peak RAM Consumption:** **369.39 MB**.
- **Model File Size:** **4.66 MB** (`xgb_final_stage5.json`).

---

## 8. Integrity Guarantees

- **Raw Data Integrity:** All 7 files under `data/raw/` remain 100% byte-for-byte identical to baseline.
- **Zero Test Data Access:** Exactly **0** test files (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`) were opened, loaded, or accessed during Stage 5.5.
- **Zero Submission Files Generated:** No `matching_results.tsv` or `candidate_pairs.tsv` was created.
- **Preservation of Baselines:** `xgb_stage4_baseline.json` and `stage5_entity_validation_results.json` remain completely unmodified.
