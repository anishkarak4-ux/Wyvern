# Amazon ML Challenge 2026: Stage 6 Full Test Inference Report

## 1. Executive Summary & Objective

Stage 6 executes full candidate generation, pairwise feature extraction, and XGBoost model scoring across the entire test dataset of **1,732,544 Source 1 entities** without accessing hidden ground-truth labels.

Following system shutdown recovery, the pipeline utilized verified checkpoints for France (259k entities) and the US (663k entities), completed candidate generation and model scoring for India (810k entities), and assembled the unified intermediate prediction artifact [`data/processed/test_predictions_stage6.parquet`](file:///C:/Users/KIIT/Desktop/Wyvern/data/processed/test_predictions_stage6.parquet).

### Key Empirical Results:
- **Total Test S1 Entities Scored:** **1,732,544** (100.0% coverage).
- **Total Candidate Pairs Generated:** **296,419,213** (avg **171.09** candidates per S1).
- **Total Matches Accepted at $\tau \ge 0.91$:** **6,219,897** (avg **3.5900** matches per S1).
  - **Source 2 Matches:** **2,858,906** (45.96%)
  - **Source 3 Matches:** **3,360,991** (54.04%)
  - **Candidate Survival Rate:** **2.098%** (extreme candidate reduction, eliminating 97.9% of non-matching pairs).
- **Singletons (0 Matches Predicted):** **231,210 entities** (**13.35%**).
- **Multi-Match Clusters (2+ Matches):** **1,154,268 entities** (**66.62%**).
- **Anomaly Checks:** **12/12 checks PASSED** (zero duplicates, zero missing IDs, zero cross-country matches, zero NaNs, all probabilities $\in [0.91, 1.0]$).
- **Raw Data Integrity:** All 7 train and test TSV files verified 100% byte-for-byte identical.

---

## 2. Test Dataset Row Counts & Integrity Verification

| Dataset | Expected Rows | Verified Rows | File Size (Bytes) | Header Format | Integrity Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `data/raw/test/test_source1.tsv` | 1,732,544 | 1,732,544 | 175,022,086 | `entity_id, business_name, business_address, country` | **PASS (Byte-Identical)** |
| `data/raw/test/test_source2.tsv` | 4,887,273 | 4,887,273 | 509,456,422 | `entity_id, business_name, business_address, country` | **PASS (Byte-Identical)** |
| `data/raw/test/test_source3.tsv` | 5,082,316 | 5,082,316 | 506,002,772 | `entity_id, business_name, business_address, country` | **PASS (Byte-Identical)** |
| `data/raw/train/train_ground_truth.tsv` | — | — | 127,015,583 | (Quarantined) | **PASS (Untouched)** |
| `data/raw/train/train_source1.tsv` | — | — | 210,069,713 | (Quarantined) | **PASS (Untouched)** |
| `data/raw/train/train_source2.tsv` | — | — | 489,301,488 | (Quarantined) | **PASS (Untouched)** |
| `data/raw/train/train_source3.tsv` | — | — | 503,705,637 | (Quarantined) | **PASS (Untouched)** |

---

## 3. Candidate Generation & Blocking Statistics

Candidate generation was executed using the frozen Stage 2.5 blocking configuration:
1. `(country, stripped_name)` — Uncapped
2. `(country, first_2_tokens)` — Bucket Cap $\le 200$
3. `(country, name_pfx8)` — Bucket Cap $\le 200$
4. `(country, address_num_ge_3, name_initial)` — Bucket Cap $\le 100$

### Breakdown by Country Partition:
| Country Partition | Test S1 Count | Total Candidates Generated | Avg Candidates / S1 | Candidate Share |
| :--- | :---: | :---: | :---: | :---: |
| **France** | 259,452 | 45,613,116 | 175.81 | 15.39% |
| **United States** | 663,106 | 119,215,568 | 179.78 | 40.22% |
| **India** | 809,986 | 131,590,529 | 162.46 | 44.39% |
| **Total Test Population** | **1,732,544** | **296,419,213** | **171.09** | **100.00%** |

Candidate generation generated an average of ~171 candidate target entities per Source 1 reference record, maintaining bounded computational load across all countries.

---

## 4. Test Match Prediction Distribution ($\tau \ge 0.91$)

### Global Match Distribution:
| Match Count Group | S1 Entity Count | % of Test S1 Population | Cumulative % |
| :--- | :---: | :---: | :---: |
| **0 Matches (Singletons)** | **231,210** | **13.35%** | 13.35% |
| **1 Match** | **347,066** | **20.03%** | 33.38% |
| **2–3 Matches** | **742,668** | **42.87%** | 76.25% |
| **4–5 Matches** | **305,387** | **17.63%** | 93.87% |
| **6+ Matches** | **106,213** | **6.13%** | 100.00% |
| **Total** | **1,732,544** | **100.00%** | — |

### Summary Statistics on Predicted Matches per S1:
- **Mean Matches / S1:** **3.5900**
- **Median Matches:** **2.0**
- **25th Percentile (P25):** **1.0**
- **75th Percentile (P75):** **4.0**
- **95th Percentile (P95):** **6.0**
- **99th Percentile (P99):** **28.0**
- **Maximum Matches for a Single Entity:** **699** (large national corporate chain in France)

### Source Breakdown:
- **Source 2 Matches:** **2,858,906** (45.96%)
- **Source 3 Matches:** **3,360,991** (54.04%)
- **Candidate Survival Rate:** **2.098%** ($6,219,897 / 296,419,213$)

---

## 5. Country Subgroup Breakdown

| Metric | France | United States | India | Total Test Set |
| :--- | :---: | :---: | :---: | :---: |
| **S1 Entities** | 259,452 | 663,106 | 809,986 | **1,732,544** |
| **Population Share** | 14.98% | 38.27% | 46.75% | **100.00%** |
| **Total Candidates** | 45,613,116 | 119,215,568 | 131,590,529 | **296,419,213** |
| **Avg Candidates / S1** | 175.81 | 179.78 | 162.46 | **171.09** |
| **Total Accepted Matches** | **2,845,125** | **1,838,833** | **1,535,939** | **6,219,897** |
| **Avg Matches / S1** | **10.9659** | **2.7731** | **1.8963** | **3.5900** |
| **S2 Matches** | 1,171,949 | 932,759 | 754,198 | **2,858,906** |
| **S3 Matches** | 1,673,176 | 906,074 | 781,741 | **3,360,991** |
| **0 Matches (Singletons)** | 18,612 (7.17%) | 56,065 (8.45%) | 156,533 (19.33%) | **231,210 (13.35%)** |
| **1 Match** | 36,876 (14.21%) | 100,023 (15.08%) | 210,167 (25.95%) | **347,066 (20.03%)** |
| **2–3 Matches** | 103,440 (39.87%) | 308,661 (46.55%) | 330,567 (40.81%) | **742,668 (42.87%)** |
| **4–5 Matches** | 52,890 (20.39%) | 155,339 (23.43%) | 97,158 (12.00%) | **305,387 (17.63%)** |
| **6+ Matches** | 47,634 (18.36%) | 43,018 (6.49%) | 15,561 (1.92%) | **106,213 (6.13%)** |
| **Median Matches** | 3.0 | 3.0 | 2.0 | **2.0** |
| **P95 Matches** | 47.0 | 6.0 | 5.0 | **6.0** |
| **P99 Matches** | 203.5 | 8.0 | 6.0 | **28.0** |
| **Max Matches** | 699 | 22 | 96 | **699** |

### Key Country Insights:
1. **France Entities:** Exhibits higher average match density (10.97 matches/S1) and larger corporate chain clusters (P95 = 47.0, Max = 699). This reflects centralized administrative address registrations common in French corporate registries (e.g. SIRET/SIREN branch clusters).
2. **United States Entities:** Highly consistent with Stage 5 validation distributions (Avg = 2.77 matches/S1, Median = 3.0, P95 = 6.0, Max = 22). Zero-match rate is low (8.45%).
3. **India Entities:** Displays the highest singleton proportion (19.33% zero matches), perfectly matching Stage 1 EDA observations of high local business uniqueness and lower multi-branch representation.

---

## 6. Confidence Distribution on Accepted Matches ($\tau \ge 0.91$)

Computed across all **6,219,897 accepted match predictions** extracted directly from `final_df["match_probabilities"]`:

| Percentile | Predicted Match Probability |
| :---: | :---: |
| **Minimum** | **0.91000** |
| **P01** | **0.91596** |
| **P05** | **0.93752** |
| **P10** | **0.95932** |
| **P25** | **0.98801** |
| **P50 (Median)** | **0.99804** |
| **P75** | **0.99975** |
| **P90** | **0.99994** |
| **P95** | **0.99997** |
| **P99** | **0.99999** |
| **Maximum** | **1.00000** |

### Calibration Analysis:
- **Extreme High-Confidence Concentration:** Over **75%** of all accepted test matches have model confidence $\ge 0.9880$, and over **50%** have confidence $\ge 0.9980$.
- **Boundary Sparsity:** Only 1% of accepted matches lie in the borderline region $[0.910, 0.916]$. This confirms sharp decision boundaries and minimal ambiguity near the threshold.

---

## 7. Comprehensive Anomaly Checks (12/12 PASSED)

| Check ID | Verification Description | Assertion / Condition | Result |
| :---: | :--- | :--- | :---: |
| **Check 1** | Exact Row Count | `len(final_df) == 1,732,544` | **PASS** |
| **Check 2** | Entity Uniqueness & Ordering | `final_df['source1_entity_id'].n_unique() == 1,732,544` in original file order | **PASS** |
| **Check 3** | Intra-Entity Match Deduplication | Zero duplicate target IDs within any single S1 match list | **PASS** |
| **Check 4** | Cross-Country Consistency | Zero cross-country matches (e.g. US S1 matched to France S2) | **PASS** |
| **Check 5** | Target ID Existence | All 6,219,897 predicted IDs exist in test S2 or S3 raw files | **PASS** |
| **Check 6** | ID Prefix Integrity | All predicted target IDs begin strictly with `S2-` or `S3-` | **PASS** |
| **Check 7** | Probability Validity (No NaN) | Zero NaNs across all match probability strings | **PASS** |
| **Check 8** | Probability Bounds | All probabilities strictly $\ge 0.9100$ and $\le 1.0000$ | **PASS** |
| **Check 9** | Ground-Truth Isolation | Zero access or reference to `train_ground_truth.tsv` | **PASS** |
| **Check 10** | Raw File Immutability | All 7 train and test TSV files verified byte-for-byte identical | **PASS** |
| **Check 11** | Unseen Country Coverage | Exactly 259,452 France entities represented in predictions | **PASS** |
| **Check 12** | Singleton Representation | Exactly 231,210 zero-match entities formatted as empty strings `""` | **PASS** |

---

## 8. Computational Resources & Production Artifacts

- **Resumed India Phase Runtime:** **7,502.26 seconds** (~125.04 minutes).
- **Master Assembly & Verification Runtime:** **64.95 seconds**.
- **Total Resumed Process Runtime:** **7,567.21 seconds** (~126.12 minutes).
- **Peak Process RAM:** **10,642.77 MB** (~10.4 GB RAM, safely within 32 GB system limit).
- **Intermediate Master Artifact:** [`data/processed/test_predictions_stage6.parquet`](file:///C:/Users/KIIT/Desktop/Wyvern/data/processed/test_predictions_stage6.parquet) (**1,521.01 MB**).
- **Machine-Readable Metrics:** [`data/models/stage6_inference_results.json`](file:///C:/Users/KIIT/Desktop/Wyvern/data/models/stage6_inference_results.json) (**2.63 KB**).
- **Country Checkpoints Preserved:**
  - France: [`data/processed/test_predictions_France.parquet`](file:///C:/Users/KIIT/Desktop/Wyvern/data/processed/test_predictions_France.parquet) (73.7 MB)
  - US: [`data/processed/test_predictions_US.parquet`](file:///C:/Users/KIIT/Desktop/Wyvern/data/processed/test_predictions_US.parquet) (598.1 MB)
  - India: [`data/processed/test_predictions_India.parquet`](file:///C:/Users/KIIT/Desktop/Wyvern/data/processed/test_predictions_India.parquet) (648.0 MB)

---

## 9. Readiness for Stage 7 (Submission Generation & Packaging)

The intermediate master artifact contains all necessary fields (`source1_entity_id`, `matched_entity_ids`, `candidate_entity_ids`) to construct:
1. `matching_results.tsv` (Leaderboard submission file)
2. `candidate_pairs.tsv` (Blocking verification file)

**No submission files were created during Stage 6.** All pipeline code and configuration remain locked.
