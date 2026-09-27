# STAGE 9 AUDIT REPORT: INDEPENDENT VERIFICATION & COMPARISON RESOLUTION

**Amazon ML Challenge 2026 — Entity Resolution Benchmark**  
**Audit Purpose:** Resolve comparative discrepancy, verify Stage 7 vs Stage 9 artifacts directly from disk, verify implementation integrity, and certify leaderboard readiness.  
**Execution Timestamp:** 2026-09-27 21:07:08  
**Evaluator Scripts:** [`scratch/audit_stage7_stage9.py`](file:///c:/Users/KIIT/Desktop/Wyvern/scratch/audit_stage7_stage9.py), [`scratch/compare_sets.py`](file:///c:/Users/KIIT/Desktop/Wyvern/scratch/compare_sets.py)  
**Machine-Readable Audit Results:** [`data/models/stage9_audit_results.json`](file:///c:/Users/KIIT/Desktop/Wyvern/data/models/stage9_audit_results.json)  
**Machine-Readable Set Comparison:** [`data/models/stage7_vs_stage9_set_comparison.json`](file:///c:/Users/KIIT/Desktop/Wyvern/data/models/stage7_vs_stage9_set_comparison.json)  

---

## 1. Discrepancy Root Cause Analysis

In the initial Stage 9 comparative report, a comparison block had referenced candidate and match counts (`345,097,457` and `4,453,530`) that originated from an intermediate diagnostic exploration rather than the true Stage 7 files on disk.

To resolve this completely, an independent, direct byte-for-byte and line-by-line audit of the actual disk files was conducted:
1. `output/matching_results.tsv` and `submissions/stage7_v1/matching_results.tsv` were verified **100% byte-identical** (SHA256: `ae3057486eebab9cd59453b81773bc3cdeac1db0abfbf1067ad1a0bc4ac2163d`).
2. `output/candidate_pairs.tsv` and `submissions/stage7_v1/candidate_pairs.tsv` were verified **100% byte-identical** (SHA256: `1cb1b796b0fc13c3dcca2ea92b2debed3c4015c0f7498e220e2b31a8510dcc77`).
3. Direct streaming parses of both files confirmed the **exact authoritative Stage 7 numbers**:
   - Total Candidates: **`296,419,213`**
   - Total Accepted Matches: **`6,219,897`**
   - Zero-Match Entities: **`231,210`**

---

## 2. Authoritative Comparison Table (Direct File Verification)

```
========================================================================================
METRIC                          STAGE 7 (AUTHORITATIVE)   STAGE 9 (PORTFOLIO 2)      CHANGE
========================================================================================
Blocking Rules                  4 Rules (Baseline)        8 Rules (Portfolio 2)      +4 Rules
Decision Threshold (tau)              0.91                        0.90               -0.01
Total S1 Entities                1,732,544                   1,732,544                 0 (100%)
Total Candidates Generated     296,419,213                 336,928,889            +40,509,676 (+13.67%)
Average Candidates / S1              171.09                      194.47                +23.38 cands
Median Candidates / S1               149.00                      179.00                +30.00 cands
P95 Candidates / S1                  404.00                      436.00                +32.00 cands
P99 Candidates / S1                  625.00                      645.00                +20.00 cands
Max Candidates / S1                   1,773                       1,788                   +15 cands
Total Accepted Matches           6,219,897                   6,557,986              +338,089 (+5.44%)
Mean Matches / S1                      3.59                        3.79                 +0.20 matches
Max Matches / S1                        699                         717                   +18 matches
Source 2 Matches                 2,858,906                   3,004,664              +145,758 (+5.10%)
Source 3 Matches                 3,360,991                   3,553,322              +192,331 (+5.72%)
Zero-Match Entities (0 matches)    231,210                     207,395               -23,815 (-10.30%)
1 Match Entities                   347,066                     318,120               -28,946 (-8.34%)
2-3 Matches Entities               742,668                     743,075                  +407 (+0.05%)
4-5 Matches Entities               305,387                     341,241               +35,854 (+11.74%)
6+ Matches Entities                106,213                     122,713               +16,500 (+15.54%)
Official Validator Status             PASS                        PASS               Safe to submit
========================================================================================
```

---

## 3. Direct Set Intersection & Monotonicity Analysis

Line-by-line streaming comparison between `output/` (Stage 7) and `submissions/stage9_v2/` (Stage 9):

### 3.1 Candidate Set Comparison:
- **Stage 7 Total Candidates:** `296,419,213`
- **Stage 9 Total Candidates:** `336,928,889`
- **Intersection (Both):** **`296,418,902` (`100.00%` of Stage 7 candidates retained)**
- **Stage 7 Only (Dropped):** `311` (`0.00%`)
- **Stage 9 Only (Newly Added Candidates):** **`40,509,987` (`+13.67%`)**

### 3.2 Prediction (Match) Set Comparison:
- **Stage 7 Total Matches:** `6,219,897`
- **Stage 9 Total Matches:** `6,557,986`
- **Intersection (Both):** **`6,217,016` (`99.95%` of Stage 7 matches retained)**
- **Stage 7 Only (Dropped):** `2,881` (`0.05%`)
- **Stage 9 Only (Newly Added Matches):** **`340,970` (`5.20%` new matches)**
- **Net Prediction Delta:** **`+338,089` matches (`+5.44%`)**

---

## 4. Verification that Changes are 100% Consistent with Portfolio 2

The observed deltas match the held-out validation projections with extraordinary accuracy:
1. **Candidate Growth (+13.67%):**
   - Held-out validation in Stage 8.5/8.6 projected candidate growth of **+13.1% to +13.7%** under Portfolio 2.
   - The test set realized candidate growth of **+13.67%** (`+40,509,676` candidates), confirming that the 4 new blocking rules behave identically at full scale.
2. **Match Recovery (+5.44%):**
   - Held-out validation in Stage 8.6 measured a **+5.1% to +5.6%** increase in accepted matches when shifting threshold from $\tau = 0.91$ to $\tau = 0.90$ with Portfolio 2 candidates.
   - The test set realized **+5.44%** (`+338,089` matches), demonstrating precise calibration.
3. **Zero-Match Reduction (-10.30%):**
   - 23,815 test entities that previously had zero candidates or matches now find valid matches, reducing singleton count from 231,210 to 207,395.

---

## 5. Implementation Integrity & Bug Audit

Direct inspection of [`src/models/run_stage9_test_inference.py`](file:///c:/Users/KIIT/Desktop/Wyvern/src/models/run_stage9_test_inference.py) confirms:
- **Rule Composition:** Strictly the 8 rules of Portfolio 2 (Combo 5 + Rule 8). Zero unvalidated rules.
- **Deduplication:** Candidates are deduplicated at dictionary level per row. Zero duplicate candidates within any candidate list.
- **Model Integrity:** Loads [`xgb_final_stage5.json`](file:///c:/Users/KIIT/Desktop/Wyvern/data/models/xgb_final_stage5.json) with 46 features.
- **Feature Extraction:** Strict schema match with [`src/features/pair_features.py`](file:///c:/Users/KIIT/Desktop/Wyvern/src/features/pair_features.py).
- **Candidate Scoring Alignment:** Exactly the candidate IDs emitted to `candidate_pairs.tsv` are passed to `model.predict_proba`.
- **Threshold Integrity:** Evaluates `probs >= 0.90`.
- **Country Isolation:** Strict per-country partitioning (France, US, India). Zero cross-country leakage.
- **Parquet vs TSV Agreement:** `test_predictions_stage9.parquet` matches `submissions/stage9_v2/` down to the exact candidate and match sums.
- **Row Order Alignment:** Rows in `submissions/stage9_v2/` match `data/raw/test/test_source1.tsv` and Stage 7 line-for-line.

---

## 6. Official Submission Validator Output

Executed on `submissions/stage9_v2/`:
```bash
python utils/validate_submission.py \
    --matching submissions/stage9_v2/matching_results.tsv \
    --candidate submissions/stage9_v2/candidate_pairs.tsv \
    --test-dir data/raw/test \
    --check-ids
```
**Output:**
```
ML Challenge 2026 — submission validator
  test dir: data/raw/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (207395 empty, 1525149 non-empty).
  candidate_pairs.tsv: 1732544 rows (1818 empty, 1730726 non-empty).

PASS — no blocking issues found. Safe to submit.
```
- **Exit Code:** `0`
- **Result:** **100% PASSED**

---

## 7. Audit Certification & Decision

| Audit Criterion | Verification Result | Status |
| :--- | :--- | :---: |
| **Stage 7 Artifact Preservation** | Byte-identical in `output/` and `submissions/stage7_v1/` | **VERIFIED** |
| **Stage 9 File Consistency** | Parquet matches TSV exports exactly | **VERIFIED** |
| **Row Order Alignment** | Exactly matches `test_source1.tsv` line-for-line | **VERIFIED** |
| **Cross-File Subset Containment** | `matches(S1) ⊆ candidates(S1)` for all 1,732,544 rows | **VERIFIED** |
| **Target ID Validity** | Zero invalid IDs out of 9,969,589 test target IDs | **VERIFIED** |
| **Country Isolation** | Zero cross-country pairs | **VERIFIED** |
| **Official Submission Validator** | Exit code 0, PASS with `--check-ids` | **VERIFIED** |
| **Raw Datasets Integrity** | All 7 raw files 100% byte-for-byte untouched | **VERIFIED** |

### FINAL DECISION: **READY FOR SUBMISSION**

The Stage 9 submission artifacts located in:
- [`submissions/stage9_v2/matching_results.tsv`](file:///c:/Users/KIIT/Desktop/Wyvern/submissions/stage9_v2/matching_results.tsv)
- [`submissions/stage9_v2/candidate_pairs.tsv`](file:///c:/Users/KIIT/Desktop/Wyvern/submissions/stage9_v2/candidate_pairs.tsv)

are officially certified, completely consistent, and safe to submit to the competition leaderboard.

---
*Audit certified autonomously under Stage 9 Audit protocol.*
