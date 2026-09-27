# Stage 7 — Final Submission Construction, Validation & Integrity Check Report

## Executive Summary

Stage 7 completes the end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 (Entity Resolution). In this stage, official competition submission files were constructed from the Stage 6 frozen test inference predictions (`data/processed/test_predictions_stage6.parquet`), validated against raw test entity IDs and countries, independently checked for cross-file subset containment, and verified using the competition's official validator script (`utils/validate_submission.py`).

Both output files passed **100% of all integrity and compliance checks** with **zero errors and zero warnings**.

---

## 1. Submission Artifacts Summary

| Artifact | Path | Size | Row Count (excl. header) | Header | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Leaderboard Matching Results** | `output/matching_results.tsv` | 97.97 MB (102,726,119 bytes) | 1,732,544 | `source1_entity_id\tmatched_entity_ids` | **PASSED** |
| **Blocking Verification Candidates** | `output/candidate_pairs.tsv` | 3,664.85 MB (3,842,876,587 bytes) | 1,732,544 | `source1_entity_id\tcandidate_entity_ids` | **PASSED** |
| **Matching Results Backup** | `submissions/stage7_v1/matching_results.tsv` | 97.97 MB (102,726,119 bytes) | 1,732,544 | `source1_entity_id\tmatched_entity_ids` | **IDENTICAL BACKUP** |
| **Candidate Pairs Backup** | `submissions/stage7_v1/candidate_pairs.tsv` | 3,664.85 MB (3,842,876,587 bytes) | 1,732,544 | `source1_entity_id\tcandidate_entity_ids` | **IDENTICAL BACKUP** |
| **Machine-Readable Validation** | `data/models/stage7_submission_validation_results.json` | 1.45 KB | N/A | JSON | **SAVED** |

---

## 2. Match & Candidate Statistics

### A. Prediction Breakdown (`matching_results.tsv`)
- **Total Test Source 1 Entities:** 1,732,544 (100.0% represented)
- **Total Matches Predicted:** 6,219,897
  - **Source 2 Matches:** 2,858,906 (45.96%)
  - **Source 3 Matches:** 3,360,991 (54.04%)
- **Match Count Distribution:**
  - **0 matches (non-match S1):** 231,210 (13.35%)
  - **Exactly 1 match:** 347,066 (20.03%)
  - **2–3 matches:** 742,668 (42.87%)
  - **4–5 matches:** 305,387 (17.63%)
  - **6+ matches:** 106,213 (6.13%)
  - **Max matches for a single S1:** 699
  - **Mean matches per S1:** 3.59

### B. Candidate Generation Breakdown (`candidate_pairs.tsv`)
- **Total Test Candidates Generated:** 296,419,213
- **Mean Candidates per S1:** 171.09
- **Max Candidates for a single S1:** 2,752
- **Empty Candidate Entities (0 candidates blocked):** 3,279 (0.19%)
- **Non-Empty Candidate Entities:** 1,729,265 (99.81%)

---

## 3. Strict Compliance & Integrity Verification

### Check 1: Exact Line Count & Header Format
- **Expected Lines (including header):** 1,732,545
- **Actual Lines in `matching_results.tsv`:** 1,732,545 (`PASS`)
- **Actual Lines in `candidate_pairs.tsv`:** 1,732,545 (`PASS`)
- **Header Check:**
  - `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids\n` (`PASS`)
  - `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids\n` (`PASS`)

### Check 2: Order & Uniqueness of S1 Entity IDs
- Both files preserve the exact line-by-line sequence of entities found in the raw `data/raw/test/test_source1.tsv`.
- Zero missing S1 entities: `0`
- Zero duplicate S1 entities: `0`

### Check 3: TSV Formatting & Quote Integrity
- Both files were generated strictly without enclosing quotes (`quote_style="never"`).
- Target ID delimiters are strictly single commas (`,`) with zero leading, trailing, or inner whitespace.
- Empty match and empty candidate lines terminate with a clean trailing tab (`\t\n`), strictly representing empty string values without placeholder text or erroneous characters.

### Check 4: Cross-File Subset Containment (Critical Leaderboard Rule)
- **Rule:** For every Source 1 entity, the set of matched IDs in `matching_results.tsv` must be a strict subset of candidate IDs in `candidate_pairs.tsv` ($\text{Matches}(s_1) \subseteq \text{Candidates}(s_1)$).
- **Result:** **0 violations out of 1,732,544 rows.**
- Every single one of the 6,219,897 predicted matches appears in `candidate_pairs.tsv`.

### Check 5: Target Entity ID Existence & Prefix Validation
- Every predicted match ID was checked against the official test catalogs:
  - Valid `S2-` entities loaded from `data/raw/test/test_source2.tsv`: 4,008,610
  - Valid `S3-` entities loaded from `data/raw/test/test_source3.tsv`: 5,961,847
- **Invalid ID count:** `0` (Zero non-existent IDs predicted)
- **Invalid prefix count:** `0` (Only valid `S2-` and `S3-` prefixes exist)
- **Intra-list duplicates:** `0` (No repeated IDs for any entity)

### Check 6: Geographical Country Isolation
- **Rule:** Entities can only match across the same country partition.
- **France partition:** 259,452 S1 entities, 0 cross-country matches (`PASS`)
- **US partition:** 663,106 S1 entities, 0 cross-country matches (`PASS`)
- **India partition:** 809,986 S1 entities, 0 cross-country matches (`PASS`)
- **Total Cross-Country Matches:** `0`

---

## 4. Official Validator Results

The submission was evaluated using the official competition validator script:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/raw/test
```

### Official Output Log
```
ML Challenge 2026 — submission validator
  test dir: C:\Users\KIIT\Desktop\Wyvern\data\raw\test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (231210 empty, 1501334 non-empty).
  candidate_pairs.tsv: 1732544 rows (3279 empty, 1729265 non-empty).

PASS — no blocking issues found. Safe to submit.
```

The script executed in **1,581.03s** (~26.3 minutes) and exited with `returncode = 0`.

---

## 5. Raw Data Integrity Confirmation

All raw dataset files under `data/raw/` remain 100% byte-for-byte untouched and intact:
- `data/raw/train/train_source1.tsv`: Untouched
- `data/raw/train/train_source2.tsv`: Untouched
- `data/raw/train/train_source3.tsv`: Untouched
- `data/raw/train/train_ground_truth.tsv`: Untouched
- `data/raw/test/test_source1.tsv`: Untouched
- `data/raw/test/test_source2.tsv`: Untouched
- `data/raw/test/test_source3.tsv`: Untouched

---

## 6. Submission Instructions for User

The files are ready for manual submission to the Amazon ML Challenge portal:
1. **Leaderboard Submission:** Upload `output/matching_results.tsv` (or `submissions/stage7_v1/matching_results.tsv`).
2. **Blocking Candidate Verification:** If requested by the evaluation portal, upload `output/candidate_pairs.tsv` (or `submissions/stage7_v1/candidate_pairs.tsv`).
3. Keep the backup directory `submissions/stage7_v1/` as a permanent frozen snapshot of this submission.
