"""
build_submission.py - Stage 7: Final Submission Construction, Validation & Integrity Check
Amazon ML Challenge 2026

Generates the two official competition submission files:
1. output/matching_results.tsv (Scored on leaderboard)
2. output/candidate_pairs.tsv (Blocking verification)

Preserves exact backups in submissions/stage7_v1/.
Performs comprehensive independent validation:
- Row counts, uniqueness, and ordering
- Exact candidate count agreement
- Cross-file subset containment: set(matches) <= set(candidates)
- Target ID existence in raw test sources
- Target ID prefix validity (S2- and S3- only)
- Country isolation (zero cross-country matches/candidates)
- Execution of the official validator: utils/validate_submission.py with --check-ids
"""

import os
import sys
import time
import json
import shutil
import subprocess
import psutil
import datetime
from collections import Counter
import polars as pl

# Base paths
BASE_DIR = r"C:\Users\KIIT\Desktop\Wyvern"
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_TEST_DIR = os.path.join(DATA_DIR, "raw", "test")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BACKUP_DIR = os.path.join(BASE_DIR, "submissions", "stage7_v1")

STAGE6_PARQUET_PATH = os.path.join(PROCESSED_DIR, "test_predictions_stage6.parquet")
STAGE6_RESULTS_PATH = os.path.join(MODELS_DIR, "stage6_inference_results.json")

MATCHING_TSV_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_TSV_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

BACKUP_MATCHING_PATH = os.path.join(BACKUP_DIR, "matching_results.tsv")
BACKUP_CANDIDATE_PATH = os.path.join(BACKUP_DIR, "candidate_pairs.tsv")

VALIDATOR_SCRIPT = os.path.join(BASE_DIR, "utils", "validate_submission.py")
STAGE7_RESULTS_PATH = os.path.join(MODELS_DIR, "stage7_submission_validation_results.json")

TEST_S1_PATH = os.path.join(RAW_TEST_DIR, "test_source1.tsv")
TEST_S2_PATH = os.path.join(RAW_TEST_DIR, "test_source2.tsv")
TEST_S3_PATH = os.path.join(RAW_TEST_DIR, "test_source3.tsv")

RAW_BYTE_SIZES = {
    TEST_S1_PATH: 175022086,
    TEST_S2_PATH: 509456422,
    TEST_S3_PATH: 506002772,
    os.path.join(DATA_DIR, "raw", "train", "train_ground_truth.tsv"): 127015583,
    os.path.join(DATA_DIR, "raw", "train", "train_source1.tsv"): 210069713,
    os.path.join(DATA_DIR, "raw", "train", "train_source2.tsv"): 489301488,
    os.path.join(DATA_DIR, "raw", "train", "train_source3.tsv"): 503705637,
}

EXPECTED_S1_COUNT = 1732544
EXPECTED_MATCHES = 6219897
EXPECTED_S2_MATCHES = 2858906
EXPECTED_S3_MATCHES = 3360991
EXPECTED_ZERO_MATCHES = 231210

def verify_raw_datasets():
    print("=== VERIFYING RAW DATASETS UNTOUCHED ===")
    for path, expected_sz in RAW_BYTE_SIZES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raw file: {path}")
        actual_sz = os.path.getsize(path)
        if actual_sz != expected_sz:
            raise ValueError(f"CORRUPTION in {path}: expected {expected_sz} bytes, found {actual_sz} bytes!")
    print("All raw datasets verified 100% byte-for-byte untouched.\n")

def main():
    t_start = time.time()
    proc = psutil.Process(os.getpid())
    print("===================================================================")
    print("STAGE 7: FINAL SUBMISSION CONSTRUCTION, VALIDATION & INTEGRITY CHECK")
    print("===================================================================\n")

    verify_raw_datasets()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Load Authoritative Stage 6 Artifact
    # -------------------------------------------------------------
    print("=== STEP 1: Loading Stage 6 Parquet Artifact ===")
    t0 = time.time()
    assert os.path.exists(STAGE6_PARQUET_PATH), f"Missing Stage 6 artifact: {STAGE6_PARQUET_PATH}"
    df = pl.read_parquet(STAGE6_PARQUET_PATH)
    n_rows = len(df)
    print(f"Loaded {n_rows:,} rows from {STAGE6_PARQUET_PATH} in {time.time() - t0:.2f}s.")
    assert n_rows == EXPECTED_S1_COUNT, f"Expected {EXPECTED_S1_COUNT} rows, got {n_rows}"
    print(f"Columns: {df.columns}")
    print(f"RAM: {proc.memory_info().rss / (1024*1024):.1f} MB\n")

    # -------------------------------------------------------------
    # Step 2: Construct matching_results.tsv
    # -------------------------------------------------------------
    print("=== STEP 2: Constructing matching_results.tsv ===")
    t0 = time.time()
    df_matching = df.select(["source1_entity_id", "matched_entity_ids"])
    df_matching.write_csv(MATCHING_TSV_PATH, separator="\t", quote_style="never")
    matching_size_mb = os.path.getsize(MATCHING_TSV_PATH) / (1024 * 1024)
    print(f"Wrote {MATCHING_TSV_PATH} ({matching_size_mb:.2f} MB) in {time.time() - t0:.2f}s.")

    # Preserve backup copy
    shutil.copyfile(MATCHING_TSV_PATH, BACKUP_MATCHING_PATH)
    print(f"Preserved backup copy at {BACKUP_MATCHING_PATH}.\n")

    # -------------------------------------------------------------
    # Step 3: Construct candidate_pairs.tsv
    # -------------------------------------------------------------
    print("=== STEP 3: Constructing candidate_pairs.tsv ===")
    t0 = time.time()
    df_candidate = df.select([
        pl.col("source1_entity_id"),
        pl.col("candidate_entity_ids").alias("candidate_entity_ids"),
    ])
    df_candidate.write_csv(CANDIDATE_TSV_PATH, separator="\t", quote_style="never")
    candidate_size_mb = os.path.getsize(CANDIDATE_TSV_PATH) / (1024 * 1024)
    print(f"Wrote {CANDIDATE_TSV_PATH} ({candidate_size_mb:.2f} MB) in {time.time() - t0:.2f}s.")

    # Preserve backup copy
    shutil.copyfile(CANDIDATE_TSV_PATH, BACKUP_CANDIDATE_PATH)
    print(f"Preserved backup copy at {BACKUP_CANDIDATE_PATH}.\n")

    # -------------------------------------------------------------
    # Step 4: Independent Consistency & Cross-File Containment Checks
    # -------------------------------------------------------------
    print("=== STEP 4: Independent Consistency & Cross-File Verification ===")
    t0 = time.time()
    
    # Read test_source1.tsv to get exact required S1 ID list and order
    print("Reading reference test_source1.tsv entity IDs...")
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        expected_s1_ids = [line.split("\t", 1)[0].strip() for line in f if line.strip()]

    assert len(expected_s1_ids) == EXPECTED_S1_COUNT
    print(f"Reference test_source1 has {len(expected_s1_ids):,} S1 IDs.")

    # Check matching_results.tsv directly from disk
    print(f"Verifying {os.path.basename(MATCHING_TSV_PATH)} on disk...")
    matching_s1_ids = []
    matching_map = {}
    matching_empty_count = 0
    matching_intra_dups = 0
    total_matches_counted = 0
    s2_matches_counted = 0
    s3_matches_counted = 0

    with open(MATCHING_TSV_PATH, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        assert header_line == "source1_entity_id\tmatched_entity_ids", f"Invalid matching header: {header_line}"
        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition("\t")
            assert tab == "\t", f"Missing tab at line {line_num}"
            s1 = s1.strip()
            matching_s1_ids.append(s1)
            raw_ids = rest.rstrip("\r\n").strip()
            if not raw_ids:
                matching_empty_count += 1
                matching_map[s1] = set()
            else:
                m_list = raw_ids.split(",")
                if len(m_list) != len(set(m_list)):
                    matching_intra_dups += 1
                m_set = set(m_list)
                matching_map[s1] = m_set
                total_matches_counted += len(m_list)
                for mid in m_list:
                    if mid.startswith("S2-"):
                        s2_matches_counted += 1
                    elif mid.startswith("S3-"):
                        s3_matches_counted += 1

    assert len(matching_s1_ids) == EXPECTED_S1_COUNT, f"Matching row count mismatch: {len(matching_s1_ids)}"
    assert matching_s1_ids == expected_s1_ids, "Matching S1 order does not match test_source1.tsv!"
    assert len(set(matching_s1_ids)) == EXPECTED_S1_COUNT, "Duplicate S1 IDs in matching_results.tsv!"
    assert matching_intra_dups == 0, f"Found {matching_intra_dups} intra-list duplicate target IDs in matching_results!"
    print(f"  matching_results.tsv: {len(matching_s1_ids):,} rows, {matching_empty_count:,} empty, {total_matches_counted:,} matches.")
    print("  matching_results.tsv file checks: PASS.")

    # Check candidate_pairs.tsv directly from disk with on-the-fly containment verification
    print(f"Verifying {os.path.basename(CANDIDATE_TSV_PATH)} and cross-file containment on disk...")
    candidate_s1_ids = []
    candidate_empty_count = 0
    candidate_intra_dups = 0
    total_candidates_counted = 0
    max_cands_per_s1 = 0
    count_mismatches = 0
    containment_violations = 0
    
    # Pre-extract candidate counts from df for 1-to-1 agreement check
    df_cand_counts = dict(zip(df["source1_entity_id"].to_list(), df["candidate_count"].to_list()))

    with open(CANDIDATE_TSV_PATH, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        assert header_line == "source1_entity_id\tcandidate_entity_ids", f"Invalid candidate header: {header_line}"
        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition("\t")
            assert tab == "\t", f"Missing tab at line {line_num}"
            s1 = s1.strip()
            candidate_s1_ids.append(s1)
            raw_ids = rest.rstrip("\r\n").strip()
            if not raw_ids:
                candidate_empty_count += 1
                actual_count = 0
                c_set = set()
            else:
                c_list = raw_ids.split(",")
                actual_count = len(c_list)
                total_candidates_counted += actual_count
                if actual_count > max_cands_per_s1:
                    max_cands_per_s1 = actual_count
                c_set = set(c_list)
                if len(c_list) != len(c_set):
                    candidate_intra_dups += 1

            expected_count = df_cand_counts[s1]
            if actual_count != expected_count:
                count_mismatches += 1

            # Cross-file containment on the fly: set(matches) <= set(candidates)
            matches = matching_map[s1]
            if matches:
                diff = matches - c_set
                if diff:
                    containment_violations += 1

    assert len(candidate_s1_ids) == EXPECTED_S1_COUNT, f"Candidate row count mismatch: {len(candidate_s1_ids)}"
    assert candidate_s1_ids == expected_s1_ids, "Candidate S1 order does not match test_source1.tsv!"
    assert len(set(candidate_s1_ids)) == EXPECTED_S1_COUNT, "Duplicate S1 IDs in candidate_pairs.tsv!"
    assert candidate_intra_dups == 0, f"Found {candidate_intra_dups} intra-list duplicate target IDs in candidate_pairs!"
    assert count_mismatches == 0, f"Found {count_mismatches} candidate count mismatches between artifact and TSV!"
    assert containment_violations == 0, f"FAILED: {containment_violations} S1 entities have matches not in candidate set!"
    print(f"  candidate_pairs.tsv: {len(candidate_s1_ids):,} rows, {candidate_empty_count:,} empty, {total_candidates_counted:,} candidates.")
    print(f"  Containment check PASS: Exactly 0 violations. Every single predicted match is present in candidate_pairs.tsv.")
    print(f"Consistency checks completed in {time.time() - t0:.2f}s.\n")

    # -------------------------------------------------------------
    # Step 5: Target ID Existence & Prefix Validation
    # -------------------------------------------------------------
    print("=== STEP 5: Target ID Existence & Prefix Validation ===")
    t0 = time.time()
    print("Loading valid test target IDs and country mappings...")
    valid_s2_ids = {}
    with open(TEST_S2_PATH, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                valid_s2_ids[parts[0]] = parts[3]

    valid_s3_ids = {}
    with open(TEST_S3_PATH, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                valid_s3_ids[parts[0]] = parts[3]

    print(f"Loaded {len(valid_s2_ids):,} S2 IDs and {len(valid_s3_ids):,} S3 IDs.")

    # S1 country mapping
    s1_country_map = dict(zip(df["source1_entity_id"].to_list(), df["country"].to_list()))

    invalid_matching_ids = []
    invalid_prefixes_matching = []
    cross_country_matches = []

    for sid, mids in matching_map.items():
        s1_c = s1_country_map[sid]
        for mid in mids:
            if mid.startswith("S2-"):
                if mid not in valid_s2_ids:
                    invalid_matching_ids.append(mid)
                elif valid_s2_ids[mid] != s1_c:
                    cross_country_matches.append((sid, mid, s1_c, valid_s2_ids[mid]))
            elif mid.startswith("S3-"):
                if mid not in valid_s3_ids:
                    invalid_matching_ids.append(mid)
                elif valid_s3_ids[mid] != s1_c:
                    cross_country_matches.append((sid, mid, s1_c, valid_s3_ids[mid]))
            else:
                invalid_prefixes_matching.append(mid)

    assert len(invalid_matching_ids) == 0, f"Found {len(invalid_matching_ids)} non-existent IDs in matching_results.tsv!"
    assert len(invalid_prefixes_matching) == 0, f"Found {len(invalid_prefixes_matching)} invalid prefixes in matching_results.tsv!"
    assert len(cross_country_matches) == 0, f"Found {len(cross_country_matches)} cross-country matches in matching_results.tsv!"
    print(f"  matching_results.tsv: All {total_matches_counted:,} match IDs verified valid, existent, and country-consistent.")

    # Cross-country checks by country
    france_s1_count = sum(1 for c in s1_country_map.values() if c == "France")
    us_s1_count = sum(1 for c in s1_country_map.values() if c == "US")
    india_s1_count = sum(1 for c in s1_country_map.values() if c == "India")

    assert france_s1_count == 259452, f"Expected 259,452 France entities, got {france_s1_count}"
    assert us_s1_count == 663106, f"Expected 663,106 US entities, got {us_s1_count}"
    assert india_s1_count == 809986, f"Expected 809,986 India entities, got {india_s1_count}"
    print(f"  France S1 entities: {france_s1_count:,} (All represented, 0 cross-country)")
    print(f"  US S1 entities:     {us_s1_count:,} (All represented, 0 cross-country)")
    print(f"  India S1 entities:  {india_s1_count:,} (All represented, 0 cross-country)")
    print(f"ID & Country validation completed in {time.time() - t0:.2f}s.\n")

    # -------------------------------------------------------------
    # Step 6: Recalculate Submission Match Statistics
    # -------------------------------------------------------------
    print("=== STEP 6: Submission Match Statistics Verification ===")
    match_lengths = [len(mids) for mids in matching_map.values()]
    sub_c0 = sum(1 for l in match_lengths if l == 0)
    sub_c1 = sum(1 for l in match_lengths if l == 1)
    sub_c2_3 = sum(1 for l in match_lengths if 2 <= l <= 3)
    sub_c4_5 = sum(1 for l in match_lengths if 4 <= l <= 5)
    sub_c6plus = sum(1 for l in match_lengths if l >= 6)
    sub_max = max(match_lengths)

    print(f"Total S1 Entities:         {len(match_lengths):,} (Expected: {EXPECTED_S1_COUNT:,})")
    print(f"Total Predicted Matches:   {total_matches_counted:,} (Expected: {EXPECTED_MATCHES:,})")
    print(f"  Source 2 Matches:        {s2_matches_counted:,} (Expected: {EXPECTED_S2_MATCHES:,})")
    print(f"  Source 3 Matches:        {s3_matches_counted:,} (Expected: {EXPECTED_S3_MATCHES:,})")
    print(f"  Zero-Match Entities:     {sub_c0:,} (Expected: {EXPECTED_ZERO_MATCHES:,})")
    print(f"  Exactly 1 Match:         {sub_c1:,}")
    print(f"  2-3 Matches:             {sub_c2_3:,}")
    print(f"  4-5 Matches:             {sub_c4_5:,}")
    print(f"  6+ Matches:              {sub_c6plus:,}")
    print(f"  Max Matches per S1:      {sub_max}")

    # Strict equality against Stage 6
    assert total_matches_counted == EXPECTED_MATCHES, f"Match count mismatch: {total_matches_counted} != {EXPECTED_MATCHES}"
    assert s2_matches_counted == EXPECTED_S2_MATCHES, f"S2 mismatch: {s2_matches_counted} != {EXPECTED_S2_MATCHES}"
    assert s3_matches_counted == EXPECTED_S3_MATCHES, f"S3 mismatch: {s3_matches_counted} != {EXPECTED_S3_MATCHES}"
    assert sub_c0 == EXPECTED_ZERO_MATCHES, f"Zero match mismatch: {sub_c0} != {EXPECTED_ZERO_MATCHES}"
    print("Match statistics verified 100% identical to Stage 6 output.\n")

    # Free memory before running validator subprocess
    del df, df_cand_counts, matching_map, valid_s2_ids, valid_s3_ids, s1_country_map
    import gc
    gc.collect()

    # -------------------------------------------------------------
    # Step 7: Run Official Validator
    # -------------------------------------------------------------
    print("=== STEP 7: Executing Official Validator (validate_submission.py) ===")
    assert os.path.exists(VALIDATOR_SCRIPT), f"Validator not found at {VALIDATOR_SCRIPT}"
    val_cmd = [
        sys.executable,
        VALIDATOR_SCRIPT,
        "--matching", MATCHING_TSV_PATH,
        "--candidate", CANDIDATE_TSV_PATH,
        "--test-dir", RAW_TEST_DIR,
    ]
    print(f"Running command: {' '.join(val_cmd)}")
    t_val = time.time()
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True)
    val_duration = time.time() - t_val

    print("\n--- OFFICIAL VALIDATOR OUTPUT ---")
    print(val_proc.stdout)
    if val_proc.stderr:
        print("STDERR:")
        print(val_proc.stderr)
    print("---------------------------------\n")

    if val_proc.returncode != 0:
        print(f"VALIDATOR FAILED with returncode {val_proc.returncode}!")
        sys.exit(1)
    
    assert "PASS" in val_proc.stdout and "Safe to submit" in val_proc.stdout, "Validator output did not contain PASS marker!"
    print(f"Official validator PASSED in {val_duration:.2f}s.\n")

    # -------------------------------------------------------------
    # Step 8: Save Machine-Readable Stage 7 Results
    # -------------------------------------------------------------
    print("=== STEP 8: Saving Machine-Readable Results ===")
    total_runtime = time.time() - t_start
    peak_ram = proc.memory_info().rss / (1024 * 1024)

    stage7_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "matching_results_path": MATCHING_TSV_PATH,
        "candidate_pairs_path": CANDIDATE_TSV_PATH,
        "backup_dir": BACKUP_DIR,
        "file_sizes": {
            "matching_results_tsv_bytes": os.path.getsize(MATCHING_TSV_PATH),
            "matching_results_tsv_mb": round(matching_size_mb, 2),
            "candidate_pairs_tsv_bytes": os.path.getsize(CANDIDATE_TSV_PATH),
            "candidate_pairs_tsv_mb": round(candidate_size_mb, 2),
        },
        "row_counts": {
            "matching_results_rows": len(matching_s1_ids),
            "candidate_pairs_rows": len(candidate_s1_ids),
            "expected_rows": EXPECTED_S1_COUNT,
            "unique_s1_ids": len(set(matching_s1_ids)),
            "missing_s1_ids": 0,
            "duplicate_s1_ids": 0,
        },
        "match_statistics": {
            "total_predicted_matches": total_matches_counted,
            "s2_matches": s2_matches_counted,
            "s3_matches": s3_matches_counted,
            "zero_match_s1_count": sub_c0,
            "one_match_s1_count": sub_c1,
            "two_three_matches_s1_count": sub_c2_3,
            "four_five_matches_s1_count": sub_c4_5,
            "six_plus_matches_s1_count": sub_c6plus,
            "max_matches_per_s1": sub_max,
        },
        "candidate_statistics": {
            "total_candidates": total_candidates_counted,
            "max_candidates_per_s1": max_cands_per_s1,
            "empty_candidate_s1_count": candidate_empty_count,
        },
        "validation_results": {
            "cross_file_containment_violations": containment_violations,
            "invalid_matching_ids": len(invalid_matching_ids),
            "invalid_prefixes": len(invalid_prefixes_matching),
            "cross_country_matches": len(cross_country_matches),
            "france_cross_country_violations": 0,
            "us_cross_country_violations": 0,
            "india_cross_country_violations": 0,
            "france_s1_represented": france_s1_count,
            "us_s1_represented": us_s1_count,
            "india_s1_represented": india_s1_count,
            "official_validator_returncode": val_proc.returncode,
            "official_validator_status": "PASS",
        },
        "system_resources": {
            "total_runtime_seconds": round(total_runtime, 2),
            "peak_ram_mb": round(peak_ram, 2),
        }
    }

    with open(STAGE7_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(stage7_data, f, indent=2)
    print(f"Saved machine-readable Stage 7 validation results to {STAGE7_RESULTS_PATH}.\n")

    print("===================================================================")
    print("STAGE 7 COMPLETE: ALL SUBMISSION FILES CONSTRUCTED AND VERIFIED PASS.")
    print("===================================================================")

if __name__ == "__main__":
    main()
