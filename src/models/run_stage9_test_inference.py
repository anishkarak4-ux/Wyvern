"""
run_stage9_test_inference.py - Stage 9: Full Test Inference with Validated Portfolio 2
Amazon ML Challenge 2026

Executes full test candidate generation & inference across all 1,732,544 test Source 1 entities:
- Enforces validated Stage 8.6 Portfolio 2 blocking rules (Combo 5 + rare_tok_and_addr_num)
- Uses validated decision threshold tau = 0.90
- Uses production model: data/models/xgb_final_stage5.json
- Uses exact 46-feature production feature extraction pipeline
- Preserves all prior stage artifacts (Stage 6 parquet, Stage 7 submission, production model)
- Saves country checkpoints to data/processed/test_predictions_stage9_{country}.parquet
- Saves combined output to data/processed/test_predictions_stage9.parquet
- Generates new versioned submission: submissions/stage9_v2/
- Runs official utils/validate_submission.py with --check-ids
"""

import os
import sys
import gc
import csv
import time
import json
import psutil
import argparse
import datetime
import subprocess
from collections import defaultdict, Counter
import numpy as np
import polars as pl
import xgboost as xgb

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.production_config import (
    BASE_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    FINAL_MODEL_PATH,
    DECISION_THRESHOLD,
    FEATURE_NAMES,
)
from src.preprocessing.normalization import (
    clean_text,
    strip_legal_suffixes,
    get_first_two_tokens,
    get_name_prefix,
    get_first_significant_word,
    normalize_address,
    extract_address_numeric_tokens,
)
from src.features.pair_features import extract_pair_features, extract_postal_code

# Raw test and train paths
RAW_DIR = os.path.join(DATA_DIR, "raw", "train")
RAW_TEST_DIR = os.path.join(DATA_DIR, "raw", "test")

TEST_S1_PATH = os.path.join(RAW_TEST_DIR, "test_source1.tsv")
TEST_S2_PATH = os.path.join(RAW_TEST_DIR, "test_source2.tsv")
TEST_S3_PATH = os.path.join(RAW_TEST_DIR, "test_source3.tsv")

RAW_BYTE_SIZES = {
    TEST_S1_PATH: 175022086,
    TEST_S2_PATH: 509456422,
    TEST_S3_PATH: 506002772,
    os.path.join(RAW_DIR, "train_ground_truth.tsv"): 127015583,
    os.path.join(RAW_DIR, "train_source1.tsv"): 210069713,
    os.path.join(RAW_DIR, "train_source2.tsv"): 489301488,
    os.path.join(RAW_DIR, "train_source3.tsv"): 503705637,
}

INTERMEDIATE_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "test_predictions_stage9.parquet")
STAGE9_RESULTS_PATH = os.path.join(MODELS_DIR, "stage9_test_inference_results.json")
REPORT_PATH = os.path.join(BASE_DIR, "docs", "stage9_test_inference_report.md")
SUBMISSIONS_V2_DIR = os.path.join(BASE_DIR, "submissions", "stage9_v2")
RARE_TOKENS_PATH = os.path.join(PROCESSED_DIR, "test_catalog_rare_tokens.json")

BATCH_SIZE = 50000

def verify_raw_datasets():
    print("=== VERIFYING RAW DATASETS UNTOUCHED ===")
    for path, expected_sz in RAW_BYTE_SIZES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raw file: {path}")
        actual_sz = os.path.getsize(path)
        if actual_sz != expected_sz:
            raise ValueError(f"CORRUPTION in {path}: expected {expected_sz} bytes, found {actual_sz} bytes!")
    print("All raw datasets verified 100% byte-for-byte untouched.\n")

def get_or_build_rare_tokens(proc):
    """Load or compute catalog rare tokens (DF <= 100) per country across test_source2 and test_source3."""
    if os.path.exists(RARE_TOKENS_PATH):
        print(f"Loading precomputed rare tokens from {RARE_TOKENS_PATH}...")
        t0 = time.time()
        with open(RARE_TOKENS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        rare_tokens_set = {c: set(toks) for c, toks in data.items()}
        print(f"Loaded rare tokens in {time.time() - t0:.2f}s | France: {len(rare_tokens_set.get('France', [])):,}, US: {len(rare_tokens_set.get('US', [])):,}, India: {len(rare_tokens_set.get('India', [])):,} | RAM: {proc.memory_info().rss/(1024*1024):.1f} MB\n")
        return rare_tokens_set

    print(f"Computing test catalog rare tokens across test_source2 and test_source3...")
    t0 = time.time()
    counts = defaultdict(Counter)
    rows = 0

    for path in [TEST_S2_PATH, TEST_S3_PATH]:
        print(f"Scanning {os.path.basename(path)} for token frequencies...")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                rows += 1
                c = r[3].strip() if len(r) > 3 else ""
                norm = clean_text(r[1])
                sn = strip_legal_suffixes(norm)
                toks = set(sn.split() if sn else norm.split())
                for t in toks:
                    if len(t) >= 4:
                        counts[c][t] += 1
                if rows % 2000000 == 0:
                    print(f"  {rows:,} rows scanned ({time.time() - t0:.1f}s, RAM: {proc.memory_info().rss/(1024*1024):.1f} MB)")

    rare_tokens_data = {}
    rare_tokens_set = {}
    for c in counts:
        rare_list = [t for t, cnt in counts[c].items() if cnt <= 100]
        rare_tokens_data[c] = rare_list
        rare_tokens_set[c] = set(rare_list)
        print(f"  {c}: {len(rare_list):,} rare tokens (DF <= 100) out of {len(counts[c]):,} total")

    with open(RARE_TOKENS_PATH, "w", encoding="utf-8") as f:
        json.dump(rare_tokens_data, f)
    print(f"Saved rare tokens to {RARE_TOKENS_PATH} in {time.time() - t0:.1f}s.\n")
    return rare_tokens_set

def make_entity_rep(eid, name, addr, country):
    """Create normalized entity representation with precomputed tokens."""
    c = country.strip() if country else ""
    norm_name = clean_text(name)
    sn = strip_legal_suffixes(norm_name)
    tokens = sn.split() if sn else norm_name.split()
    num_tok = len(tokens)

    tok2 = (tokens[0] + "_" + tokens[1]) if num_tok >= 2 else (tokens[0] if num_tok == 1 else "")
    pfx8 = sn[:8] if len(sn) >= 3 else ""
    fw3 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 3) else ""
    fw4 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 4) else ""
    sorted_tok2 = "_".join(sorted(tokens[:2])) if num_tok >= 1 else ""
    fc = norm_name[0] if norm_name else ""

    norm_addr = normalize_address(addr) if addr else ""
    nums_all = extract_address_numeric_tokens(norm_addr)
    nums_sel = [n for n in nums_all if len(n) >= 3]
    postal = extract_postal_code(norm_addr, c)

    return {
        "entity_id": eid,
        "country": c,
        "norm_name": norm_name,
        "stripped_name": sn,
        "first_two_tokens": tok2,
        "name_pfx8": pfx8,
        "first_word": fw3,
        "first_word4": fw4,
        "sorted_tok2": sorted_tok2,
        "name_tokens": set(norm_name.split()),
        "stripped_tokens": set(tokens),
        "norm_addr": norm_addr,
        "addr_tokens": set(norm_addr.split()) if norm_addr else set(),
        "addr_numeric": nums_all,
        "addr_nums_sel": nums_sel,
        "first_char": fc,
        "postal_code": postal,
    }

def process_country(country_name, s1_rows_country, model, rare_tokens_set, proc):
    """
    Run candidate generation, feature extraction, and XGBoost scoring for a single country.
    Saves country parquet checkpoint.
    """
    country_start_time = time.time()
    country_parquet = os.path.join(PROCESSED_DIR, f"test_predictions_stage9_{country_name}.parquet")
    print(f"\n===================================================================")
    print(f"PROCESSING COUNTRY: {country_name} ({len(s1_rows_country):,} S1 Entities)")
    print(f"===================================================================")

    if os.path.exists(country_parquet):
        print(f"Found existing Stage 9 checkpoint at {country_parquet}. Skipping recomputation.")
        df_existing = pl.read_parquet(country_parquet)
        return df_existing, {
            "country": country_name,
            "s1_count": len(df_existing),
            "candidates": int(df_existing["candidate_count"].sum()),
            "matches": int(df_existing["match_count"].sum()),
            "duration": 0.0,
        }

    # Step 1: Index S1 representations
    t0 = time.time()
    print("Indexing S1 representations for all 8 blocking rules...")
    s1_order = []
    s1_reps = {}
    idx_stripped = defaultdict(list)
    idx_tok2 = defaultdict(list)
    idx_pfx8 = defaultdict(list)
    idx_addr_num = defaultdict(list)
    idx_fw_addr = defaultdict(list)
    idx_sorted_tok2 = defaultdict(list)
    idx_fw4 = defaultdict(list)
    idx_rare_addr = defaultdict(list)

    s1_cands = defaultdict(list)
    s1_matches = defaultdict(list)

    country_rare_set = rare_tokens_set.get(country_name, set())

    for row in s1_rows_country:
        sid, name, addr, c = row
        s1_order.append(sid)
        rep = make_entity_rep(sid, name, addr, c)
        s1_reps[sid] = rep

        sn = rep["stripped_name"]
        tok2 = rep["first_two_tokens"]
        pfx8 = rep["name_pfx8"]
        fc = rep["first_char"]
        fw3 = rep["first_word"]
        fw4 = rep["first_word4"]
        sorted_tok2 = rep["sorted_tok2"]
        nums_sel = rep["addr_nums_sel"]
        stripped_toks = rep["stripped_tokens"]

        # Rule 1: Stripped name
        if sn: idx_stripped[sn].append(sid)
        # Rule 2: First 2 tokens
        if tok2: idx_tok2[tok2].append(sid)
        # Rule 3: Name prefix 8
        if pfx8: idx_pfx8[pfx8].append(sid)
        # Rule 4: Addr num + first char
        if fc:
            for num in nums_sel:
                idx_addr_num[(num, fc)].append(sid)
        # Rule 5: First word + addr num
        if fw3:
            for num in nums_sel:
                idx_fw_addr[(fw3, num)].append(sid)
        # Rule 6: Sorted first 2 tokens
        if sorted_tok2: idx_sorted_tok2[sorted_tok2].append(sid)
        # Rule 7: First word >= 4 chars
        if fw4: idx_fw4[fw4].append(sid)
        # Rule 8: Rare token (DF <= 100, len >= 4) + addr num
        if nums_sel:
            rare_s1 = [t for t in stripped_toks if len(t) >= 4 and t in country_rare_set]
            for t in rare_s1:
                for num in nums_sel:
                    idx_rare_addr[(t, num)].append(sid)

    print(f"Loaded {len(s1_order):,} S1 representations in {time.time() - t0:.2f}s | RAM: {proc.memory_info().rss/(1024*1024):.1f} MB")

    # Step 2: Streaming candidate generation & inference
    bucket_counts_tok2 = Counter()
    bucket_counts_pfx8 = Counter()
    bucket_counts_addr_num = Counter()
    bucket_counts_fw_addr = Counter()
    bucket_counts_sorted_tok2 = Counter()
    bucket_counts_fw4 = Counter()
    bucket_counts_rare_addr = Counter()

    candidate_batch = []
    total_candidates_generated = 0
    total_matches_accepted = 0
    total_s2_candidates = 0
    total_s3_candidates = 0
    total_s2_matches = 0
    total_s3_matches = 0
    accepted_probs = []

    def flush_batch():
        nonlocal candidate_batch, total_matches_accepted, total_s2_matches, total_s3_matches
        if not candidate_batch:
            return
        n = len(candidate_batch)
        X_batch = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        for i in range(n):
            sid, tid, s1r, tr, mask, s3_flag = candidate_batch[i]
            prov = {
                "stripped": bool(mask & 1),
                "tok2": bool(mask & 2),
                "pfx8": bool(mask & 4),
                "addr_num": bool(mask & 8),
            }
            X_batch[i] = extract_pair_features(s1r, tr, prov, is_s3=s3_flag)

        probs = model.predict_proba(X_batch)[:, 1]
        match_idxs = np.where(probs >= DECISION_THRESHOLD)[0]

        for idx in match_idxs:
            sid, tid, _, _, _, s3_flag = candidate_batch[idx]
            p = float(probs[idx])
            s1_matches[sid].append((tid, p, s3_flag))
            accepted_probs.append(p)
            total_matches_accepted += 1
            if s3_flag:
                total_s3_matches += 1
            else:
                total_s2_matches += 1

        candidate_batch.clear()

    # Stream target file function
    def stream_target_file(target_path, is_s3):
        nonlocal total_candidates_generated, total_s2_candidates, total_s3_candidates
        src_label = "Source 3" if is_s3 else "Source 2"
        print(f"Streaming {src_label} for {country_name}...")
        t_src_start = time.time()
        rows_scanned = 0
        pairs_for_file = 0

        with open(target_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4 or parts[3] != country_name:
                    continue
                rows_scanned += 1
                tid, t_name, t_addr, t_c = parts[0], parts[1], parts[2], parts[3]

                norm_name = clean_text(t_name)
                sn = strip_legal_suffixes(norm_name)
                tokens = sn.split() if sn else norm_name.split()
                num_tok = len(tokens)

                tok2 = (tokens[0] + "_" + tokens[1]) if num_tok >= 2 else (tokens[0] if num_tok == 1 else "")
                pfx8 = sn[:8] if len(sn) >= 3 else ""
                fw3 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 3) else ""
                fw4 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 4) else ""
                sorted_tok2 = "_".join(sorted(tokens[:2])) if num_tok >= 1 else ""
                fc = norm_name[0] if norm_name else ""

                if tok2: bucket_counts_tok2[tok2] += 1
                if pfx8: bucket_counts_pfx8[pfx8] += 1
                if sorted_tok2: bucket_counts_sorted_tok2[sorted_tok2] += 1
                if fw4: bucket_counts_fw4[fw4] += 1

                norm_addr = normalize_address(t_addr) if t_addr else ""
                nums_all = extract_address_numeric_tokens(norm_addr)
                nums_sel = [n for n in nums_all if len(n) >= 3]

                matched_s1s = {}

                # Rule 1: Stripped Name (Uncapped)
                if sn and sn in idx_stripped:
                    for sid in idx_stripped[sn]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 1

                # Rule 2: First 2 Tokens (Cap <= 200)
                if tok2 and bucket_counts_tok2[tok2] <= 200 and tok2 in idx_tok2:
                    for sid in idx_tok2[tok2]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 2

                # Rule 3: Name Prefix 8 (Cap <= 200)
                if pfx8 and bucket_counts_pfx8[pfx8] <= 200 and pfx8 in idx_pfx8:
                    for sid in idx_pfx8[pfx8]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 4

                # Rule 4: Address Num + First Char (Cap <= 100)
                if fc:
                    for num in nums_sel:
                        bucket_counts_addr_num[(num, fc)] += 1
                        if bucket_counts_addr_num[(num, fc)] <= 100 and (num, fc) in idx_addr_num:
                            for sid in idx_addr_num[(num, fc)]:
                                matched_s1s[sid] = matched_s1s.get(sid, 0) | 8

                # Rule 5 (Combo 5): First Word + Addr Num (Cap <= 100)
                if fw3:
                    for num in nums_sel:
                        bucket_counts_fw_addr[(fw3, num)] += 1
                        if bucket_counts_fw_addr[(fw3, num)] <= 100 and (fw3, num) in idx_fw_addr:
                            for sid in idx_fw_addr[(fw3, num)]:
                                matched_s1s[sid] = matched_s1s.get(sid, 0) | 16

                # Rule 6 (Combo 5): Sorted First 2 Tokens (Cap <= 200)
                if sorted_tok2 and bucket_counts_sorted_tok2[sorted_tok2] <= 200 and sorted_tok2 in idx_sorted_tok2:
                    for sid in idx_sorted_tok2[sorted_tok2]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 32

                # Rule 7 (Combo 5): First Word >= 4 Chars (Cap <= 50)
                if fw4 and bucket_counts_fw4[fw4] <= 50 and fw4 in idx_fw4:
                    for sid in idx_fw4[fw4]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 64

                # Rule 8 (Stage 8.6 Portfolio 2): Rare Name Token (DF <= 100) + Addr Num (Cap <= 100)
                if nums_sel:
                    t_rare = [t for t in set(tokens) if len(t) >= 4 and t in country_rare_set]
                    for t in t_rare:
                        for num in nums_sel:
                            bucket_counts_rare_addr[(t, num)] += 1
                            if bucket_counts_rare_addr[(t, num)] <= 100 and (t, num) in idx_rare_addr:
                                for sid in idx_rare_addr[(t, num)]:
                                    matched_s1s[sid] = matched_s1s.get(sid, 0) | 128

                if matched_s1s:
                    postal = extract_postal_code(norm_addr, country_name)
                    t_rep = {
                        "entity_id": tid,
                        "country": country_name,
                        "norm_name": norm_name,
                        "stripped_name": sn,
                        "first_two_tokens": tok2,
                        "name_pfx8": pfx8,
                        "first_word": fw3,
                        "first_word4": fw4,
                        "sorted_tok2": sorted_tok2,
                        "name_tokens": set(norm_name.split()),
                        "stripped_tokens": set(tokens),
                        "norm_addr": norm_addr,
                        "addr_tokens": set(norm_addr.split()) if norm_addr else set(),
                        "addr_numeric": nums_all,
                        "addr_nums_sel": nums_sel,
                        "first_char": fc,
                        "postal_code": postal,
                    }
                    for sid, mask in matched_s1s.items():
                        s1_cands[sid].append(tid)
                        candidate_batch.append((sid, tid, s1_reps[sid], t_rep, mask, is_s3))
                        pairs_for_file += 1
                        total_candidates_generated += 1
                        if is_s3:
                            total_s3_candidates += 1
                        else:
                            total_s2_candidates += 1

                    if len(candidate_batch) >= BATCH_SIZE:
                        flush_batch()

                if rows_scanned % 500000 == 0:
                    print(f"  [{src_label} {country_name}] Scanned {rows_scanned:,} rows | Candidates: {pairs_for_file:,} | Matches: {total_matches_accepted:,} | RAM: {proc.memory_info().rss/(1024*1024):.1f} MB")

        flush_batch()
        src_duration = time.time() - t_src_start
        print(f"Completed {src_label} for {country_name}: {rows_scanned:,} rows scanned, {pairs_for_file:,} candidates generated in {src_duration:.2f}s.\n")

    # Run S2 then S3
    stream_target_file(TEST_S2_PATH, is_s3=False)
    stream_target_file(TEST_S3_PATH, is_s3=True)

    # Step 3: Build Country Polars DataFrame
    t_df = time.time()
    print("Constructing Polars DataFrame for country...")
    col_sid = []
    col_country = []
    col_cand_count = []
    col_cand_ids = []
    col_match_count = []
    col_matched_ids = []
    col_match_probs = []
    col_s2_count = []
    col_s3_count = []

    for sid in s1_order:
        col_sid.append(sid)
        col_country.append(country_name)

        cands = s1_cands.get(sid, [])
        col_cand_count.append(len(cands))
        col_cand_ids.append(",".join(cands))

        matches = s1_matches.get(sid, [])
        col_match_count.append(len(matches))
        col_matched_ids.append(",".join(m[0] for m in matches))
        col_match_probs.append(",".join(f"{m[1]:.5f}" for m in matches))
        col_s2_count.append(sum(1 for m in matches if not m[2]))
        col_s3_count.append(sum(1 for m in matches if m[2]))

    df_country = pl.DataFrame({
        "source1_entity_id": col_sid,
        "country": col_country,
        "candidate_count": pl.Series(col_cand_count, dtype=pl.Int32),
        "candidate_entity_ids": col_cand_ids,
        "match_count": pl.Series(col_match_count, dtype=pl.Int32),
        "matched_entity_ids": col_matched_ids,
        "match_probabilities": col_match_probs,
        "s2_match_count": pl.Series(col_s2_count, dtype=pl.Int32),
        "s3_match_count": pl.Series(col_s3_count, dtype=pl.Int32),
    })

    print(f"Built DataFrame in {time.time() - t_df:.2f}s. Writing to {country_parquet}...")
    df_country.write_parquet(country_parquet, compression="zstd")
    file_sz_mb = os.path.getsize(country_parquet) / (1024 * 1024)
    print(f"Saved {country_parquet} ({file_sz_mb:.2f} MB).")

    # Summary
    country_duration = time.time() - country_start_time
    stats = {
        "country": country_name,
        "s1_count": len(s1_order),
        "total_candidates": total_candidates_generated,
        "s2_candidates": total_s2_candidates,
        "s3_candidates": total_s3_candidates,
        "total_matches": total_matches_accepted,
        "s2_matches": total_s2_matches,
        "s3_matches": total_s3_matches,
        "zero_match_s1": sum(1 for c in col_match_count if c == 0),
        "one_match_s1": sum(1 for c in col_match_count if c == 1),
        "two_to_three_match_s1": sum(1 for c in col_match_count if 2 <= c <= 3),
        "four_to_five_match_s1": sum(1 for c in col_match_count if 4 <= c <= 5),
        "six_plus_match_s1": sum(1 for c in col_match_count if c >= 6),
        "max_candidates": int(max(col_cand_count)) if col_cand_count else 0,
        "max_matches": int(max(col_match_count)) if col_match_count else 0,
        "duration_seconds": round(country_duration, 2),
    }

    del s1_reps, idx_stripped, idx_tok2, idx_pfx8, idx_addr_num, idx_fw_addr, idx_sorted_tok2, idx_fw4, idx_rare_addr
    del s1_cands, s1_matches
    gc.collect()

    return df_country, stats

def main():
    parser = argparse.ArgumentParser(description="Stage 9 Full Test Inference")
    parser.add_argument("--country", type=str, choices=["France", "US", "India", "all"], default="all",
                        help="Process a single country or all countries (default: all)")
    args = parser.parse_args()

    t_global_start = time.time()
    proc = psutil.Process(os.getpid())

    print("===================================================================")
    print("STAGE 9: FULL TEST CANDIDATE GENERATION & INFERENCE")
    print(f"Amazon ML Challenge 2026 | Decision Threshold tau = {DECISION_THRESHOLD:.2f}")
    print("===================================================================\n")

    verify_raw_datasets()

    # Step 1: Load Rare Tokens Catalog (DF <= 100)
    rare_tokens_set = get_or_build_rare_tokens(proc)

    # Step 2: Load Model
    print(f"Loading production XGBoost model from {FINAL_MODEL_PATH}...")
    model = xgb.XGBClassifier()
    model.load_model(FINAL_MODEL_PATH)
    print("Loaded model successfully.\n")

    # Step 3: Load S1 entities grouped by country
    print("Loading test Source 1 entities...")
    t_s1 = time.time()
    s1_by_country = defaultdict(list)
    total_s1 = 0

    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for r in reader:
            if len(r) >= 4:
                sid, name, addr, c = r[0], r[1], r[2], r[3].strip()
                s1_by_country[c].append((sid, name, addr, c))
                total_s1 += 1

    print(f"Loaded {total_s1:,} test S1 entities in {time.time() - t_s1:.2f}s:")
    for c, rows in sorted(s1_by_country.items()):
        print(f"  {c:10s}: {len(rows):,} entities")
    print()

    # Determine countries to process
    countries_to_run = ["France", "US", "India"] if args.country == "all" else [args.country]

    country_dfs = {}
    country_stats = {}

    for c in countries_to_run:
        df_c, stats_c = process_country(c, s1_by_country[c], model, rare_tokens_set, proc)
        country_dfs[c] = df_c
        country_stats[c] = stats_c

    # If all countries were processed or checkpoints exist for all 3:
    all_checkpoints_exist = all(
        os.path.exists(os.path.join(PROCESSED_DIR, f"test_predictions_stage9_{c}.parquet"))
        for c in ["France", "US", "India"]
    )

    if all_checkpoints_exist:
        print("\n===================================================================")
        print("CONSOLIDATING ALL COUNTRY PREDICTIONS INTO test_predictions_stage9.parquet")
        print("===================================================================")
        
        dfs_to_concat = []
        for c in ["France", "US", "India"]:
            c_parquet = os.path.join(PROCESSED_DIR, f"test_predictions_stage9_{c}.parquet")
            print(f"Reading {c_parquet}...")
            dfs_to_concat.append(pl.read_parquet(c_parquet))

        final_df = pl.concat(dfs_to_concat)
        print(f"Combined total S1 rows: {len(final_df):,}. Writing to {INTERMEDIATE_OUTPUT_PATH}...")
        final_df.write_parquet(INTERMEDIATE_OUTPUT_PATH, compression="zstd")
        parquet_sz_mb = os.path.getsize(INTERMEDIATE_OUTPUT_PATH) / (1024 * 1024)
        print(f"Saved {INTERMEDIATE_OUTPUT_PATH} ({parquet_sz_mb:.2f} MB).\n")

        # Global Statistics
        total_candidates = int(final_df["candidate_count"].sum())
        total_matches = int(final_df["match_count"].sum())
        total_s2_matches = int(final_df["s2_match_count"].sum())
        total_s3_matches = int(final_df["s3_match_count"].sum())
        avg_cands_per_s1 = total_candidates / len(final_df)
        avg_matches_per_s1 = total_matches / len(final_df)

        zero_matches = int((final_df["match_count"] == 0).sum())
        one_match = int((final_df["match_count"] == 1).sum())
        two_three_matches = int(((final_df["match_count"] >= 2) & (final_df["match_count"] <= 3)).sum())
        four_five_matches = int(((final_df["match_count"] >= 4) & (final_df["match_count"] <= 5)).sum())
        six_plus_matches = int((final_df["match_count"] >= 6).sum())

        max_cands = int(final_df["candidate_count"].max())
        max_matches = int(final_df["match_count"].max())

        print("=== GLOBAL INFERENCE METRICS ===")
        print(f"Total S1 Entities:             {len(final_df):,}")
        print(f"Total Candidates Generated:     {total_candidates:,} (Mean: {avg_cands_per_s1:.2f}/S1, Max: {max_cands})")
        print(f"Total Matches Accepted (>=0.90):{total_matches:,} (Mean: {avg_matches_per_s1:.2f}/S1, Max: {max_matches})")
        print(f"  Source 2 Matches:             {total_s2_matches:,} ({total_s2_matches/total_matches*100:.2f}%)")
        print(f"  Source 3 Matches:             {total_s3_matches:,} ({total_s3_matches/total_matches*100:.2f}%)")
        print(f"Match Cardinality Distribution:")
        print(f"  Zero Matches (Singletons):    {zero_matches:,} ({zero_matches/len(final_df)*100:.2f}%)")
        print(f"  1 Match:                      {one_match:,} ({one_match/len(final_df)*100:.2f}%)")
        print(f"  2-3 Matches:                  {two_three_matches:,} ({two_three_matches/len(final_df)*100:.2f}%)")
        print(f"  4-5 Matches:                  {four_five_matches:,} ({four_five_matches/len(final_df)*100:.2f}%)")
        print(f"  6+ Matches:                   {six_plus_matches:,} ({six_plus_matches/len(final_df)*100:.2f}%)\n")

        # Step 4: Construct Versioned Submission Files in submissions/stage9_v2/
        print("===================================================================")
        print("CONSTRUCTING VERSIONED SUBMISSION FILES IN submissions/stage9_v2/")
        print("===================================================================")
        os.makedirs(SUBMISSIONS_V2_DIR, exist_ok=True)
        sub_matching_path = os.path.join(SUBMISSIONS_V2_DIR, "matching_results.tsv")
        sub_candidate_path = os.path.join(SUBMISSIONS_V2_DIR, "candidate_pairs.tsv")

        t_sub = time.time()
        print(f"Writing {sub_matching_path}...")
        final_df.select(["source1_entity_id", "matched_entity_ids"]).write_csv(
            sub_matching_path, separator="\t", quote_style="never"
        )

        print(f"Writing {sub_candidate_path}...")
        final_df.select(["source1_entity_id", "candidate_entity_ids"]).write_csv(
            sub_candidate_path, separator="\t", quote_style="never"
        )

        print(f"Submission files written in {time.time() - t_sub:.2f}s:")
        print(f"  {sub_matching_path}: {os.path.getsize(sub_matching_path)/(1024*1024):.2f} MB")
        print(f"  {sub_candidate_path}: {os.path.getsize(sub_candidate_path)/(1024*1024):.2f} MB\n")

        # Step 5: Comprehensive Cross-File Subset Containment Verification
        print("===================================================================")
        print("CROSS-FILE SUBSET CONTAINMENT & SAFETY VERIFICATION")
        print("===================================================================")
        t_ver = time.time()
        subset_failures = 0
        duplicate_s1_errors = 0
        s1_seen = set()

        for sid, c_str, m_str in zip(
            final_df["source1_entity_id"],
            final_df["candidate_entity_ids"],
            final_df["matched_entity_ids"]
        ):
            if sid in s1_seen:
                duplicate_s1_errors += 1
            s1_seen.add(sid)

            if m_str:
                m_set = set(m_str.split(","))
                c_set = set(c_str.split(",")) if c_str else set()
                if not m_set.issubset(c_set):
                    subset_failures += 1

        assert duplicate_s1_errors == 0, f"Found {duplicate_s1_errors} duplicate S1 entities!"
        assert subset_failures == 0, f"Found {subset_failures} entities where matches were NOT subset of candidates!"
        assert len(s1_seen) == 1732544, f"Entity count mismatch: {len(s1_seen)} != 1,732,544!"
        print(f"Cross-file verification passed 100% in {time.time() - t_ver:.2f}s.")
        print(">> matches(S1) [subset of] candidates(S1) verified for all 1,732,544 entities. <<\n")

        # Step 6: Run Official Submission Validator
        print("===================================================================")
        print("RUNNING OFFICIAL SUBMISSION VALIDATOR (utils/validate_submission.py --check-ids)")
        print("===================================================================")
        validator_script = os.path.join(BASE_DIR, "utils", "validate_submission.py")
        cmd = [
            sys.executable,
            validator_script,
            "--matching", sub_matching_path,
            "--candidate", sub_candidate_path,
            "--test-dir", RAW_TEST_DIR,
            "--check-ids",
        ]
        val_res = subprocess.run(cmd, capture_output=True, text=True)
        print(val_res.stdout)
        if val_res.stderr:
            print("Validator STDERR:", val_res.stderr)
        assert val_res.returncode == 0, f"Official validator FAILED with exit code {val_res.returncode}!"
        print(">> OFFICIAL SUBMISSION VALIDATOR PASSED WITH ZERO ERRORS (EXIT CODE 0) <<\n")

        # Step 7: Compare Against Stage 7 Baseline Statistics
        # Stage 7 recorded metrics:
        stage7_candidates = 345097457
        stage7_matches = 4453530
        stage7_s2_matches = 2174384
        stage7_s3_matches = 2279146

        cand_delta = total_candidates - stage7_candidates
        match_delta = total_matches - stage7_matches

        print("=== STAGE 9 vs STAGE 7 COMPARISON ===")
        print(f"Candidates: Stage 9 = {total_candidates:,} | Stage 7 = {stage7_candidates:,} | Delta = {cand_delta:+,} ({cand_delta/stage7_candidates*100:+.2f}%)")
        print(f"Matches:    Stage 9 = {total_matches:,} | Stage 7 = {stage7_matches:,} | Delta = {match_delta:+,} ({match_delta/stage7_matches*100:+.2f}%)")
        print(f"Threshold:  Stage 9 = {DECISION_THRESHOLD:.2f} | Stage 7 = 0.91\n")

        # Save Machine-Readable JSON
        total_pipeline_time = time.time() - t_global_start
        stage9_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "decision_threshold": DECISION_THRESHOLD,
            "total_s1_entities": len(final_df),
            "total_candidates": total_candidates,
            "avg_candidates_per_s1": round(avg_cands_per_s1, 2),
            "max_candidates": max_cands,
            "total_matches": total_matches,
            "avg_matches_per_s1": round(avg_matches_per_s1, 2),
            "max_matches": max_matches,
            "source2_matches": total_s2_matches,
            "source3_matches": total_s3_matches,
            "match_cardinality_distribution": {
                "zero_matches": zero_matches,
                "one_match": one_match,
                "two_three_matches": two_three_matches,
                "four_five_matches": four_five_matches,
                "six_plus_matches": six_plus_matches,
            },
            "comparison_with_stage7": {
                "stage7_candidates": stage7_candidates,
                "candidate_delta": cand_delta,
                "candidate_delta_pct": round(cand_delta / stage7_candidates * 100, 2),
                "stage7_matches": stage7_matches,
                "match_delta": match_delta,
                "match_delta_pct": round(match_delta / stage7_matches * 100, 2),
            },
            "country_breakdown": country_stats,
            "official_validation": {
                "validator_exit_code": val_res.returncode,
                "validator_passed": True,
                "check_ids_passed": True,
                "subset_containment_passed": True,
            },
            "submission_artifacts": {
                "matching_results_path": sub_matching_path,
                "candidate_pairs_path": sub_candidate_path,
                "intermediate_parquet_path": INTERMEDIATE_OUTPUT_PATH,
            },
            "total_duration_seconds": round(total_pipeline_time, 2),
        }

        with open(STAGE9_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(stage9_results, f, indent=2)
        print(f"Saved machine-readable results to {STAGE9_RESULTS_PATH}.\n")

        # Verify raw datasets one last time
        verify_raw_datasets()

        print("===================================================================")
        print(f"STAGE 9 TEST INFERENCE & VALIDATION COMPLETE in {total_pipeline_time/60:.1f} minutes.")
        print("===================================================================")

if __name__ == "__main__":
    main()
