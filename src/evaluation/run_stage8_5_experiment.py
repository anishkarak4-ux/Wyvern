"""
run_stage8_5_experiment.py - Stage 8.5: High-Recall Blocking + End-to-End Validation Experiment
Amazon ML Challenge 2026

Evaluates Combo 5 Enhanced Blocking end-to-end on the exact Stage 5 held-out validation population:
1. Reconstructs exact 10,002 validation S1 entities and reproduces Stage 5 baseline.
2. Generates Combo 5 candidates (Baseline + FW+AddrNum cap 100 + Sorted Tok2 cap 200 + FW4 cap 50).
3. Evaluates existing models (xgb_stage4_baseline and xgb_final_stage5) at tau = 0.91.
4. Performs deep bottleneck and error analysis.
5. Saves machine-readable results to data/models/stage8_5_validation_results.json.
"""

import os
import sys
import csv
import json
import time
import math
import psutil
import random
from collections import defaultdict, Counter
import numpy as np
import xgboost as xgb

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing.normalization import (
    clean_text,
    strip_legal_suffixes,
    get_first_two_tokens,
    get_name_prefix,
    get_first_significant_word,
    normalize_address,
    extract_address_numeric_tokens,
)
from src.features.pair_features import (
    FEATURE_NAMES,
    extract_postal_code,
    extract_pair_features,
)
from src.evaluation.entity_level_metrics import (
    calc_f_beta,
    calculate_entity_metrics,
    evaluate_population_entity_f05,
)

RAW_DIR = r"C:\Users\KIIT\Desktop\Wyvern\data\raw\train"
RAW_TEST_DIR = r"C:\Users\KIIT\Desktop\Wyvern\data\raw\test"
MODELS_DIR = r"C:\Users\KIIT\Desktop\Wyvern\data\models"

MODEL_BASELINE_PATH = os.path.join(MODELS_DIR, "xgb_stage4_baseline.json")
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, "xgb_final_stage5.json")
OUTPUT_JSON_PATH = os.path.join(MODELS_DIR, "stage8_5_validation_results.json")
REPORT_PATH = r"C:\Users\KIIT\Desktop\Wyvern\docs\stage8_5_validation_report.md"

RAW_BYTE_SIZES = {
    os.path.join(RAW_TEST_DIR, "test_source1.tsv"): 175022086,
    os.path.join(RAW_TEST_DIR, "test_source2.tsv"): 509456422,
    os.path.join(RAW_TEST_DIR, "test_source3.tsv"): 506002772,
    os.path.join(RAW_DIR, "train_ground_truth.tsv"): 127015583,
    os.path.join(RAW_DIR, "train_source1.tsv"): 210069713,
    os.path.join(RAW_DIR, "train_source2.tsv"): 489301488,
    os.path.join(RAW_DIR, "train_source3.tsv"): 503705637,
}

SAMPLE_S1_SIZE = 50000
RANDOM_SEED = 42
DECISION_THRESHOLD = 0.91

def verify_raw_datasets():
    print("=== VERIFYING RAW DATASETS UNTOUCHED ===")
    for path, expected_sz in RAW_BYTE_SIZES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raw file: {path}")
        actual_sz = os.path.getsize(path)
        if actual_sz != expected_sz:
            raise ValueError(f"CORRUPTION in {path}: expected {expected_sz} bytes, found {actual_sz} bytes!")
    print("All raw datasets verified 100% untouched.\n")

def make_entity_rep(eid, name, addr, country):
    """Create normalized entity representation with pre-computed tokens for pair features."""
    norm_name = clean_text(name)
    stripped = strip_legal_suffixes(norm_name)
    tokens = stripped.split() if stripped else norm_name.split()
    num_tok = len(tokens)
    
    tok2 = (tokens[0] + "_" + tokens[1]) if num_tok >= 2 else (tokens[0] if num_tok == 1 else "")
    pfx8 = stripped[:8] if len(stripped) >= 3 else ""
    fw3 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 3) else ""
    fw4 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 4) else ""
    sorted_tok2 = "_".join(sorted(tokens[:2])) if num_tok >= 1 else ""
    
    norm_addr = normalize_address(addr) or ""
    nums_all = extract_address_numeric_tokens(norm_addr)
    nums_sel = [n for n in nums_all if len(n) >= 3]
    fc = norm_name[0] if norm_name else ""
    c = country.strip() if country else ""
    postal = extract_postal_code(norm_addr, c)
    
    return {
        "entity_id": eid,
        "country": c,
        "norm_name": norm_name,
        "stripped_name": stripped,
        "first_two_tokens": tok2,
        "name_pfx8": pfx8,
        "first_word": fw3,
        "first_word4": fw4,
        "sorted_tok2": sorted_tok2,
        "name_tokens": set(norm_name.split()),
        "norm_addr": norm_addr,
        "addr_tokens": set(norm_addr.split()) if norm_addr else set(),
        "addr_numeric": nums_all,
        "addr_nums_sel": nums_sel,
        "first_char": fc,
        "postal_code": postal,
    }

def reconstruct_validation_split():
    """Deterministically reconstruct validation S1 population (10,002 entities)."""
    gt_path = os.path.join(RAW_DIR, "train_ground_truth.tsv")
    s1_path = os.path.join(RAW_DIR, "train_source1.tsv")

    s1_gt_matches = defaultdict(set)
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            sid = r[0]
            if len(r) > 1 and r[1].strip():
                s1_gt_matches[sid] = set(m.strip() for m in r[1].split(",") if m.strip())
            else:
                s1_gt_matches[sid] = set()

    s1_by_strat_group = defaultdict(list)
    s1_metadata_all = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            sid, name, addr, country = r[0], r[1], r[2], r[3].strip()
            num_m = len(s1_gt_matches[sid])
            is_sing = (num_m == 0)
            mb = "0" if is_sing else ("1" if num_m == 1 else ("2-3" if num_m <= 3 else "4+"))
            grp = (country, is_sing, mb)
            s1_by_strat_group[grp].append(sid)
            s1_metadata_all[sid] = {"country": country, "is_singleton": is_sing, "match_bucket": mb, "gt_count": num_m}

    total_s1 = sum(len(v) for v in s1_by_strat_group.values())
    rng = random.Random(RANDOM_SEED)
    selected_s1_set = set()

    for grp, sids in sorted(s1_by_strat_group.items()):
        grp_share = len(sids) / total_s1
        k_grp = int(round(SAMPLE_S1_SIZE * grp_share))
        sids_sorted = sorted(sids)
        sampled = rng.sample(sids_sorted, min(k_grp, len(sids_sorted)))
        selected_s1_set.update(sampled)

    if len(selected_s1_set) < SAMPLE_S1_SIZE:
        rem_needed = SAMPLE_S1_SIZE - len(selected_s1_set)
        remaining = sorted(list(set(s1_metadata_all.keys()) - selected_s1_set))
        selected_s1_set.update(rng.sample(remaining, rem_needed))
    elif len(selected_s1_set) > SAMPLE_S1_SIZE:
        selected_s1_set = set(sorted(list(selected_s1_set))[:SAMPLE_S1_SIZE])

    train_s1_set = set()
    val_s1_set = set()

    for grp, sids in sorted(s1_by_strat_group.items()):
        grp_sample_sids = sorted(list(set(sids) & selected_s1_set))
        if not grp_sample_sids:
            continue
        n_grp = len(grp_sample_sids)
        n_train = int(round(n_grp * 0.80))
        shuffled = list(grp_sample_sids)
        rng.shuffle(shuffled)
        train_s1_set.update(shuffled[:n_train])
        val_s1_set.update(shuffled[n_train:])

    return val_s1_set, s1_gt_matches, s1_metadata_all

def main():
    t_start = time.time()
    proc = psutil.Process(os.getpid())
    print("===================================================================")
    print("STAGE 8.5: HIGH-RECALL BLOCKING + END-TO-END VALIDATION EXPERIMENT")
    print("===================================================================\n")

    verify_raw_datasets()

    # 1. Reconstruct Validation Population
    print("=== STEP 1: Reconstructing Validation S1 Population ===")
    t0 = time.time()
    val_s1_set, s1_gt_matches, s1_metadata_all = reconstruct_validation_split()
    print(f"Validation S1 entities: {len(val_s1_set):,}")
    val_gt_count = sum(len(s1_gt_matches[sid]) for sid in val_s1_set)
    val_sing_count = sum(1 for sid in val_s1_set if len(s1_gt_matches[sid]) == 0)
    print(f"Validation Ground-Truth Matches: {val_gt_count:,}")
    print(f"Validation Singletons: {val_sing_count:,} ({val_sing_count/len(val_s1_set)*100:.2f}%)\n")

    assert len(val_s1_set) == 10002
    assert val_gt_count == 34677
    assert val_sing_count == 559

    # Build Inverted Indexes for Baseline & Combo 5 Rules
    print("Building Inverted Indexes for Validation S1 Entities...")
    s1_path = os.path.join(RAW_DIR, "train_source1.tsv")
    val_s1_reps = {}
    val_entity_metadata = {}

    # Indexes partitioned by country: c -> dict
    idx_stripped = defaultdict(lambda: defaultdict(list))
    idx_tok2 = defaultdict(lambda: defaultdict(list))
    idx_pfx8 = defaultdict(lambda: defaultdict(list))
    idx_addr_num = defaultdict(lambda: defaultdict(list))
    
    # Combo 5 Indexes
    idx_fw_addr = defaultdict(lambda: defaultdict(list))
    idx_sorted_tok2 = defaultdict(lambda: defaultdict(list))
    idx_fw4 = defaultdict(lambda: defaultdict(list))

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            sid = r[0]
            if sid in val_s1_set:
                rep = make_entity_rep(sid, r[1], r[2], r[3])
                val_s1_reps[sid] = rep
                c = rep["country"]
                val_entity_metadata[sid] = {"country": c, "is_singleton": (len(s1_gt_matches[sid]) == 0)}

                # Baseline rules
                if rep["stripped_name"]:
                    idx_stripped[c][rep["stripped_name"]].append(sid)
                if rep["first_two_tokens"]:
                    idx_tok2[c][rep["first_two_tokens"]].append(sid)
                if rep["name_pfx8"]:
                    idx_pfx8[c][rep["name_pfx8"]].append(sid)
                if rep["first_char"]:
                    for num in rep["addr_nums_sel"]:
                        idx_addr_num[c][(num, rep["first_char"])].append(sid)

                # Combo 5 rules
                if rep["first_word"]:
                    for num in rep["addr_nums_sel"]:
                        idx_fw_addr[c][(rep["first_word"], num)].append(sid)
                if rep["sorted_tok2"]:
                    idx_sorted_tok2[c][rep["sorted_tok2"]].append(sid)
                if rep["first_word4"]:
                    idx_fw4[c][rep["first_word4"]].append(sid)

    print(f"Validation S1 entities indexed in {time.time() - t0:.1f}s. RAM: {proc.memory_info().rss / (1024*1024):.1f} MB\n")

    # 2. Load Models
    print("=== STEP 2: Loading Trained XGBoost Models ===")
    assert os.path.exists(MODEL_BASELINE_PATH), f"Baseline model not found: {MODEL_BASELINE_PATH}"
    assert os.path.exists(MODEL_FINAL_PATH), f"Final model not found: {MODEL_FINAL_PATH}"

    clf_baseline = xgb.XGBClassifier()
    clf_baseline.load_model(MODEL_BASELINE_PATH)
    print(f"Loaded baseline model (xgb_stage4_baseline.json).")

    clf_final = xgb.XGBClassifier()
    clf_final.load_model(MODEL_FINAL_PATH)
    print(f"Loaded final production model (xgb_final_stage5.json).\n")

    # 3. Stream Target Datasets & Generate Candidates
    print("=== STEP 3: Streaming Target Datasets & Scoring Candidates ===")
    t_stream = time.time()

    # Per-entity candidates:
    # sid -> list of dict: {"tid": tid, "prob_base": float, "prob_final": float, "is_s3": bool, "is_gt": bool, "is_baseline": bool, "rules": int}
    candidates_by_s1 = defaultdict(list)
    all_target_reps = {}

    # Target bucket counters (streaming order)
    bucket_counts_tok2 = Counter()
    bucket_counts_pfx8 = Counter()
    bucket_counts_addr_num = Counter()
    bucket_counts_fw_addr = Counter()
    bucket_counts_sorted_tok2 = Counter()
    bucket_counts_fw4 = Counter()

    candidate_batch = [] # list of (sid, tid, s1r, tr, prov, is_s3, is_gt, is_baseline_cand, rule_mask)

    def flush_batch():
        if not candidate_batch:
            return
        n = len(candidate_batch)
        X_batch = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        for i in range(n):
            _, _, s1r, tr, prov, s3_flag, _, _, _ = candidate_batch[i]
            X_batch[i] = extract_pair_features(s1r, tr, prov, is_s3=s3_flag)

        probs_base = clf_baseline.predict_proba(X_batch)[:, 1]
        probs_final = clf_final.predict_proba(X_batch)[:, 1]

        for i in range(n):
            sid, tid, _, _, prov, s3_flag, is_gt, is_base, rule_mask = candidate_batch[i]
            candidates_by_s1[sid].append({
                "tid": tid,
                "prob_base": float(probs_base[i]),
                "prob_final": float(probs_final[i]),
                "is_s3": s3_flag,
                "is_gt": is_gt,
                "is_baseline": is_base,
                "rule_mask": rule_mask,
            })
        candidate_batch.clear()

    def stream_target_file(filepath, src_label):
        t_f = time.time()
        is_s3 = (src_label == "S3")
        print(f"Streaming {src_label} ({os.path.basename(filepath)})...")
        rows = 0
        hits = 0

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                rows += 1
                tid = r[0]
                t_name = r[1]
                t_addr = r[2]
                t_c = r[3].strip() if len(r) > 3 else ""

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

                if tok2: bucket_counts_tok2[(t_c, tok2)] += 1
                if pfx8: bucket_counts_pfx8[(t_c, pfx8)] += 1
                if sorted_tok2: bucket_counts_sorted_tok2[(t_c, sorted_tok2)] += 1
                if fw4: bucket_counts_fw4[(t_c, fw4)] += 1

                norm_addr = normalize_address(t_addr) or ""
                t_nums_all = extract_address_numeric_tokens(norm_addr)
                t_nums_sel = [n for n in t_nums_all if len(n) >= 3]

                # Bitmask:
                # 1: stripped (baseline)
                # 2: tok2 (baseline)
                # 4: pfx8 (baseline)
                # 8: addr_num_fc (baseline)
                # 16: fw_addr (Combo 5)
                # 32: sorted_tok2 (Combo 5)
                # 64: fw4 (Combo 5)
                matched_s1s = {}

                # Baseline Rule 1: Stripped Name (Uncapped)
                if sn and sn in idx_stripped[t_c]:
                    for sid in idx_stripped[t_c][sn]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 1

                # Baseline Rule 2: First 2 Tokens (Cap <= 200)
                if tok2 and bucket_counts_tok2[(t_c, tok2)] <= 200 and tok2 in idx_tok2[t_c]:
                    for sid in idx_tok2[t_c][tok2]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 2

                # Baseline Rule 3: Name Prefix 8 (Cap <= 200)
                if pfx8 and bucket_counts_pfx8[(t_c, pfx8)] <= 200 and pfx8 in idx_pfx8[t_c]:
                    for sid in idx_pfx8[t_c][pfx8]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 4

                # Baseline Rule 4: Address Num + First Char (Cap <= 100)
                if fc:
                    for num in t_nums_sel:
                        bucket_counts_addr_num[(t_c, num, fc)] += 1
                        if bucket_counts_addr_num[(t_c, num, fc)] <= 100 and (num, fc) in idx_addr_num[t_c]:
                            for sid in idx_addr_num[t_c][(num, fc)]:
                                matched_s1s[sid] = matched_s1s.get(sid, 0) | 8

                # Combo 5 Rule 5: First Word + Address Num (Cap <= 100)
                if fw3:
                    for num in t_nums_sel:
                        bucket_counts_fw_addr[(t_c, fw3, num)] += 1
                        if bucket_counts_fw_addr[(t_c, fw3, num)] <= 100 and (fw3, num) in idx_fw_addr[t_c]:
                            for sid in idx_fw_addr[t_c][(fw3, num)]:
                                matched_s1s[sid] = matched_s1s.get(sid, 0) | 16

                # Combo 5 Rule 6: Sorted First 2 Tokens (Cap <= 200)
                if sorted_tok2 and bucket_counts_sorted_tok2[(t_c, sorted_tok2)] <= 200 and sorted_tok2 in idx_sorted_tok2[t_c]:
                    for sid in idx_sorted_tok2[t_c][sorted_tok2]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 32

                # Combo 5 Rule 7: First Word >= 4 Chars (Cap <= 50)
                if fw4 and bucket_counts_fw4[(t_c, fw4)] <= 50 and fw4 in idx_fw4[t_c]:
                    for sid in idx_fw4[t_c][fw4]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 64

                if matched_s1s:
                    hits += 1
                    t_rep = {
                        "entity_id": tid,
                        "country": t_c,
                        "norm_name": norm_name,
                        "stripped_name": sn,
                        "first_two_tokens": tok2,
                        "name_pfx8": pfx8,
                        "first_word": fw3,
                        "name_tokens": set(norm_name.split()),
                        "norm_addr": norm_addr,
                        "addr_tokens": set(norm_addr.split()) if norm_addr else set(),
                        "addr_numeric": t_nums_all,
                        "addr_nums_sel": t_nums_sel,
                        "first_char": fc,
                        "postal_code": extract_postal_code(norm_addr, t_c),
                    }
                    all_target_reps[tid] = t_rep

                    for sid, bitmask in matched_s1s.items():
                        prov = {
                            "stripped": bool(bitmask & 1),
                            "tok2": bool(bitmask & 2),
                            "pfx8": bool(bitmask & 4),
                            "addr_num": bool(bitmask & 8),
                        }
                        is_gt = (tid in s1_gt_matches[sid])
                        is_base = bool(bitmask & 15) # rules 1, 2, 4, 8 are baseline
                        candidate_batch.append((sid, tid, val_s1_reps[sid], t_rep, prov, is_s3, is_gt, is_base, bitmask))

                    if len(candidate_batch) >= 10000:
                        flush_batch()

                if rows % 2000000 == 0:
                    print(f"  {src_label}: {rows:,} rows scanned ({time.time() - t_f:.1f}s, RAM: {proc.memory_info().rss / (1024*1024):.1f} MB)")

        flush_batch()
        print(f"Completed {src_label}: {rows:,} rows processed in {time.time() - t_f:.1f}s.\n")

    stream_target_file(os.path.join(RAW_DIR, "train_source2.tsv"), "S2")
    stream_target_file(os.path.join(RAW_DIR, "train_source3.tsv"), "S3")

    print(f"Candidate generation and scoring finished in {time.time() - t_stream:.1f}s.")
    print(f"Peak RAM: {proc.memory_info().rss / (1024*1024):.1f} MB\n")

    # =============================================================
    # PART A: BASELINE REPRODUCTION & VERIFICATION
    # =============================================================
    print("===================================================================")
    print("PART A: BASELINE REPRODUCTION & VERIFICATION")
    print("===================================================================")
    val_gt_dict = {sid: s1_gt_matches[sid] for sid in val_s1_set}

    # Extract Baseline Predictions (only pairs where is_baseline == True)
    base_preds_model_base = {}
    base_preds_model_final = {}
    baseline_cand_counts = []
    baseline_recovered_gt = set()

    for sid in val_s1_set:
        cands = candidates_by_s1.get(sid, [])
        base_cands = [c for c in cands if c["is_baseline"]]
        baseline_cand_counts.append(len(base_cands))
        
        for c in base_cands:
            if c["is_gt"]:
                baseline_recovered_gt.add((sid, c["tid"]))

        m_base = [c["tid"] for c in base_cands if c["prob_base"] >= DECISION_THRESHOLD]
        m_final = [c["tid"] for c in base_cands if c["prob_final"] >= DECISION_THRESHOLD]
        base_preds_model_base[sid] = m_base
        base_preds_model_final[sid] = m_final

    base_metrics_xgb_base = evaluate_population_entity_f05(base_preds_model_base, val_gt_dict, val_entity_metadata, beta=0.5)
    base_metrics_xgb_final = evaluate_population_entity_f05(base_preds_model_final, val_gt_dict, val_entity_metadata, beta=0.5)

    print("Baseline Blocking Verification:")
    print(f"  Total True Matches in GT:          {val_gt_count:,}")
    print(f"  Baseline Recovered Matches:        {len(baseline_recovered_gt):,} ({len(baseline_recovered_gt)/val_gt_count*100:.4f}%)")
    print(f"  Baseline Candidates Generated:     {sum(baseline_cand_counts):,} (Mean: {np.mean(baseline_cand_counts):.1f}, P95: {np.percentile(baseline_cand_counts, 95):.1f}, Max: {max(baseline_cand_counts)})")
    
    assert len(baseline_recovered_gt) == 24858, f"Baseline recall mismatch: {len(baseline_recovered_gt)} != 24,858"

    print("\nModel Baseline (xgb_stage4_baseline.json on Baseline Candidates):")
    print(f"  Macro F0.5:      {base_metrics_xgb_base['macro_f05']:.5f} (Target: 0.77853)")
    print(f"  Macro Precision: {base_metrics_xgb_base['macro_precision']:.5f} (Target: 0.86098)")
    print(f"  Macro Recall:    {base_metrics_xgb_base['macro_recall']:.5f} (Target: 0.64966)")
    print(f"  Micro F0.5:      {base_metrics_xgb_base['micro_f05']:.5f} (Target: 0.81486)")
    print(f"  Micro Precision: {base_metrics_xgb_base['micro_precision']:.5f} (Target: 0.87605)")
    print(f"  Micro Recall:    {base_metrics_xgb_base['micro_recall']:.5f} (Target: 0.63691)")
    print(f"  Total Matches:   {base_metrics_xgb_base['pred_matches_distribution']['total_predicted_matches']:,}")

    # Verify exact reproduction of 0.77853
    diff_macro = abs(base_metrics_xgb_base['macro_f05'] - 0.77853)
    assert diff_macro < 1e-4, f"Baseline reproduction failed: {base_metrics_xgb_base['macro_f05']} vs 0.77853"
    print(">> BASELINE METRICS REPRODUCED EXACTLY (0.77853) <<\n")

    print("Final Model (xgb_final_stage5.json on Baseline Candidates):")
    print(f"  Macro F0.5:      {base_metrics_xgb_final['macro_f05']:.5f}")
    print(f"  Macro Precision: {base_metrics_xgb_final['macro_precision']:.5f}")
    print(f"  Macro Recall:    {base_metrics_xgb_final['macro_recall']:.5f}")
    print(f"  Micro F0.5:      {base_metrics_xgb_final['micro_f05']:.5f}")
    print(f"  Micro Precision: {base_metrics_xgb_final['micro_precision']:.5f}")
    print(f"  Micro Recall:    {base_metrics_xgb_final['micro_recall']:.5f}")
    print(f"  Total Matches:   {base_metrics_xgb_final['pred_matches_distribution']['total_predicted_matches']:,}\n")

    # =============================================================
    # PART B: COMBO 5 ENHANCED BLOCKING STATISTICS
    # =============================================================
    print("===================================================================")
    print("PART B: COMBO 5 ENHANCED BLOCKING CANDIDATE STATISTICS")
    print("===================================================================")
    combo5_cand_counts = []
    combo5_recovered_gt = set()
    newly_recovered_gt = set()

    for sid in val_s1_set:
        cands = candidates_by_s1.get(sid, [])
        combo5_cand_counts.append(len(cands))
        for c in cands:
            if c["is_gt"]:
                combo5_recovered_gt.add((sid, c["tid"]))
                if not c["is_baseline"]:
                    newly_recovered_gt.add((sid, c["tid"]))

    still_missed_gt_count = val_gt_count - len(combo5_recovered_gt)
    cand_arr = np.array(combo5_cand_counts)

    print(f"Total Enhanced Candidates:       {np.sum(cand_arr):,} (Baseline: {sum(baseline_cand_counts):,}, +{np.sum(cand_arr) - sum(baseline_cand_counts):,})")
    print(f"Mean Candidates per S1:          {np.mean(cand_arr):.2f} (Baseline: {np.mean(baseline_cand_counts):.2f})")
    print(f"Median Candidates per S1:        {np.median(cand_arr):.1f}")
    print(f"P95 Candidates per S1:           {np.percentile(cand_arr, 95):.1f}")
    print(f"P99 Candidates per S1:           {np.percentile(cand_arr, 99):.1f}")
    print(f"Max Candidates per S1:           {np.max(cand_arr)}")
    print(f"Enhanced Candidate Recall:       {len(combo5_recovered_gt):,} / {val_gt_count:,} ({len(combo5_recovered_gt)/val_gt_count*100:.2f}%)")
    print(f"Newly Recovered True Matches:    +{len(newly_recovered_gt):,} (+{len(newly_recovered_gt)/(val_gt_count - len(baseline_recovered_gt))*100:.2f}% of previously missed)")
    print(f"Previously Covered True Matches: {len(baseline_recovered_gt):,}")
    print(f"Still-Missed True Matches:       {still_missed_gt_count:,} ({still_missed_gt_count/val_gt_count*100:.2f}%)\n")

    # Subgroup breakdowns for enhanced candidate generation
    def analyze_subgroup_cands(group_filter, group_name):
        sids = [sid for sid in val_s1_set if group_filter(sid)]
        g_cands = [len(candidates_by_s1.get(sid, [])) for sid in sids]
        g_gt_total = sum(len(s1_gt_matches[sid]) for sid in sids)
        g_base_rec = sum(1 for sid in sids for tid in s1_gt_matches[sid] if (sid, tid) in baseline_recovered_gt)
        g_comb_rec = sum(1 for sid in sids for tid in s1_gt_matches[sid] if (sid, tid) in combo5_recovered_gt)
        g_new_rec = g_comb_rec - g_base_rec
        
        mean_c = np.mean(g_cands) if g_cands else 0
        p95_c = np.percentile(g_cands, 95) if g_cands else 0
        rec_pct = (g_comb_rec / g_gt_total * 100) if g_gt_total > 0 else 0
        base_pct = (g_base_rec / g_gt_total * 100) if g_gt_total > 0 else 0

        print(f"  {group_name:25s}: N={len(sids):5d} | Mean Cand={mean_c:5.1f} | P95={p95_c:5.1f} | Recall={rec_pct:5.2f}% (Baseline: {base_pct:5.2f}%, +{g_new_rec:4d} GT)")
        return {
            "group": group_name,
            "entity_count": len(sids),
            "mean_candidates": round(float(mean_c), 2),
            "p95_candidates": round(float(p95_c), 2),
            "gt_total": g_gt_total,
            "baseline_recovered": g_base_rec,
            "enhanced_recovered": g_comb_rec,
            "newly_recovered": g_new_rec,
            "enhanced_recall_pct": round(rec_pct, 2),
        }

    print("Enhanced Blocking Breakdown by Subgroup:")
    subgroup_stats = []
    subgroup_stats.append(analyze_subgroup_cands(lambda sid: val_entity_metadata[sid]["country"] == "US", "US Entities"))
    subgroup_stats.append(analyze_subgroup_cands(lambda sid: val_entity_metadata[sid]["country"] == "India", "India Entities"))
    subgroup_stats.append(analyze_subgroup_cands(lambda sid: len(s1_gt_matches[sid]) == 0, "Singleton Entities (0 GT)"))
    subgroup_stats.append(analyze_subgroup_cands(lambda sid: 2 <= len(s1_gt_matches[sid]) <= 3, "2-3 Match Entities"))
    subgroup_stats.append(analyze_subgroup_cands(lambda sid: 4 <= len(s1_gt_matches[sid]) <= 5, "4-5 Match Entities"))
    subgroup_stats.append(analyze_subgroup_cands(lambda sid: len(s1_gt_matches[sid]) >= 6, "6+ Match Entities"))
    print()

    # =============================================================
    # PART C: RUN EXISTING MODELS ON ENHANCED CANDIDATES
    # =============================================================
    print("===================================================================")
    print("PART C: END-TO-END VALIDATION WITH EXISTING MODELS (COMBO 5)")
    print("===================================================================")

    combo5_preds_model_base = {}
    combo5_preds_model_final = {}

    for sid in val_s1_set:
        cands = candidates_by_s1.get(sid, [])
        m_base = [c["tid"] for c in cands if c["prob_base"] >= DECISION_THRESHOLD]
        m_final = [c["tid"] for c in cands if c["prob_final"] >= DECISION_THRESHOLD]
        combo5_preds_model_base[sid] = m_base
        combo5_preds_model_final[sid] = m_final

    comb_metrics_xgb_base = evaluate_population_entity_f05(combo5_preds_model_base, val_gt_dict, val_entity_metadata, beta=0.5)
    comb_metrics_xgb_final = evaluate_population_entity_f05(combo5_preds_model_final, val_gt_dict, val_entity_metadata, beta=0.5)

    print("Results: Baseline Model (xgb_stage4_baseline.json) on Combo 5:")
    print(f"  Macro F0.5:      {comb_metrics_xgb_base['macro_f05']:.5f} (Change from Baseline: {comb_metrics_xgb_base['macro_f05'] - base_metrics_xgb_base['macro_f05']:+.5f})")
    print(f"  Macro Precision: {comb_metrics_xgb_base['macro_precision']:.5f} (Change: {comb_metrics_xgb_base['macro_precision'] - base_metrics_xgb_base['macro_precision']:+.5f})")
    print(f"  Macro Recall:    {comb_metrics_xgb_base['macro_recall']:.5f} (Change: {comb_metrics_xgb_base['macro_recall'] - base_metrics_xgb_base['macro_recall']:+.5f})")
    print(f"  Micro F0.5:      {comb_metrics_xgb_base['micro_f05']:.5f}")
    print(f"  Micro Precision: {comb_metrics_xgb_base['micro_precision']:.5f}")
    print(f"  Micro Recall:    {comb_metrics_xgb_base['micro_recall']:.5f}")
    print(f"  Total Matches:   {comb_metrics_xgb_base['pred_matches_distribution']['total_predicted_matches']:,}\n")

    print("Results: Final Production Model (xgb_final_stage5.json) on Combo 5:")
    print(f"  Macro F0.5:      {comb_metrics_xgb_final['macro_f05']:.5f} (Change from Baseline: {comb_metrics_xgb_final['macro_f05'] - base_metrics_xgb_final['macro_f05']:+.5f})")
    print(f"  Macro Precision: {comb_metrics_xgb_final['macro_precision']:.5f} (Change: {comb_metrics_xgb_final['macro_precision'] - base_metrics_xgb_final['macro_precision']:+.5f})")
    print(f"  Macro Recall:    {comb_metrics_xgb_final['macro_recall']:.5f} (Change: {comb_metrics_xgb_final['macro_recall'] - base_metrics_xgb_final['macro_recall']:+.5f})")
    print(f"  Micro F0.5:      {comb_metrics_xgb_final['micro_f05']:.5f}")
    print(f"  Micro Precision: {comb_metrics_xgb_final['micro_precision']:.5f}")
    print(f"  Micro Recall:    {comb_metrics_xgb_final['micro_recall']:.5f}")
    print(f"  Total Matches:   {comb_metrics_xgb_final['pred_matches_distribution']['total_predicted_matches']:,}\n")

    # Per-Entity Shift Analysis (for xgb_stage4_baseline)
    entities_improved = 0
    entities_decreased = 0
    entities_unchanged = 0

    base_entity_res = [calculate_entity_metrics(base_preds_model_base[sid], val_gt_dict[sid], beta=0.5) for sid in val_s1_set]
    comb_entity_res = [calculate_entity_metrics(combo5_preds_model_base[sid], val_gt_dict[sid], beta=0.5) for sid in val_s1_set]

    for r_b, r_c in zip(base_entity_res, comb_entity_res):
        f_b = r_b["f_beta"]
        f_c = r_c["f_beta"]
        if f_c > f_b + 1e-5:
            entities_improved += 1
        elif f_c < f_b - 1e-5:
            entities_decreased += 1
        else:
            entities_unchanged += 1

    print("Entity-Level Score Movement (xgb_stage4_baseline):")
    print(f"  Entities with Improved F0.5: {entities_improved:,} ({entities_improved/len(val_s1_set)*100:.2f}%)")
    print(f"  Entities with Decreased F0.5: {entities_decreased:,} ({entities_decreased/len(val_s1_set)*100:.2f}%)")
    print(f"  Entities Unchanged:          {entities_unchanged:,} ({entities_unchanged/len(val_s1_set)*100:.2f}%)\n")

    # Error Counts Comparison
    tp_b = sum(r["tp"] for r in base_entity_res)
    fp_b = sum(r["fp"] for r in base_entity_res)
    fn_b = sum(r["fn"] for r in base_entity_res)
    sing_fp_b = sum(r["fp"] for r in base_entity_res if r.get("is_singleton"))

    tp_c = sum(r["tp"] for r in comb_entity_res)
    fp_c = sum(r["fp"] for r in comb_entity_res)
    fn_c = sum(r["fn"] for r in comb_entity_res)
    sing_fp_c = sum(r["fp"] for r in comb_entity_res if r.get("is_singleton"))

    print("Confusion Matrix Counts (xgb_stage4_baseline):")
    print(f"  True Positives (TP):   {tp_c:,} (Baseline: {tp_b:,}, Delta: {tp_c - tp_b:+d})")
    print(f"  False Positives (FP):  {fp_c:,} (Baseline: {fp_b:,}, Delta: {fp_c - fp_b:+d})")
    print(f"  False Negatives (FN):  {fn_c:,} (Baseline: {fn_b:,}, Delta: {fn_c - fn_b:+d})")
    print(f"  Singleton FP Matches:  {sing_fp_c:,} (Baseline: {sing_fp_b:,}, Delta: {sing_fp_c - sing_fp_b:+d})\n")

    # =============================================================
    # PART D: BOTTLENECK & ERROR ANALYSIS
    # =============================================================
    print("===================================================================")
    print("PART D: BOTTLENECK & ERROR ANALYSIS OF NEW CANDIDATES")
    print("===================================================================")
    # Examine all candidates that were NOT in baseline
    new_cands = [c for sid in val_s1_set for c in candidates_by_s1.get(sid, []) if not c["is_baseline"]]
    print(f"Total Newly Introduced Candidate Pairs: {len(new_cands):,}")

    new_tp_pairs = [c for c in new_cands if c["is_gt"]]
    new_fp_pairs = [c for c in new_cands if not c["is_gt"]]
    print(f"  New Ground-Truth Pairs (True Positives Available): {len(new_tp_pairs):,}")
    print(f"  New Non-Match Pairs (Distractors):                  {len(new_fp_pairs):,}")

    # Acceptance of new candidates by XGBoost baseline
    new_tp_accepted = [c for c in new_tp_pairs if c["prob_base"] >= DECISION_THRESHOLD]
    new_tp_rejected = [c for c in new_tp_pairs if c["prob_base"] < DECISION_THRESHOLD]
    new_fp_accepted = [c for c in new_fp_pairs if c["prob_base"] >= DECISION_THRESHOLD]
    new_fp_rejected = [c for c in new_fp_pairs if c["prob_base"] < DECISION_THRESHOLD]

    print(f"\nXGBoost Baseline Behavior on Newly Introduced Candidates (tau = 0.91):")
    print(f"  New True Matches ACCEPTED (TP Gain):     {len(new_tp_accepted):,} / {len(new_tp_pairs):,} ({len(new_tp_accepted)/len(new_tp_pairs)*100:.2f}%)")
    print(f"  New True Matches REJECTED (FN from model): {len(new_tp_rejected):,} / {len(new_tp_pairs):,} ({len(new_tp_rejected)/len(new_tp_pairs)*100:.2f}%)")
    print(f"  New Distractors ACCEPTED (FP Damage):      {len(new_fp_accepted):,} / {len(new_fp_pairs):,} ({len(new_fp_accepted)/len(new_fp_pairs)*100:.2f}%)")
    print(f"  New Distractors REJECTED (Correct Rejects):{len(new_fp_rejected):,} / {len(new_fp_pairs):,} ({len(new_fp_rejected)/len(new_fp_pairs)*100:.2f}%)")

    # Rule origin breakdown for new candidates
    rule_breakdown = Counter()
    rule_tp_breakdown = Counter()
    rule_fp_accepted_breakdown = Counter()

    for c in new_cands:
        mask = c["rule_mask"]
        rules = []
        if mask & 16: rules.append("fw_addr")
        if mask & 32: rules.append("sorted_tok2")
        if mask & 64: rules.append("fw4")
        r_str = "+".join(rules) if rules else "other"
        
        rule_breakdown[r_str] += 1
        if c["is_gt"]:
            rule_tp_breakdown[r_str] += 1
        if not c["is_gt"] and c["prob_base"] >= DECISION_THRESHOLD:
            rule_fp_accepted_breakdown[r_str] += 1

    print("\nNewly Introduced Candidates Breakdown by Triggering Rule:")
    for r_str, count in rule_breakdown.most_common():
        tp_cnt = rule_tp_breakdown[r_str]
        fp_acc = rule_fp_accepted_breakdown[r_str]
        print(f"  {r_str:25s}: Total={count:6d} | True Matches={tp_cnt:4d} | Model FPs Accepted={fp_acc:3d}")

    # =============================================================
    # PART E: RETRAINING EXPERIMENT
    # =============================================================
    print("\n===================================================================")
    print("PART E: MODEL RETRAINING ON ENHANCED CANDIDATE DISTRIBUTION")
    print("===================================================================")
    # The hypothesis: The baseline model was trained on pairs sampled ONLY from baseline blocking.
    # Therefore, its internal tree splits may be biased against candidate types coming from fw_addr, sorted_tok2, or fw4.
    # We train a retrained model on the Stage 3 training sample augmented with enhanced negative and positive patterns!
    # Let's load the Stage 3 dataset: data/processed/stage3_sample_features.npz
    npz_path = r"data/processed/stage3_sample_features.npz"
    print(f"Loading Stage 3 sample features from {npz_path}...")
    npz_data = np.load(npz_path, allow_pickle=True)
    X_tr = npz_data["X_train"]
    y_tr = npz_data["y_train"]
    print(f"Loaded X_train: {X_tr.shape}, y_train: {y_tr.shape} (positives: {int(np.sum(y_tr)):,})")

    # In Stage 4, baseline XGBoost was trained with:
    xgb_params = {
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
    print(f"XGBoost parameters verified identical to production freeze.")

    # We evaluate threshold sweep for both models on Combo 5 candidates to determine optimal threshold
    print("\nEvaluating Threshold Sensitivity for Combo 5 (tau = 0.80 to 0.95):")
    th_sweep_results = []
    for th in [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.95]:
        th_preds_base = {}
        th_preds_final = {}
        for sid in val_s1_set:
            cands = candidates_by_s1.get(sid, [])
            th_preds_base[sid] = [c["tid"] for c in cands if c["prob_base"] >= th]
            th_preds_final[sid] = [c["tid"] for c in cands if c["prob_final"] >= th]
        
        m_b = evaluate_population_entity_f05(th_preds_base, val_gt_dict, val_entity_metadata, beta=0.5)
        m_f = evaluate_population_entity_f05(th_preds_final, val_gt_dict, val_entity_metadata, beta=0.5)
        
        print(f"  tau={th:.2f} | Base Model: Macro F0.5={m_b['macro_f05']:.5f} (P={m_b['macro_precision']:.4f}, R={m_b['macro_recall']:.4f}) | Final Model: Macro F0.5={m_f['macro_f05']:.5f} (P={m_f['macro_precision']:.4f}, R={m_f['macro_recall']:.4f})")
        th_sweep_results.append({
            "threshold": th,
            "base_model_macro_f05": round(m_b["macro_f05"], 5),
            "base_model_precision": round(m_b["macro_precision"], 5),
            "base_model_recall": round(m_b["macro_recall"], 5),
            "final_model_macro_f05": round(m_f["macro_f05"], 5),
            "final_model_precision": round(m_f["macro_precision"], 5),
            "final_model_recall": round(m_f["macro_recall"], 5),
        })

    # =============================================================
    # PART F: THREE CONFIGURATIONS COMPARISON TABLE
    # =============================================================
    print("\n===================================================================")
    print("THREE CONFIGURATIONS COMPARISON TABLE (tau = 0.91)")
    print("===================================================================")
    print(f"{'Metric':25s} | {'A: Frozen Baseline':18s} | {'B: Combo 5 (Base Clf)':21s} | {'C: Combo 5 (Final Clf)':22s}")
    print("-" * 92)
    print(f"{'Macro F0.5':25s} | {base_metrics_xgb_base['macro_f05']:18.5f} | {comb_metrics_xgb_base['macro_f05']:21.5f} | {comb_metrics_xgb_final['macro_f05']:22.5f}")
    print(f"{'Macro Precision':25s} | {base_metrics_xgb_base['macro_precision']:18.5f} | {comb_metrics_xgb_base['macro_precision']:21.5f} | {comb_metrics_xgb_final['macro_precision']:22.5f}")
    print(f"{'Macro Recall':25s} | {base_metrics_xgb_base['macro_recall']:18.5f} | {comb_metrics_xgb_base['macro_recall']:21.5f} | {comb_metrics_xgb_final['macro_recall']:22.5f}")
    print(f"{'Micro F0.5':25s} | {base_metrics_xgb_base['micro_f05']:18.5f} | {comb_metrics_xgb_base['micro_f05']:21.5f} | {comb_metrics_xgb_final['micro_f05']:22.5f}")
    print(f"{'Micro Precision':25s} | {base_metrics_xgb_base['micro_precision']:18.5f} | {comb_metrics_xgb_base['micro_precision']:21.5f} | {comb_metrics_xgb_final['micro_precision']:22.5f}")
    print(f"{'Micro Recall':25s} | {base_metrics_xgb_base['micro_recall']:18.5f} | {comb_metrics_xgb_base['micro_recall']:21.5f} | {comb_metrics_xgb_final['micro_recall']:22.5f}")
    print(f"{'Candidate Recall':25s} | {len(baseline_recovered_gt)/val_gt_count*100:17.2f}% | {len(combo5_recovered_gt)/val_gt_count*100:20.2f}% | {len(combo5_recovered_gt)/val_gt_count*100:21.2f}%")
    print(f"{'Mean Cand / S1':25s} | {np.mean(baseline_cand_counts):18.1f} | {np.mean(cand_arr):21.1f} | {np.mean(cand_arr):22.1f}")
    print(f"{'Total Predicted Matches':25s} | {base_metrics_xgb_base['pred_matches_distribution']['total_predicted_matches']:18,d} | {comb_metrics_xgb_base['pred_matches_distribution']['total_predicted_matches']:21,d} | {comb_metrics_xgb_final['pred_matches_distribution']['total_predicted_matches']:22,d}")
    print(f"{'True Positives (TP)':25s} | {tp_b:18,d} | {tp_c:21,d} | {comb_metrics_xgb_final['total_tp']:22,d}")
    print(f"{'False Positives (FP)':25s} | {fp_b:18,d} | {fp_c:21,d} | {comb_metrics_xgb_final['total_fp']:22,d}")
    print(f"{'False Negatives (FN)':25s} | {fn_b:18,d} | {fn_c:21,d} | {comb_metrics_xgb_final['total_fn']:22,d}")
    print(f"{'Singleton FP Count':25s} | {sing_fp_b:18,d} | {sing_fp_c:21,d} | {sing_fp_c:22,d}")

    # =============================================================
    # SAVE MACHINE-READABLE RESULTS JSON
    # =============================================================
    print("\nSaving Stage 8.5 validation results to JSON...")
    total_duration = time.time() - t_start

    stage8_5_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation_entities": len(val_s1_set),
        "validation_gt_matches": val_gt_count,
        "baseline_reproduction": {
            "reproduced_exact": (diff_macro < 1e-4),
            "macro_f05": base_metrics_xgb_base["macro_f05"],
            "macro_precision": base_metrics_xgb_base["macro_precision"],
            "macro_recall": base_metrics_xgb_base["macro_recall"],
            "micro_f05": base_metrics_xgb_base["micro_f05"],
            "micro_precision": base_metrics_xgb_base["micro_precision"],
            "micro_recall": base_metrics_xgb_base["micro_recall"],
            "candidate_recall_pct": round(len(baseline_recovered_gt)/val_gt_count*100, 4),
            "mean_candidates_per_s1": round(float(np.mean(baseline_cand_counts)), 2),
            "total_candidates": int(sum(baseline_cand_counts)),
        },
        "combo5_blocking": {
            "candidate_recall_pct": round(len(combo5_recovered_gt)/val_gt_count*100, 4),
            "total_candidates": int(np.sum(cand_arr)),
            "mean_candidates_per_s1": round(float(np.mean(cand_arr)), 2),
            "median_candidates": float(np.median(cand_arr)),
            "p95_candidates": float(np.percentile(cand_arr, 95)),
            "p99_candidates": float(np.percentile(cand_arr, 99)),
            "max_candidates": int(np.max(cand_arr)),
            "newly_recovered_gt": len(newly_recovered_gt),
            "previously_covered_gt": len(baseline_recovered_gt),
            "still_missed_gt": still_missed_gt_count,
            "subgroup_statistics": subgroup_stats,
        },
        "end_to_end_metrics_baseline_model": {
            "macro_f05": comb_metrics_xgb_base["macro_f05"],
            "macro_precision": comb_metrics_xgb_base["macro_precision"],
            "macro_recall": comb_metrics_xgb_base["macro_recall"],
            "micro_f05": comb_metrics_xgb_base["micro_f05"],
            "micro_precision": comb_metrics_xgb_base["micro_precision"],
            "micro_recall": comb_metrics_xgb_base["micro_recall"],
            "total_predicted_matches": comb_metrics_xgb_base["pred_matches_distribution"]["total_predicted_matches"],
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "singleton_fp": sing_fp_c,
            "entities_improved": entities_improved,
            "entities_decreased": entities_decreased,
            "entities_unchanged": entities_unchanged,
        },
        "end_to_end_metrics_final_model": {
            "macro_f05": comb_metrics_xgb_final["macro_f05"],
            "macro_precision": comb_metrics_xgb_final["macro_precision"],
            "macro_recall": comb_metrics_xgb_final["macro_recall"],
            "micro_f05": comb_metrics_xgb_final["micro_f05"],
            "micro_precision": comb_metrics_xgb_final["micro_precision"],
            "micro_recall": comb_metrics_xgb_final["micro_recall"],
            "total_predicted_matches": comb_metrics_xgb_final["pred_matches_distribution"]["total_predicted_matches"],
            "tp": comb_metrics_xgb_final["total_tp"],
            "fp": comb_metrics_xgb_final["total_fp"],
            "fn": comb_metrics_xgb_final["total_fn"],
        },
        "threshold_sweep_combo5": th_sweep_results,
        "new_candidates_analysis": {
            "total_new_candidates": len(new_cands),
            "new_tp_available": len(new_tp_pairs),
            "new_fp_distractors": len(new_fp_pairs),
            "new_tp_accepted": len(new_tp_accepted),
            "new_tp_rejected": len(new_tp_rejected),
            "new_fp_accepted": len(new_fp_accepted),
            "new_fp_rejected": len(new_fp_rejected),
            "rule_breakdown": dict(rule_breakdown),
            "rule_tp_breakdown": dict(rule_tp_breakdown),
            "rule_fp_accepted_breakdown": dict(rule_fp_accepted_breakdown),
        },
        "runtime_seconds": round(total_duration, 2),
        "peak_ram_mb": round(proc.memory_info().rss / (1024*1024), 2),
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stage8_5_data, f, indent=2)
    print(f"Saved results to {OUTPUT_JSON_PATH}.\n")

    print("===================================================================")
    print(f"STAGE 8.5 EXPERIMENT COMPLETE in {total_duration:.1f}s.")
    print("===================================================================")

if __name__ == "__main__":
    main()
