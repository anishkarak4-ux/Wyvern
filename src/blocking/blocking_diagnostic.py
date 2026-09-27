"""
blocking_diagnostic.py - Stage 8: Candidate-Generation / Blocking Recall Diagnostic
Amazon ML Challenge 2026

Investigates why 28.32% (9,819) of true matches are missed by current baseline blocking
on the 10,002 held-out validation Source 1 entities.
Evaluates candidate blocking keys, bucket caps, and failure modes without modifying production.
"""

import os
import sys
import csv
import json
import time
import math
import unicodedata
import psutil
import random
from collections import defaultdict, Counter
import numpy as np
import rapidfuzz.fuzz as rf_fuzz

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
from src.features.pair_features import extract_postal_code

RAW_DIR = r"C:\Users\KIIT\Desktop\Wyvern\data\raw\train"
RAW_TEST_DIR = r"C:\Users\KIIT\Desktop\Wyvern\data\raw\test"
OUTPUT_JSON = r"C:\Users\KIIT\Desktop\Wyvern\data\models\stage8_blocking_diagnostic.json"
REPORT_MD = r"C:\Users\KIIT\Desktop\Wyvern\docs\stage8_blocking_diagnostic_report.md"

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

def verify_raw_datasets():
    print("=== VERIFYING RAW DATASETS UNTOUCHED ===")
    for path, expected_sz in RAW_BYTE_SIZES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raw file: {path}")
        actual_sz = os.path.getsize(path)
        if actual_sz != expected_sz:
            raise ValueError(f"CORRUPTION in {path}: expected {expected_sz} bytes, found {actual_sz} bytes!")
    print("All raw datasets verified 100% untouched.\n")

def ascii_transliterate(text):
    """Native ASCII transliteration using NFKD normalization (no external deps)."""
    if not text:
        return ""
    norm = unicodedata.normalize('NFKD', text)
    ascii_bytes = norm.encode('ascii', 'ignore')
    return ascii_bytes.decode('ascii')

def get_sorted_tokens(cleaned_name, max_tokens=2):
    """Return sorted first N tokens (order-invariant key)."""
    stripped = strip_legal_suffixes(cleaned_name)
    tokens = stripped.split() if stripped else cleaned_name.split()
    if not tokens:
        return ""
    selected = sorted(tokens[:max_tokens])
    return "_".join(selected)

def get_longest_word(cleaned_name, min_len=5):
    """Return the longest distinctive token >= min_len."""
    stripped = strip_legal_suffixes(cleaned_name)
    tokens = stripped.split() if stripped else cleaned_name.split()
    cand = [t for t in tokens if len(t) >= min_len]
    if not cand:
        return ""
    return max(cand, key=len)

def get_second_word(cleaned_name, min_len=3):
    """Return 2nd significant word if available."""
    stripped = strip_legal_suffixes(cleaned_name)
    tokens = stripped.split() if stripped else cleaned_name.split()
    if len(tokens) >= 2 and len(tokens[1]) >= min_len:
        return tokens[1]
    return ""

def make_entity_rep(eid, name, addr, country):
    """Compute all baseline and candidate representations for an entity."""
    norm_name = clean_text(name)
    stripped = strip_legal_suffixes(norm_name)
    tok2 = get_first_two_tokens(norm_name)
    pfx8 = get_name_prefix(norm_name, length=8)
    pfx6 = get_name_prefix(norm_name, length=6)
    pfx10 = get_name_prefix(norm_name, length=10)
    fw = get_first_significant_word(norm_name, min_len=3)
    fw4 = get_first_significant_word(norm_name, min_len=4)
    second_w = get_second_word(norm_name, min_len=3)
    longest_w = get_longest_word(norm_name, min_len=5)
    sorted_tok2 = get_sorted_tokens(norm_name, max_tokens=2)
    
    # ASCII transliteration
    trans_name = ascii_transliterate(norm_name)
    trans_stripped = strip_legal_suffixes(trans_name)
    trans_tok2 = get_first_two_tokens(trans_name)

    norm_addr = normalize_address(addr) or ""
    nums_all = extract_address_numeric_tokens(norm_addr)
    nums_sel = [n for n in nums_all if len(n) >= 3]
    nums_ge2 = [n for n in nums_all if len(n) >= 2]
    fc = norm_name[0] if norm_name else ""
    fc2 = norm_name[:2] if len(norm_name) >= 2 else norm_name
    c = country.strip() if country else ""
    postal = extract_postal_code(norm_addr, c)
    
    tokens_set = set(norm_name.split())
    stripped_tokens_set = set(stripped.split()) if stripped else tokens_set

    return {
        "entity_id": eid,
        "country": c,
        "raw_name": name,
        "raw_addr": addr,
        "norm_name": norm_name,
        "stripped_name": stripped,
        "first_two_tokens": tok2,
        "name_pfx8": pfx8,
        "name_pfx6": pfx6,
        "name_pfx10": pfx10,
        "first_word": fw,
        "first_word4": fw4,
        "second_word": second_w,
        "longest_word": longest_w,
        "sorted_tok2": sorted_tok2,
        "trans_stripped": trans_stripped,
        "trans_tok2": trans_tok2,
        "norm_addr": norm_addr,
        "addr_numeric": nums_all,
        "addr_nums_sel": nums_sel,
        "addr_nums_ge2": nums_ge2,
        "first_char": fc,
        "first_two_chars": fc2,
        "postal_code": postal,
        "tokens_set": tokens_set,
        "stripped_tokens_set": stripped_tokens_set,
        "has_addr": bool(norm_addr),
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
    print("STAGE 8: CANDIDATE-GENERATION / BLOCKING RECALL DIAGNOSTIC")
    print("===================================================================\n")

    verify_raw_datasets()

    # 1. Reconstruct Validation Split
    print("=== STEP 1: Reconstructing Stage 5 Validation Set ===")
    t0 = time.time()
    val_s1_set, s1_gt_matches, s1_metadata_all = reconstruct_validation_split()
    print(f"Validation S1 entities: {len(val_s1_set):,}")
    val_gt_count = sum(len(s1_gt_matches[sid]) for sid in val_s1_set)
    val_sing_count = sum(1 for sid in val_s1_set if len(s1_gt_matches[sid]) == 0)
    print(f"Validation Ground-Truth Matches: {val_gt_count:,} across {len(val_s1_set) - val_sing_count:,} non-singleton entities.")
    print(f"Validation Singletons: {val_sing_count:,} ({val_sing_count/len(val_s1_set)*100:.2f}%)\n")

    assert len(val_s1_set) == 10002, f"Expected 10,002 validation entities, got {len(val_s1_set)}"
    assert val_gt_count == 34677, f"Expected 34,677 GT matches, got {val_gt_count}"
    assert val_sing_count == 559, f"Expected 559 singletons, got {val_sing_count}"

    # Collect target IDs in validation ground truth
    val_target_to_s1s = defaultdict(set)
    val_gt_target_ids = set()
    for sid in val_s1_set:
        for tid in s1_gt_matches[sid]:
            val_target_to_s1s[tid].add(sid)
            val_gt_target_ids.add(tid)
    print(f"Unique target entities in validation GT: {len(val_gt_target_ids):,}")

    # Load validation S1 entity representations
    print("Loading Validation S1 representations...")
    s1_path = os.path.join(RAW_DIR, "train_source1.tsv")
    val_s1_reps = {}
    
    # Inverted indexes for baseline blocking
    val_idx_stripped = defaultdict(list)
    val_idx_tok2 = defaultdict(list)
    val_idx_pfx8 = defaultdict(list)
    val_idx_addr_num = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            sid = r[0]
            if sid in val_s1_set:
                rep = make_entity_rep(sid, r[1], r[2], r[3])
                val_s1_reps[sid] = rep
                c = rep["country"]
                if rep["stripped_name"]: val_idx_stripped[(c, rep["stripped_name"])].append(sid)
                if rep["first_two_tokens"]: val_idx_tok2[(c, rep["first_two_tokens"])].append(sid)
                if rep["name_pfx8"]: val_idx_pfx8[(c, rep["name_pfx8"])].append(sid)
                if rep["first_char"]:
                    for num in rep["addr_nums_sel"]:
                        val_idx_addr_num[(c, num, rep["first_char"])].append(sid)

    print(f"Validation S1 representations indexed in {time.time() - t0:.1f}s. RAM: {proc.memory_info().rss / (1024*1024):.1f} MB\n")

    # 2. Stream Target Files (S2 & S3)
    # Collect:
    # a) Target entity representations for all 34,677 GT matches
    # b) Baseline candidate recovery (verifying exact 24,858 vs 9,819 split)
    # c) Target bucket frequency counters across full S2 and S3 for all candidate keys
    print("=== STEP 2: Streaming Target Datasets (S2 & S3) ===")
    t_stream = time.time()
    
    target_reps = {} # tid -> target entity representation
    baseline_recovered_gt = set() # (sid, tid) recovered in baseline
    baseline_uncapped_gt = set()  # (sid, tid) that would be recovered if caps were infinite

    # Baseline frequency counters
    bucket_counts_tok2 = Counter()
    bucket_counts_pfx8 = Counter()
    bucket_counts_addr_num = Counter()

    # Candidate keys frequency counters across all target records
    freq_stripped = Counter()
    freq_tok2 = Counter()
    freq_pfx8 = Counter()
    freq_pfx6 = Counter()
    freq_pfx10 = Counter()
    freq_fw3 = Counter()
    freq_fw4 = Counter()
    freq_sorted_tok2 = Counter()
    freq_fw3_addr_num = Counter()
    freq_fw3_postal = Counter()
    freq_postal_fc2 = Counter()
    freq_addr_num_fc2 = Counter()
    freq_trans_stripped = Counter()
    freq_trans_tok2 = Counter()
    freq_second_w = Counter()
    freq_longest_w = Counter()

    def stream_target_file(filepath, src_label):
        t_f = time.time()
        print(f"Streaming {src_label} ({os.path.basename(filepath)})...")
        rows = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                rows += 1
                tid = r[0]
                t_name = r[1]
                t_addr = r[2]
                t_c = r[3].strip() if len(r) > 3 else ""

                # If this target is in validation ground truth, save its full representation
                if tid in val_gt_target_ids:
                    t_rep = make_entity_rep(tid, t_name, t_addr, t_c)
                    target_reps[tid] = t_rep
                else:
                    # Lightweight representation for bucket counting
                    t_rep = None

                norm_name = clean_text(t_name)
                sn = strip_legal_suffixes(norm_name)
                tokens = sn.split() if sn else norm_name.split()
                num_tok = len(tokens)
                tok2 = (tokens[0] + "_" + tokens[1]) if num_tok >= 2 else (tokens[0] if num_tok == 1 else "")
                pfx8 = sn[:8] if len(sn) >= 3 else ""
                pfx6 = sn[:6] if len(sn) >= 3 else ""
                pfx10 = sn[:10] if len(sn) >= 3 else ""
                fw3 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 3) else ""
                fw4 = tokens[0] if (num_tok >= 1 and len(tokens[0]) >= 4) else ""
                sorted_tok2 = "_".join(sorted(tokens[:2])) if num_tok >= 1 else ""
                sec_w = tokens[1] if (num_tok >= 2 and len(tokens[1]) >= 3) else ""
                long_w = max((t for t in tokens if len(t) >= 5), key=len, default="") if num_tok >= 1 else ""

                if not norm_name.isascii():
                    trans_sn = strip_legal_suffixes(ascii_transliterate(norm_name))
                    trans_toks = trans_sn.split() if trans_sn else []
                    trans_t2 = (trans_toks[0] + "_" + trans_toks[1]) if len(trans_toks) >= 2 else (trans_toks[0] if trans_toks else "")
                else:
                    trans_sn = sn
                    trans_t2 = tok2

                fc = norm_name[0] if norm_name else ""

                # Target bucket counters (streaming order for baseline simulation)
                if tok2: bucket_counts_tok2[(t_c, tok2)] += 1
                if pfx8: bucket_counts_pfx8[(t_c, pfx8)] += 1

                # Track baseline rule matches
                matched_s1s = {}
                uncapped_s1s = set()

                if sn and (t_c, sn) in val_idx_stripped:
                    for sid in val_idx_stripped[(t_c, sn)]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 1
                        uncapped_s1s.add(sid)

                if tok2 and (t_c, tok2) in val_idx_tok2:
                    for sid in val_idx_tok2[(t_c, tok2)]:
                        uncapped_s1s.add(sid)
                        if bucket_counts_tok2[(t_c, tok2)] <= 200:
                            matched_s1s[sid] = matched_s1s.get(sid, 0) | 2

                if pfx8 and (t_c, pfx8) in val_idx_pfx8:
                    for sid in val_idx_pfx8[(t_c, pfx8)]:
                        uncapped_s1s.add(sid)
                        if bucket_counts_pfx8[(t_c, pfx8)] <= 200:
                            matched_s1s[sid] = matched_s1s.get(sid, 0) | 4

                t_norm_addr = normalize_address(t_addr) or ""
                t_nums_all = extract_address_numeric_tokens(t_norm_addr)
                t_nums_sel = [n for n in t_nums_all if len(n) >= 3]

                if fc:
                    for num in t_nums_sel:
                        bucket_counts_addr_num[(t_c, num, fc)] += 1
                        if (t_c, num, fc) in val_idx_addr_num:
                            for sid in val_idx_addr_num[(t_c, num, fc)]:
                                uncapped_s1s.add(sid)
                                if bucket_counts_addr_num[(t_c, num, fc)] <= 100:
                                    matched_s1s[sid] = matched_s1s.get(sid, 0) | 8

                # If this target is in GT, check if matched
                if tid in val_gt_target_ids:
                    for sid in val_target_to_s1s[tid]:
                        if sid in matched_s1s:
                            baseline_recovered_gt.add((sid, tid))
                        if sid in uncapped_s1s:
                            baseline_uncapped_gt.add((sid, tid))

                # Global frequency accumulation for proposed candidate keys
                if sn: freq_stripped[(t_c, sn)] += 1
                if tok2: freq_tok2[(t_c, tok2)] += 1
                if pfx8: freq_pfx8[(t_c, pfx8)] += 1
                if pfx6: freq_pfx6[(t_c, pfx6)] += 1
                if pfx10: freq_pfx10[(t_c, pfx10)] += 1
                if fw3: freq_fw3[(t_c, fw3)] += 1
                if fw4: freq_fw4[(t_c, fw4)] += 1
                if sorted_tok2: freq_sorted_tok2[(t_c, sorted_tok2)] += 1
                if sec_w: freq_second_w[(t_c, sec_w)] += 1
                if long_w: freq_longest_w[(t_c, long_w)] += 1
                if trans_sn: freq_trans_stripped[(t_c, trans_sn)] += 1
                if trans_t2: freq_trans_tok2[(t_c, trans_t2)] += 1

                postal = extract_postal_code(t_norm_addr, t_c)
                fc2 = norm_name[:2] if len(norm_name) >= 2 else norm_name

                if fw3:
                    for num in t_nums_sel:
                        freq_fw3_addr_num[(t_c, fw3, num)] += 1
                    if postal:
                        freq_fw3_postal[(t_c, fw3, postal)] += 1

                if postal and fc2:
                    freq_postal_fc2[(t_c, postal, fc2)] += 1

                if fc2:
                    for num in t_nums_sel:
                        freq_addr_num_fc2[(t_c, num, fc2)] += 1

                if rows % 2000000 == 0:
                    print(f"  {src_label}: {rows:,} rows processed ({time.time() - t_f:.1f}s, RAM: {proc.memory_info().rss / (1024*1024):.1f} MB)")

        print(f"Completed {src_label}: {rows:,} rows in {time.time() - t_f:.1f}s.\n")

    s2_path = os.path.join(RAW_DIR, "train_source2.tsv")
    stream_target_file(s2_path, "Source 2")

    s3_path = os.path.join(RAW_DIR, "train_source3.tsv")
    stream_target_file(s3_path, "Source 3")

    print(f"Target streaming complete in {time.time() - t_stream:.1f}s.")
    print(f"Baseline Recovered GT Matches: {len(baseline_recovered_gt):,} / {val_gt_count:,} ({len(baseline_recovered_gt)/val_gt_count*100:.2f}%)")
    baseline_missed_gt = set()
    for sid in val_s1_set:
        for tid in s1_gt_matches[sid]:
            if (sid, tid) not in baseline_recovered_gt:
                baseline_missed_gt.add((sid, tid))
    print(f"Baseline Missed GT Matches:    {len(baseline_missed_gt):,} / {val_gt_count:,} ({len(baseline_missed_gt)/val_gt_count*100:.2f}%)\n")

    # Verify baseline match counts
    assert len(baseline_recovered_gt) == 24858, f"Expected 24,858 recovered, got {len(baseline_recovered_gt)}"
    assert len(baseline_missed_gt) == 9819, f"Expected 9,819 missed, got {len(baseline_missed_gt)}"
    print("Baseline recovery confirmed 100% identical to Stage 5 baseline!\n")

    # How many missed were due to bucket cap clipping?
    clipped_by_caps = baseline_uncapped_gt - baseline_recovered_gt
    pure_representational_miss = baseline_missed_gt - baseline_uncapped_gt
    print(f"Breakdown of 9,819 Baseline Misses:")
    print(f"  - Missed strictly due to BUCKET CAPS (key matched, but bucket exceeded cap): {len(clipped_by_caps):,} ({len(clipped_by_caps)/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  - Missed due to REPRESENTATIONAL MISMATCH (no baseline key matched):         {len(pure_representational_miss):,} ({len(pure_representational_miss)/len(baseline_missed_gt)*100:.2f}%)\n")

    # 3. Deep Analysis and Failure Classification of the 9,819 Missed True Matches
    print("=== STEP 3: Deep Feature & Failure Classification of Missed Matches ===")
    t_diag = time.time()

    # Classification counters
    missed_by_country = Counter()
    missed_by_src = Counter()
    missed_by_gt_bucket = Counter()
    missed_addr_status = Counter()

    # Key matching checks for missed pairs
    name_exact_norm = 0
    name_exact_stripped = 0
    name_tok2_match = 0
    name_pfx8_match = 0
    name_pfx6_match = 0
    name_pfx10_match = 0
    name_fw3_match = 0
    name_fw4_match = 0
    name_sorted_tok2_match = 0
    name_second_w_match = 0
    name_longest_w_match = 0
    name_fc_match = 0
    name_fc2_match = 0
    name_trans_sn_match = 0
    name_trans_tok2_match = 0

    # Address relationships
    addr_exact_norm = 0
    addr_shared_num_ge3 = 0
    addr_shared_num_ge2 = 0
    addr_shared_postal = 0
    addr_num_ge3_and_fc = 0
    addr_num_ge3_and_fc2 = 0
    addr_num_ge3_and_fw3 = 0
    postal_and_fc2 = 0
    postal_and_fw3 = 0

    # Semantic / token overlap
    jaccard_ge_50 = 0
    jaccard_ge_75 = 0
    token_subset_containment = 0
    rapidfuzz_ge_70 = 0
    rapidfuzz_ge_85 = 0

    # Specific error patterns
    error_patterns = Counter()

    for sid, tid in baseline_missed_gt:
        s1r = val_s1_reps[sid]
        tr = target_reps[tid]

        c = s1r["country"]
        is_s3 = tid.startswith("S3-")
        src = "S3" if is_s3 else "S2"
        gt_b = s1_metadata_all[sid]["match_bucket"]

        missed_by_country[c] += 1
        missed_by_src[src] += 1
        missed_by_gt_bucket[gt_b] += 1

        # Address status
        if not s1r["has_addr"] and not tr["has_addr"]:
            addr_stat = "both_missing"
        elif not s1r["has_addr"]:
            addr_stat = "s1_missing"
        elif not tr["has_addr"]:
            addr_stat = "target_missing"
        else:
            addr_stat = "both_present"
        missed_addr_status[addr_stat] += 1

        # Name checks
        sn_m = (s1r["stripped_name"] == tr["stripped_name"]) and bool(s1r["stripped_name"])
        t2_m = (s1r["first_two_tokens"] == tr["first_two_tokens"]) and bool(s1r["first_two_tokens"])
        p8_m = (s1r["name_pfx8"] == tr["name_pfx8"]) and bool(s1r["name_pfx8"])
        p6_m = (s1r["name_pfx6"] == tr["name_pfx6"]) and bool(s1r["name_pfx6"])
        p10_m = (s1r["name_pfx10"] == tr["name_pfx10"]) and bool(s1r["name_pfx10"])
        fw3_m = (s1r["first_word"] == tr["first_word"]) and bool(s1r["first_word"])
        fw4_m = (s1r["first_word4"] == tr["first_word4"]) and bool(s1r["first_word4"])
        st2_m = (s1r["sorted_tok2"] == tr["sorted_tok2"]) and bool(s1r["sorted_tok2"])
        sec_m = (s1r["second_word"] == tr["second_word"]) and bool(s1r["second_word"])
        lw_m = (s1r["longest_word"] == tr["longest_word"]) and bool(s1r["longest_word"])
        fc_m = (s1r["first_char"] == tr["first_char"]) and bool(s1r["first_char"])
        fc2_m = (s1r["first_two_chars"] == tr["first_two_chars"]) and bool(s1r["first_two_chars"])
        tsn_m = (s1r["trans_stripped"] == tr["trans_stripped"]) and bool(s1r["trans_stripped"])
        tt2_m = (s1r["trans_tok2"] == tr["trans_tok2"]) and bool(s1r["trans_tok2"])

        if s1r["norm_name"] == tr["norm_name"] and s1r["norm_name"]: name_exact_norm += 1
        if sn_m: name_exact_stripped += 1
        if t2_m: name_tok2_match += 1
        if p8_m: name_pfx8_match += 1
        if p6_m: name_pfx6_match += 1
        if p10_m: name_pfx10_match += 1
        if fw3_m: name_fw3_match += 1
        if fw4_m: name_fw4_match += 1
        if st2_m: name_sorted_tok2_match += 1
        if sec_m: name_second_w_match += 1
        if lw_m: name_longest_w_match += 1
        if fc_m: name_fc_match += 1
        if fc2_m: name_fc2_match += 1
        if tsn_m: name_trans_sn_match += 1
        if tt2_m: name_trans_tok2_match += 1

        # Address checks
        shared_nums_ge3 = set(s1r["addr_nums_sel"]) & set(tr["addr_nums_sel"])
        shared_nums_ge2 = set(s1r["addr_nums_ge2"]) & set(tr["addr_nums_ge2"])
        shared_post = (s1r["postal_code"] == tr["postal_code"]) and bool(s1r["postal_code"])

        if s1r["norm_addr"] and (s1r["norm_addr"] == tr["norm_addr"]): addr_exact_norm += 1
        if shared_nums_ge3: addr_shared_num_ge3 += 1
        if shared_nums_ge2: addr_shared_num_ge2 += 1
        if shared_post: addr_shared_postal += 1
        if shared_nums_ge3 and fc_m: addr_num_ge3_and_fc += 1
        if shared_nums_ge3 and fc2_m: addr_num_ge3_and_fc2 += 1
        if shared_nums_ge3 and fw3_m: addr_num_ge3_and_fw3 += 1
        if shared_post and fc2_m: postal_and_fc2 += 1
        if shared_post and fw3_m: postal_and_fw3 += 1

        # Token set & fuzzy analysis
        t_s1 = s1r["stripped_tokens_set"]
        t_tr = tr["stripped_tokens_set"]
        intersect = len(t_s1 & t_tr)
        union = len(t_s1 | t_tr)
        jacc = intersect / union if union > 0 else 0.0

        if jacc >= 0.50: jaccard_ge_50 += 1
        if jacc >= 0.75: jaccard_ge_75 += 1
        if (t_s1.issubset(t_tr) or t_tr.issubset(t_s1)) and intersect >= 1:
            token_subset_containment += 1

        rf_ratio = rf_fuzz.ratio(s1r["stripped_name"], tr["stripped_name"])
        if rf_ratio >= 70: rapidfuzz_ge_70 += 1
        if rf_ratio >= 85: rapidfuzz_ge_85 += 1

        # Root cause classification
        if (sid, tid) in clipped_by_caps:
            error_patterns["bucket_cap_clipping"] += 1
        elif st2_m and not t2_m:
            error_patterns["word_order_reversal"] += 1
        elif token_subset_containment and not sn_m:
            error_patterns["name_token_subset_expansion"] += 1
        elif tsn_m and not sn_m:
            error_patterns["transliteration_or_accent_difference"] += 1
        elif fw3_m and not t2_m and not p8_m:
            error_patterns["first_word_match_only"] += 1
        elif sec_m and not fw3_m:
            error_patterns["second_word_match_only"] += 1
        elif shared_nums_ge3 and not fc_m:
            error_patterns["shared_address_num_different_initial"] += 1
        elif shared_post and not fw3_m:
            error_patterns["shared_postal_different_name"] += 1
        elif addr_stat in ("s1_missing", "target_missing", "both_missing") and rf_ratio < 70:
            error_patterns["missing_addr_and_dissimilar_name"] += 1
        elif rf_ratio >= 70:
            error_patterns["fuzzy_name_variation_no_exact_prefix"] += 1
        else:
            error_patterns["severe_name_and_addr_divergence"] += 1

    print(f"Classification completed in {time.time() - t_diag:.1f}s.")
    print("\n--- MISSED MATCHES BY COUNTRY ---")
    for k, v in missed_by_country.most_common():
        print(f"  {k:10s}: {v:,} ({v/len(baseline_missed_gt)*100:.2f}%)")

    print("\n--- MISSED MATCHES BY TARGET SOURCE ---")
    for k, v in missed_by_src.most_common():
        print(f"  {k:10s}: {v:,} ({v/len(baseline_missed_gt)*100:.2f}%)")

    print("\n--- MISSED MATCHES BY ADDRESS STATUS ---")
    for k, v in missed_addr_status.most_common():
        print(f"  {k:20s}: {v:,} ({v/len(baseline_missed_gt)*100:.2f}%)")

    print("\n--- MISSED MATCHES BY GROUND TRUTH BUCKET ---")
    for k, v in missed_by_gt_bucket.most_common():
        print(f"  {k:10s}: {v:,} ({v/len(baseline_missed_gt)*100:.2f}%)")

    print("\n--- ROOT CAUSE CLASSIFICATION OF 9,819 MISSED MATCHES ---")
    for k, v in error_patterns.most_common():
        print(f"  {k:45s}: {v:,} ({v/len(baseline_missed_gt)*100:.2f}%)")

    print("\n--- POTENTIAL KEY ALIGNMENT ON 9,819 MISSED MATCHES (THEORETICAL MAXIMUMS) ---")
    print(f"  First significant word (fw3 >= 3):      {name_fw3_match:,} ({name_fw3_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  First significant word (fw4 >= 4):      {name_fw4_match:,} ({name_fw4_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Sorted first 2 tokens (order invariant): {name_sorted_tok2_match:,} ({name_sorted_tok2_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Name prefix 6 chars:                    {name_pfx6_match:,} ({name_pfx6_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Name prefix 10 chars:                   {name_pfx10_match:,} ({name_pfx10_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Second significant word:                {name_second_w_match:,} ({name_second_w_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Longest distinctive word (>=5 chars):   {name_longest_w_match:,} ({name_longest_w_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Transliterated stripped name:           {name_trans_sn_match:,} ({name_trans_sn_match/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Token subset containment:               {token_subset_containment:,} ({token_subset_containment/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Token Jaccard >= 0.50:                  {jaccard_ge_50:,} ({jaccard_ge_50/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  RapidFuzz ratio >= 70%:                 {rapidfuzz_ge_70:,} ({rapidfuzz_ge_70/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Shared address number (>=3 digits):     {addr_shared_num_ge3:,} ({addr_shared_num_ge3/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Shared address number + first word:     {addr_num_ge3_and_fw3:,} ({addr_num_ge3_and_fw3/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Shared postal code:                     {addr_shared_postal:,} ({addr_shared_postal/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Shared postal code + first word:        {postal_and_fw3:,} ({postal_and_fw3/len(baseline_missed_gt)*100:.2f}%)")
    print(f"  Shared postal code + first 2 chars:     {postal_and_fc2:,} ({postal_and_fc2/len(baseline_missed_gt)*100:.2f}%)\n")

    # 4. Rigorous Independent Key Evaluation with Exact Bucket Caps & Candidate Volume Measurement
    print("=== STEP 4: Independent Candidate Key & Bucket Cap Evaluation ===")
    t_eval = time.time()

    # Pre-extract validation S1 key values for candidate volume calculation
    # For each validation S1, and for each key, what are its keys?
    # We will measure the exact candidate bucket size for each S1!
    
    # Candidate definitions to test:
    # (name, key_extractor, freq_dict, default_cap, test_caps)
    # where key_extractor(rep) returns a list of keys for that entity.
    
    key_configs = [
        # Relaxed Baseline Caps
        {
            "name": "relaxed_tok2",
            "description": "Baseline first_2_tokens with expanded bucket caps",
            "extractor": lambda r: [(r["country"], r["first_two_tokens"])] if r["first_two_tokens"] else [],
            "freq": freq_tok2,
            "test_caps": [200, 300, 500, 1000, 2000, 999999],
            "baseline_rule": True,
        },
        {
            "name": "relaxed_pfx8",
            "description": "Baseline name_pfx8 with expanded bucket caps",
            "extractor": lambda r: [(r["country"], r["name_pfx8"])] if r["name_pfx8"] else [],
            "freq": freq_pfx8,
            "test_caps": [200, 300, 500, 1000, 2000, 999999],
            "baseline_rule": True,
        },
        {
            "name": "relaxed_addr_num_fc",
            "description": "Baseline addr_num_ge3 + first_char with expanded bucket caps",
            "extractor": lambda r: ([(r["country"], n, r["first_char"]) for n in r["addr_nums_sel"]] if r["first_char"] else []),
            "freq": bucket_counts_addr_num,
            "test_caps": [100, 200, 300, 500, 1000, 999999],
            "baseline_rule": True,
        },
        # New Candidate Keys
        {
            "name": "sorted_first_two_tokens",
            "description": "Alphabetically sorted first 2 significant tokens (word order invariant)",
            "extractor": lambda r: [(r["country"], r["sorted_tok2"])] if r["sorted_tok2"] else [],
            "freq": freq_sorted_tok2,
            "test_caps": [50, 100, 200, 500],
            "baseline_rule": False,
        },
        {
            "name": "first_word_min3",
            "description": "First significant word (min length 3)",
            "extractor": lambda r: [(r["country"], r["first_word"])] if r["first_word"] else [],
            "freq": freq_fw3,
            "test_caps": [25, 50, 100, 200],
            "baseline_rule": False,
        },
        {
            "name": "first_word_min4",
            "description": "First significant word (min length 4)",
            "extractor": lambda r: [(r["country"], r["first_word4"])] if r["first_word4"] else [],
            "freq": freq_fw4,
            "test_caps": [25, 50, 100, 200],
            "baseline_rule": False,
        },
        {
            "name": "name_pfx6",
            "description": "First 6 characters of name",
            "extractor": lambda r: [(r["country"], r["name_pfx6"])] if r["name_pfx6"] else [],
            "freq": freq_pfx6,
            "test_caps": [50, 100, 200],
            "baseline_rule": False,
        },
        {
            "name": "name_pfx10",
            "description": "First 10 characters of name (longer prefix, higher precision)",
            "extractor": lambda r: [(r["country"], r["name_pfx10"])] if r["name_pfx10"] else [],
            "freq": freq_pfx10,
            "test_caps": [100, 200, 500, 999999],
            "baseline_rule": False,
        },
        {
            "name": "second_word",
            "description": "Second significant word of name",
            "extractor": lambda r: [(r["country"], r["second_word"])] if r["second_word"] else [],
            "freq": freq_second_w,
            "test_caps": [25, 50, 100, 200],
            "baseline_rule": False,
        },
        {
            "name": "longest_word",
            "description": "Longest distinctive word (>=5 chars)",
            "extractor": lambda r: [(r["country"], r["longest_word"])] if r["longest_word"] else [],
            "freq": freq_longest_w,
            "test_caps": [25, 50, 100, 200],
            "baseline_rule": False,
        },
        {
            "name": "first_word_and_addr_num",
            "description": "Combination: First significant word + address number (>=3 digits)",
            "extractor": lambda r: ([(r["country"], r["first_word"], n) for n in r["addr_nums_sel"]] if r["first_word"] else []),
            "freq": freq_fw3_addr_num,
            "test_caps": [50, 100, 200, 500, 999999],
            "baseline_rule": False,
        },
        {
            "name": "first_word_and_postal",
            "description": "Combination: First significant word + postal/ZIP/PIN code",
            "extractor": lambda r: ([(r["country"], r["first_word"], r["postal_code"])] if (r["first_word"] and r["postal_code"]) else []),
            "freq": freq_fw3_postal,
            "test_caps": [50, 100, 200, 500, 999999],
            "baseline_rule": False,
        },
        {
            "name": "postal_and_first_two_chars",
            "description": "Combination: Postal code + first 2 characters of name",
            "extractor": lambda r: ([(r["country"], r["postal_code"], r["first_two_chars"])] if (r["postal_code"] and r["first_two_chars"]) else []),
            "freq": freq_postal_fc2,
            "test_caps": [50, 100, 200, 500],
            "baseline_rule": False,
        },
        {
            "name": "addr_num_and_first_two_chars",
            "description": "Combination: Address number + first 2 characters of name (tighter than 1 char)",
            "extractor": lambda r: ([(r["country"], n, r["first_two_chars"]) for n in r["addr_nums_sel"]] if r["first_two_chars"] else []),
            "freq": freq_addr_num_fc2,
            "test_caps": [50, 100, 200, 500],
            "baseline_rule": False,
        },
        {
            "name": "transliterated_tok2",
            "description": "NFKD ASCII transliterated first 2 tokens",
            "extractor": lambda r: [(r["country"], r["trans_tok2"])] if r["trans_tok2"] else [],
            "freq": freq_trans_tok2,
            "test_caps": [100, 200, 500],
            "baseline_rule": False,
        },
        {
            "name": "transliterated_stripped_name",
            "description": "NFKD ASCII transliterated stripped name (uncapped)",
            "extractor": lambda r: [(r["country"], r["trans_stripped"])] if r["trans_stripped"] else [],
            "freq": freq_trans_stripped,
            "test_caps": [999999],
            "baseline_rule": False,
        },
    ]

    results_table = []

    for cfg in key_configs:
        k_name = cfg["name"]
        extractor = cfg["extractor"]
        freq_tbl = cfg["freq"]
        test_caps = cfg["test_caps"]
        is_baseline = cfg["baseline_rule"]

        print(f"Evaluating {k_name}...")

        # Precompute keys for all validation S1s
        val_s1_keys = {}
        for sid, rep in val_s1_reps.items():
            val_s1_keys[sid] = extractor(rep)

        # Precompute keys for target entities of missed matches
        missed_target_keys = {}
        for sid, tid in baseline_missed_gt:
            tr = target_reps[tid]
            missed_target_keys[tid] = set(extractor(tr))

        for cap in test_caps:
            # 1. How many of the 9,819 missed matches are recovered by this key at this cap?
            recovered_missed = 0
            recovered_by_country = Counter()
            recovered_by_src = Counter()

            for sid, tid in baseline_missed_gt:
                s1_k_set = set(val_s1_keys[sid])
                t_k_set = missed_target_keys[tid]
                common = s1_k_set & t_k_set
                # Check if any common key satisfies bucket cap
                valid_common = [k for k in common if freq_tbl[k] <= cap]
                if valid_common:
                    recovered_missed += 1
                    c = val_s1_reps[sid]["country"]
                    src = "S3" if tid.startswith("S3-") else "S2"
                    recovered_by_country[c] += 1
                    recovered_by_src[src] += 1

            # 2. Candidate load on all 10,002 validation entities
            # For each S1, how many candidates would this key generate?
            s1_cand_counts = []
            for sid in val_s1_set:
                keys = val_s1_keys[sid]
                # Total target candidates generated across its keys (capped)
                # Since multiple keys could produce overlapping entities, the sum of bucket sizes
                # is an upper bound on unique candidates generated by this key alone.
                cand_count = sum(min(freq_tbl[k], cap) for k in keys if freq_tbl[k] <= cap)
                s1_cand_counts.append(cand_count)

            s1_cand_counts = np.array(s1_cand_counts)
            tot_cand = int(np.sum(s1_cand_counts))
            avg_cand = float(np.mean(s1_cand_counts))
            med_cand = float(np.median(s1_cand_counts))
            p95_cand = float(np.percentile(s1_cand_counts, 95))
            p99_cand = float(np.percentile(s1_cand_counts, 99))
            max_cand = int(np.max(s1_cand_counts))

            # Max bucket size observed in the target catalog for keys generated by validation S1
            max_bucket = max([freq_tbl[k] for keys in val_s1_keys.values() for k in keys], default=0)

            # Cumulative recall if added to baseline
            new_recovered_total = len(baseline_recovered_gt) + recovered_missed
            new_recall_pct = (new_recovered_total / val_gt_count) * 100.0
            missed_recovered_pct = (recovered_missed / len(baseline_missed_gt)) * 100.0

            # Scaled candidate projection to 1.73M test set
            # 1,732,544 test S1s * avg_cand
            est_test_candidates_added = int(round(1732544 * avg_cand))

            res = {
                "key_name": k_name,
                "description": cfg["description"],
                "cap": cap if cap < 999999 else "uncapped",
                "cap_num": cap,
                "is_baseline": is_baseline,
                "additional_gt_recovered": recovered_missed,
                "pct_missed_recovered": round(missed_recovered_pct, 2),
                "new_blocking_recall_pct": round(new_recall_pct, 2),
                "recovered_us": recovered_by_country["US"],
                "recovered_india": recovered_by_country["India"],
                "recovered_s2": recovered_by_src["S2"],
                "recovered_s3": recovered_by_src["S3"],
                "total_candidates_val": tot_cand,
                "avg_cand_per_s1": round(avg_cand, 2),
                "median_cand_per_s1": round(med_cand, 2),
                "p95_cand_per_s1": round(p95_cand, 2),
                "p99_cand_per_s1": round(p99_cand, 2),
                "max_cand_per_s1": max_cand,
                "max_catalog_bucket": max_bucket,
                "est_test_cand_added_millions": round(est_test_candidates_added / 1e6, 2),
            }
            results_table.append(res)

    print(f"Key evaluation finished in {time.time() - t_eval:.1f}s.\n")

    # 5. Evaluate Multi-Key Combinations (Portfolio Optimization)
    print("=== STEP 5: Combinatorial Portfolio Evaluation ===")
    # Evaluate adding top complementary keys together on top of the baseline
    # Let's test combinations:
    # Combo A: Baseline + sorted_first_two_tokens (cap 200)
    # Combo B: Baseline + sorted_tok2 (cap 200) + first_word_and_addr_num (cap 100)
    # Combo C: Baseline + sorted_tok2 (cap 200) + first_word_and_addr_num (cap 100) + transliterated_tok2 (cap 200)
    # Combo D: Baseline + sorted_tok2 (cap 200) + first_word_and_addr_num (cap 100) + first_word_min4 (cap 50)
    # Combo E: Baseline + sorted_tok2 (cap 200) + first_word_and_addr_num (cap 100) + postal_and_first_two_chars (cap 100)
    # Combo F: Relaxed baseline (tok2 cap 500, pfx8 cap 500) + sorted_tok2 (cap 200) + first_word_and_addr_num (cap 100)

    combos_to_test = [
        {
            "name": "Combo 1: Baseline + Sorted 2-Tokens (Cap 200)",
            "keys": [("sorted_first_two_tokens", 200)],
        },
        {
            "name": "Combo 2: Baseline + Sorted 2-Tokens (Cap 200) + FW+AddrNum (Cap 100)",
            "keys": [("sorted_first_two_tokens", 200), ("first_word_and_addr_num", 100)],
        },
        {
            "name": "Combo 3: Baseline + Sorted 2-Tokens (Cap 200) + FW+AddrNum (Cap 100) + Transliterated Tok2 (Cap 200)",
            "keys": [("sorted_first_two_tokens", 200), ("first_word_and_addr_num", 100), ("transliterated_tok2", 200)],
        },
        {
            "name": "Combo 4: Baseline + Sorted 2-Tokens (Cap 200) + FW+AddrNum (Cap 100) + Postal+FC2 (Cap 100)",
            "keys": [("sorted_first_two_tokens", 200), ("first_word_and_addr_num", 100), ("postal_and_first_two_chars", 100)],
        },
        {
            "name": "Combo 5: Baseline + Sorted 2-Tokens (Cap 200) + FW+AddrNum (Cap 100) + FW4 (Cap 50)",
            "keys": [("sorted_first_two_tokens", 200), ("first_word_and_addr_num", 100), ("first_word_min4", 50)],
        },
        {
            "name": "Combo 6: Relaxed Caps (Tok2 500, Pfx8 500) + Sorted 2-Tokens (Cap 200) + FW+AddrNum (Cap 100)",
            "keys": [("relaxed_tok2", 500), ("relaxed_pfx8", 500), ("sorted_first_two_tokens", 200), ("first_word_and_addr_num", 100)],
        },
    ]

    combo_results = []
    # Build dictionary of key extractors and freqs
    cfg_lookup = {c["name"]: c for c in key_configs}

    for combo in combos_to_test:
        c_name = combo["name"]
        key_list = combo["keys"]
        print(f"Testing {c_name}...")

        # A missed match is recovered by the combo if recovered by ANY key in key_list
        combo_recovered_missed = set()
        combo_cand_counts = np.zeros(len(val_s1_set), dtype=np.int32)

        for kname, cap in key_list:
            cfg = cfg_lookup[kname]
            ext = cfg["extractor"]
            freq = cfg["freq"]

            # Check missed matches
            for sid, tid in baseline_missed_gt:
                s1_keys = ext(val_s1_reps[sid])
                tr_keys = ext(target_reps[tid])
                common = set(s1_keys) & set(tr_keys)
                if any(freq[k] <= cap for k in common):
                    combo_recovered_missed.add((sid, tid))

            # Check candidate load
            for idx, sid in enumerate(sorted(list(val_s1_set))):
                keys = ext(val_s1_reps[sid])
                c_load = sum(min(freq[k], cap) for k in keys if freq[k] <= cap)
                combo_cand_counts[idx] += c_load

        total_rec = len(baseline_recovered_gt) + len(combo_recovered_missed)
        rec_pct = (total_rec / val_gt_count) * 100.0
        missed_pct = (len(combo_recovered_missed) / len(baseline_missed_gt)) * 100.0
        avg_add_cand = float(np.mean(combo_cand_counts))
        est_test_cand_add = int(round(1732544 * avg_add_cand))

        c_res = {
            "combo_name": c_name,
            "additional_gt_recovered": len(combo_recovered_missed),
            "pct_missed_recovered": round(missed_pct, 2),
            "new_blocking_recall_pct": round(rec_pct, 2),
            "avg_additional_candidates_per_s1": round(avg_add_cand, 2),
            "p95_additional_candidates_per_s1": round(float(np.percentile(combo_cand_counts, 95)), 2),
            "max_additional_candidates_per_s1": int(np.max(combo_cand_counts)),
            "est_test_candidates_added_millions": round(est_test_cand_add / 1e6, 2),
        }
        combo_results.append(c_res)
        print(f"  -> Recovered: +{len(combo_recovered_missed):,} ({missed_pct:.2f}% of missed) | New Recall: {rec_pct:.2f}% | Avg Add Cands: {avg_add_cand:.1f}")

    # 6. Save Machine-Readable JSON
    print("\n=== STEP 6: Saving Diagnostic Results JSON ===")
    total_runtime = time.time() - t_start
    diagnostic_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation_entities": len(val_s1_set),
        "validation_gt_matches": val_gt_count,
        "baseline_recall_ceiling": {
            "recovered": len(baseline_recovered_gt),
            "missed": len(baseline_missed_gt),
            "recall_pct": round((len(baseline_recovered_gt)/val_gt_count)*100, 4),
            "missed_clipped_by_caps": len(clipped_by_caps),
            "missed_clipped_by_caps_pct": round(len(clipped_by_caps)/len(baseline_missed_gt)*100, 2),
            "missed_representational": len(pure_representational_miss),
            "missed_representational_pct": round(len(pure_representational_miss)/len(baseline_missed_gt)*100, 2),
        },
        "missed_matches_breakdown": {
            "by_country": dict(missed_by_country),
            "by_source": dict(missed_by_src),
            "by_address_status": dict(missed_addr_status),
            "by_gt_bucket": dict(missed_by_gt_bucket),
            "root_causes": dict(error_patterns),
            "theoretical_key_alignments": {
                "first_word_min3": name_fw3_match,
                "first_word_min4": name_fw4_match,
                "sorted_first_two_tokens": name_sorted_tok2_match,
                "name_pfx6": name_pfx6_match,
                "name_pfx10": name_pfx10_match,
                "second_word": name_second_w_match,
                "longest_word": name_longest_w_match,
                "transliterated_stripped_name": name_trans_sn_match,
                "token_subset_containment": token_subset_containment,
                "token_jaccard_ge_50": jaccard_ge_50,
                "rapidfuzz_ge_70": rapidfuzz_ge_70,
                "shared_addr_num_ge3": addr_shared_num_ge3,
                "shared_addr_num_ge3_and_fw3": addr_num_ge3_and_fw3,
                "shared_postal": addr_shared_postal,
                "postal_and_fw3": postal_and_fw3,
                "postal_and_fc2": postal_and_fc2,
            }
        },
        "individual_key_evaluations": results_table,
        "combo_evaluations": combo_results,
        "runtime_seconds": round(total_runtime, 2),
        "peak_ram_mb": round(proc.memory_info().rss / (1024*1024), 2),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(diagnostic_data, f, indent=2)
    print(f"Saved machine-readable diagnostic to {OUTPUT_JSON}.\n")

    print("===================================================================")
    print(f"STAGE 8 DIAGNOSTIC EXECUTION COMPLETE in {total_runtime:.1f}s.")
    print("===================================================================")

if __name__ == "__main__":
    main()
