"""
run_cracked_experiment.py - 30-Minute Emergency Optimization
Team Wyvern - Amazon ML Challenge 2026

Work strictly in experiments/cracked_fast/
Never touch frozen production models, submission zip, or stage 7/9 artifacts.

1. Reconstruct exact 10,002 validation entities & ground truth.
2. Generate Portfolio 2 (P2) candidates and score with xgb_final_stage5.json.
3. Verify baseline Macro F0.5 = 0.79804 at tau = 0.90.
4. Brute force Rule A, B, C, D, E, F and Extra Cracked Score.
5. Report best results and decision.
"""

import os
import sys
import csv
import json
import time
import math
from collections import defaultdict, Counter
import numpy as np
import xgboost as xgb

# Project root
BASE_DIR = r"C:\Users\KIIT\Desktop\Wyvern"
sys.path.insert(0, BASE_DIR)

from src.preprocessing.normalization import (
    clean_text,
    strip_legal_suffixes,
    normalize_address,
    extract_address_numeric_tokens,
)
from src.features.pair_features import (
    FEATURE_NAMES,
    extract_postal_code,
    extract_pair_features,
)
from src.evaluation.entity_level_metrics import evaluate_population_entity_f05
from src.evaluation.run_stage8_5_experiment import reconstruct_validation_split

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "train")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, "xgb_final_stage5.json")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "cracked_fast")
CACHE_FILE = os.path.join(EXP_DIR, "val_p2_scored_candidates.json")

def make_entity_rep(eid, name, addr, country):
    norm_name = clean_text(name)
    sn = strip_legal_suffixes(norm_name)
    tokens = sn.split() if sn else norm_name.split()
    num_tok = len(tokens)
    
    tok2 = (tokens[0] + "_" + tokens[1]) if num_tok >= 2 else (tokens[0] if num_tok == 1 else "")
    pfx8 = sn[:8] if len(sn) >= 3 else ""
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
        "stripped_name": sn,
        "stripped_tokens": tokens,
        "first_two_tokens": tok2,
        "name_pfx8": pfx8,
        "first_word": fw3,
        "first_word4": fw4,
        "sorted_tok2": sorted_tok2,
        "norm_addr": norm_addr,
        "addr_numeric": nums_all,
        "addr_nums_sel": nums_sel,
        "first_char": fc,
        "postal_code": postal,
        "has_addr": bool(norm_addr),
    }

def get_or_build_candidates():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached validation candidates from {CACHE_FILE}...")
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        return data["val_data"], data["gt_matches"]
    
    print("Reconstructing validation split (10,002 entities)...")
    val_s1_set, s1_gt_matches, _ = reconstruct_validation_split()
    val_targets = set(tid for sid in val_s1_set for tid in s1_gt_matches[sid])
    
    print("Indexing validation S1 representations...")
    s1_path = os.path.join(RAW_DIR, "train_source1.tsv")
    val_s1_reps = {}
    
    idx_stripped = defaultdict(lambda: defaultdict(list))
    idx_tok2 = defaultdict(lambda: defaultdict(list))
    idx_pfx8 = defaultdict(lambda: defaultdict(list))
    idx_addr_num = defaultdict(lambda: defaultdict(list))
    idx_fw_addr = defaultdict(lambda: defaultdict(list))
    idx_sorted_tok2 = defaultdict(lambda: defaultdict(list))
    idx_fw4 = defaultdict(lambda: defaultdict(list))
    idx_rare_addr = defaultdict(lambda: defaultdict(list))
    
    # Load rare tokens if available
    rare_tokens_set = defaultdict(set)
    rare_json = os.path.join(BASE_DIR, "data", "processed", "test_catalog_rare_tokens.json")
    if os.path.exists(rare_json):
        try:
            with open(rare_json, "r") as f:
                r_dict = json.load(f)
                for c_name, tok_list in r_dict.items():
                    rare_tokens_set[c_name] = set(tok_list)
        except Exception as e:
            print("Note on rare tokens:", e)
            
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            sid = r[0]
            if sid in val_s1_set:
                rep = make_entity_rep(sid, r[1], r[2], r[3])
                val_s1_reps[sid] = rep
                c = rep["country"]
                
                if rep["stripped_name"]: idx_stripped[c][rep["stripped_name"]].append(sid)
                if rep["first_two_tokens"]: idx_tok2[c][rep["first_two_tokens"]].append(sid)
                if rep["name_pfx8"]: idx_pfx8[c][rep["name_pfx8"]].append(sid)
                if rep["first_char"]:
                    for num in rep["addr_nums_sel"]:
                        idx_addr_num[c][(num, rep["first_char"])].append(sid)
                if rep["first_word"]:
                    for num in rep["addr_nums_sel"]:
                        idx_fw_addr[c][(rep["first_word"], num)].append(sid)
                if rep["sorted_tok2"]: idx_sorted_tok2[c][rep["sorted_tok2"]].append(sid)
                if rep["first_word4"]: idx_fw4[c][rep["first_word4"]].append(sid)
                if rep["addr_nums_sel"]:
                    c_rare = rare_tokens_set.get(c, set())
                    for tok in rep["stripped_tokens"]:
                        if len(tok) >= 4 and (tok in c_rare or not c_rare):
                            for num in rep["addr_nums_sel"]:
                                idx_rare_addr[c][(tok, num)].append(sid)

    print("Loading production model xgb_final_stage5.json...")
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_FINAL_PATH)
    
    bucket_counts_tok2 = Counter()
    bucket_counts_pfx8 = Counter()
    bucket_counts_addr_num = Counter()
    bucket_counts_fw_addr = Counter()
    bucket_counts_sorted_tok2 = Counter()
    bucket_counts_fw4 = Counter()
    bucket_counts_rare_addr = Counter()
    
    # Store candidates per S1
    # sid -> list of {"tid": tid, "prob": p, "is_gt": bool, "name_sim": float, "addr_sim": float, "num_sim": float, "tok_overlap": float}
    val_candidates = defaultdict(list)
    val_cand_counts = defaultdict(int)
    
    cand_batch = []
    
    # Find feature indices for cracked extra test
    name_sort_idx = FEATURE_NAMES.index("name_token_sort_ratio")
    addr_sort_idx = FEATURE_NAMES.index("addr_token_sort_ratio")
    num_match_idx = FEATURE_NAMES.index("exact_numeric_match")
    jaccard_idx = FEATURE_NAMES.index("name_token_jaccard")
    
    def flush_batch():
        if not cand_batch: return
        n = len(cand_batch)
        X_batch = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        for i in range(n):
            _, _, s1r, tr, prov, s3_flag, _ = cand_batch[i]
            X_batch[i] = extract_pair_features(s1r, tr, prov, is_s3=s3_flag)
            
        probs = clf.predict_proba(X_batch)[:, 1]
        for i in range(n):
            sid, tid, _, _, _, _, is_gt = cand_batch[i]
            prob = float(probs[i])
            val_cand_counts[sid] += 1
            # Save only candidates with prob >= 0.35 or if it's ground truth, to keep memory tight
            if prob >= 0.35 or is_gt:
                val_candidates[sid].append({
                    "tid": tid,
                    "prob": round(prob, 4),
                    "is_gt": bool(is_gt),
                    "name_sim": round(float(X_batch[i, name_sort_idx]) / 100.0, 4),
                    "addr_sim": round(float(X_batch[i, addr_sort_idx]) / 100.0, 4),
                    "num_sim": round(float(X_batch[i, num_match_idx]), 4),
                    "tok_overlap": round(float(X_batch[i, jaccard_idx]), 4),
                })
        cand_batch.clear()
        
    def stream_file(filepath, src_label):
        t_f = time.time()
        is_s3 = (src_label == "S3")
        print(f"Streaming {src_label} ({os.path.basename(filepath)})...")
        rows = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                rows += 1
                tid = r[0]
                t_name, t_addr = r[1], r[2]
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
                
                matched_s1s = {}
                # Rule 1: Stripped name
                if sn and sn in idx_stripped[t_c]:
                    for sid in idx_stripped[t_c][sn]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 1
                # Rule 2: First 2 tokens
                if tok2 and bucket_counts_tok2[(t_c, tok2)] <= 200 and tok2 in idx_tok2[t_c]:
                    for sid in idx_tok2[t_c][tok2]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 2
                # Rule 3: Name prefix 8
                if pfx8 and bucket_counts_pfx8[(t_c, pfx8)] <= 200 and pfx8 in idx_pfx8[t_c]:
                    for sid in idx_pfx8[t_c][pfx8]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 4
                # Rule 4: Addr num + first char
                if fc:
                    for num in t_nums_sel:
                        bucket_counts_addr_num[(t_c, num, fc)] += 1
                        if bucket_counts_addr_num[(t_c, num, fc)] <= 100 and (num, fc) in idx_addr_num[t_c]:
                            for sid in idx_addr_num[t_c][(num, fc)]:
                                matched_s1s[sid] = matched_s1s.get(sid, 0) | 8
                # Rule 5: First word + addr num
                if fw3:
                    for num in t_nums_sel:
                        bucket_counts_fw_addr[(t_c, fw3, num)] += 1
                        if bucket_counts_fw_addr[(t_c, fw3, num)] <= 100 and (fw3, num) in idx_fw_addr[t_c]:
                            for sid in idx_fw_addr[t_c][(fw3, num)]:
                                matched_s1s[sid] = matched_s1s.get(sid, 0) | 16
                # Rule 6: Sorted first 2 tokens
                if sorted_tok2 and bucket_counts_sorted_tok2[(t_c, sorted_tok2)] <= 200 and sorted_tok2 in idx_sorted_tok2[t_c]:
                    for sid in idx_sorted_tok2[t_c][sorted_tok2]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 32
                # Rule 7: First word >= 4 chars
                if fw4 and bucket_counts_fw4[(t_c, fw4)] <= 50 and fw4 in idx_fw4[t_c]:
                    for sid in idx_fw4[t_c][fw4]:
                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 64
                # Rule 8: Rare token + addr num
                if t_nums_sel:
                    c_rare = rare_tokens_set.get(t_c, set())
                    for tok in tokens:
                        if len(tok) >= 4 and (tok in c_rare or not c_rare):
                            for num in t_nums_sel:
                                bucket_counts_rare_addr[(t_c, tok, num)] += 1
                                if bucket_counts_rare_addr[(t_c, tok, num)] <= 100 and (tok, num) in idx_rare_addr[t_c]:
                                    for sid in idx_rare_addr[t_c][(tok, num)]:
                                        matched_s1s[sid] = matched_s1s.get(sid, 0) | 128

                if matched_s1s:
                    t_rep = {
                        "entity_id": tid,
                        "country": t_c,
                        "norm_name": norm_name,
                        "stripped_name": sn,
                        "first_two_tokens": tok2,
                        "name_pfx8": pfx8,
                        "first_word": fw3,
                        "first_word4": fw4,
                        "name_tokens": set(norm_name.split()),
                        "norm_addr": norm_addr,
                        "addr_tokens": set(norm_addr.split()) if norm_addr else set(),
                        "addr_numeric": t_nums_all,
                        "addr_nums_sel": t_nums_sel,
                        "first_char": fc,
                        "postal_code": extract_postal_code(norm_addr, t_c),
                    }
                    for sid, bitmask in matched_s1s.items():
                        prov = {
                            "stripped": bool(bitmask & 1),
                            "tok2": bool(bitmask & 2),
                            "pfx8": bool(bitmask & 4),
                            "addr_num": bool(bitmask & 8),
                        }
                        is_gt = (tid in s1_gt_matches[sid])
                        cand_batch.append((sid, tid, val_s1_reps[sid], t_rep, prov, is_s3, is_gt))
                        if len(cand_batch) >= 10000:
                            flush_batch()
                            
                if rows % 2000000 == 0:
                    print(f"  {src_label}: {rows:,} rows scanned ({time.time() - t_f:.1f}s)")
                    
        flush_batch()
        print(f"Finished {src_label} in {time.time() - t_f:.1f}s.")

    stream_file(os.path.join(RAW_DIR, "train_source2.tsv"), "S2")
    stream_file(os.path.join(RAW_DIR, "train_source3.tsv"), "S3")

    # Format data for fast vectorized rule evaluation
    val_data = {}
    for sid in sorted(list(val_s1_set)):
        cands = val_candidates.get(sid, [])
        # sort descending by prob
        cands.sort(key=lambda x: x["prob"], reverse=True)
        total_c = max(val_cand_counts[sid], len(cands))
        val_data[sid] = {
            "total_cand_count": total_c,
            "cands": cands,
        }
        
    gt_matches_clean = {sid: list(s1_gt_matches[sid]) for sid in sorted(list(val_s1_set))}
    
    print(f"Saving cache to {CACHE_FILE}...")
    with open(CACHE_FILE, "w") as f:
        json.dump({"val_data": val_data, "gt_matches": gt_matches_clean}, f)
        
    return val_data, gt_matches_clean

def evaluate_predictions(pred_dict, gt_dict):
    """Compute official entity-level Macro F0.5."""
    res = evaluate_population_entity_f05(pred_dict, gt_dict, beta=0.5)
    macro_f05 = res["macro_f05"]
    prec = res["macro_precision"]
    rec = res["macro_recall"]
    # Count FP, FN
    fp = 0
    fn = 0
    for sid, true_set in gt_dict.items():
        p_set = pred_dict.get(sid, set())
        fp += len(p_set - true_set)
        fn += len(true_set - p_set)
    return macro_f05, prec, rec, fp, fn

def run_experiment():
    t_start = time.time()
    val_data, gt_matches_list = get_or_build_candidates()
    gt_dict = {sid: set(tids) for sid, tids in gt_matches_list.items()}
    val_s1_list = sorted(list(val_data.keys()))
    
    print("\n" + "=" * 80)
    print("VERIFYING BASELINE ON VALIDATION SET")
    print("=" * 80)
    # Baseline: Rule A with threshold = 0.90
    base_preds = {}
    for sid in val_s1_list:
        base_preds[sid] = {c["tid"] for c in val_data[sid]["cands"] if c["prob"] >= 0.90}
    b_f05, b_p, b_r, b_fp, b_fn = evaluate_predictions(base_preds, gt_dict)
    print(f"Verified Baseline (tau = 0.90):")
    print(f"  Macro F0.5: {b_f05:.5f}")
    print(f"  Precision:  {b_p:.5f}")
    print(f"  Recall:     {b_r:.5f}")
    print(f"  FP: {b_fp:,} | FN: {b_fn:,}")
    
    best_overall = {
        "f05": b_f05,
        "rule": "Baseline (tau = 0.90)",
        "thresh": 0.90,
        "params": {},
        "prec": b_p,
        "rec": b_r,
        "fp": b_fp,
        "fn": b_fn,
    }
    
    thresholds = [0.80, 0.82, 0.84, 0.86, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
    relative_ratios = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    score_gaps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    
    # -------------------------------------------------------------
    # RULE A: Standard Threshold Sweep
    # -------------------------------------------------------------
    print("\n--- Testing RULE A: Simple Probability Threshold ---")
    for th in thresholds:
        preds = {sid: {c["tid"] for c in val_data[sid]["cands"] if c["prob"] >= th} for sid in val_s1_list}
        f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
        print(f"  th = {th:.2f} -> Macro F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f}, FP: {fp}, FN: {fn})")
        if f05 > best_overall["f05"]:
            best_overall.update({"f05": f05, "rule": "Rule A", "thresh": th, "params": {}, "prec": p, "rec": r, "fp": fp, "fn": fn})

    # -------------------------------------------------------------
    # RULE B: Relative Ratio (p >= th AND p / p1 >= ratio)
    # -------------------------------------------------------------
    print("\n--- Testing RULE B: Relative Ratio to Top Candidate (p/p1 >= ratio) ---")
    for th in [0.86, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]:
        for ratio in relative_ratios:
            preds = {}
            for sid in val_s1_list:
                cands = val_data[sid]["cands"]
                if not cands or cands[0]["prob"] < th:
                    preds[sid] = set()
                    continue
                p1 = cands[0]["prob"]
                preds[sid] = {c["tid"] for c in cands if c["prob"] >= th and (c["prob"] / p1) >= ratio}
            f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
            if f05 > best_overall["f05"]:
                best_overall.update({"f05": f05, "rule": "Rule B", "thresh": th, "params": {"ratio": ratio}, "prec": p, "rec": r, "fp": fp, "fn": fn})
                print(f"  * NEW BEST (Rule B): th={th:.2f}, ratio={ratio:.2f} -> Macro F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f})")

    # -------------------------------------------------------------
    # RULE C: Score Gap (p >= th AND (p1 - p) <= gap)
    # -------------------------------------------------------------
    print("\n--- Testing RULE C: Score Gap to Top Candidate (p1 - p <= gap) ---")
    for th in [0.86, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]:
        for gap in score_gaps:
            preds = {}
            for sid in val_s1_list:
                cands = val_data[sid]["cands"]
                if not cands or cands[0]["prob"] < th:
                    preds[sid] = set()
                    continue
                p1 = cands[0]["prob"]
                preds[sid] = {c["tid"] for c in cands if c["prob"] >= th and (p1 - c["prob"]) <= gap}
            f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
            if f05 > best_overall["f05"]:
                best_overall.update({"f05": f05, "rule": "Rule C", "thresh": th, "params": {"gap": gap}, "prec": p, "rec": r, "fp": fp, "fn": fn})
                print(f"  * NEW BEST (Rule C): th={th:.2f}, gap={gap:.3f} -> Macro F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f})")

    # -------------------------------------------------------------
    # RULE D: Adaptive Threshold by Candidate-Count Group
    # Groups: 1, 2-5, 6-20, 21-100, 100+
    # -------------------------------------------------------------
    print("\n--- Testing RULE D: Candidate-Count Adaptive Thresholding ---")
    # For large candidate pools, false positive risk increases; test stricter threshold on high candidate counts
    # and slightly more permissive on small pools
    candidate_schemes = [
        # (th_1, th_2_5, th_6_20, th_21_100, th_100p)
        (0.88, 0.89, 0.90, 0.92, 0.94),
        (0.86, 0.88, 0.90, 0.91, 0.93),
        (0.89, 0.90, 0.90, 0.91, 0.92),
        (0.90, 0.90, 0.91, 0.92, 0.95),
        (0.88, 0.90, 0.91, 0.93, 0.96),
        (0.85, 0.88, 0.90, 0.92, 0.95),
    ]
    for s in candidate_schemes:
        t1, t2, t3, t4, t5 = s
        preds = {}
        for sid in val_s1_list:
            n_c = val_data[sid]["total_cand_count"]
            if n_c <= 1: th = t1
            elif n_c <= 5: th = t2
            elif n_c <= 20: th = t3
            elif n_c <= 100: th = t4
            else: th = t5
            preds[sid] = {c["tid"] for c in val_data[sid]["cands"] if c["prob"] >= th}
        f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
        if f05 > best_overall["f05"]:
            best_overall.update({"f05": f05, "rule": "Rule D", "thresh": 0.90, "params": {"scheme": s}, "prec": p, "rec": r, "fp": fp, "fn": fn})
            print(f"  * NEW BEST (Rule D): scheme={s} -> Macro F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f})")

    # -------------------------------------------------------------
    # RULE E: Dominant Top Candidate (p1 >> p2 => stricter on p2+)
    # -------------------------------------------------------------
    print("\n--- Testing RULE E: Dominant Top Candidate Pruning ---")
    for base_th in [0.89, 0.90, 0.91]:
        for p1_high in [0.95, 0.97, 0.98]:
            for p2_penalty in [0.03, 0.05, 0.08]:
                preds = {}
                for sid in val_s1_list:
                    cands = val_data[sid]["cands"]
                    if not cands:
                        preds[sid] = set()
                        continue
                    p1 = cands[0]["prob"]
                    res = set()
                    for idx, c in enumerate(cands):
                        req_th = base_th
                        if idx > 0 and p1 >= p1_high:
                            req_th = min(base_th + p2_penalty, 0.99)
                        if c["prob"] >= req_th:
                            res.add(c["tid"])
                    preds[sid] = res
                f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
                if f05 > best_overall["f05"]:
                    best_overall.update({"f05": f05, "rule": "Rule E", "thresh": base_th, "params": {"p1_high": p1_high, "penalty": p2_penalty}, "prec": p, "rec": r, "fp": fp, "fn": fn})
                    print(f"  * NEW BEST (Rule E): base={base_th}, p1_high={p1_high}, pen={p2_penalty} -> Macro F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f})")

    # -------------------------------------------------------------
    # RULE F: Combined Probability + Rank + Score Gap
    # -------------------------------------------------------------
    print("\n--- Testing RULE F: Combined Rank + Score Gap ---")
    for base_th in [0.89, 0.90, 0.91]:
        for max_rank in [2, 3, 5]:
            for max_gap in [0.02, 0.05, 0.08]:
                preds = {}
                for sid in val_s1_list:
                    cands = val_data[sid]["cands"]
                    if not cands or cands[0]["prob"] < base_th:
                        preds[sid] = set()
                        continue
                    p1 = cands[0]["prob"]
                    res = set()
                    for rank, c in enumerate(cands):
                        if c["prob"] >= base_th:
                            if rank == 0:
                                res.add(c["tid"])
                            elif rank < max_rank and (p1 - c["prob"]) <= max_gap:
                                res.add(c["tid"])
                    preds[sid] = res
                f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
                if f05 > best_overall["f05"]:
                    best_overall.update({"f05": f05, "rule": "Rule F", "thresh": base_th, "params": {"max_rank": max_rank, "max_gap": max_gap}, "prec": p, "rec": r, "fp": fp, "fn": fn})
                    print(f"  * NEW BEST (Rule F): base={base_th}, rank={max_rank}, gap={max_gap} -> Macro F0.5 = {f05:.5f}")

    # -------------------------------------------------------------
    # EXTRA CRACKED TEST: Weighted Score with Similarity Features
    # score2 = 0.60 * prob + 0.15 * name_sim + 0.10 * addr_sim + 0.10 * num_sim + 0.05 * tok_overlap
    # -------------------------------------------------------------
    print("\n--- Testing EXTRA CRACKED TEST: Multi-Feature Weighted Score ---")
    for score2_th in [0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]:
        preds = {}
        for sid in val_s1_list:
            cands = val_data[sid]["cands"]
            res = set()
            for c in cands:
                s2 = (0.60 * c["prob"] +
                      0.15 * c["name_sim"] +
                      0.10 * c["addr_sim"] +
                      0.10 * c["num_sim"] +
                      0.05 * c["tok_overlap"])
                if s2 >= score2_th and c["prob"] >= 0.70: # safeguard prob
                    res.add(c["tid"])
            preds[sid] = res
        f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
        print(f"  score2_th = {score2_th:.2f} -> Macro F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f}, FP: {fp}, FN: {fn})")
        if f05 > best_overall["f05"]:
            best_overall.update({"f05": f05, "rule": "Extra Cracked (score2)", "thresh": score2_th, "params": {}, "prec": p, "rec": r, "fp": fp, "fn": fn})
            print(f"  * NEW BEST (score2): th={score2_th:.2f} -> Macro F0.5 = {f05:.5f}")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL EMERGENCY OPTIMIZATION REPORT")
    print("=" * 80)
    improvement = best_overall["f05"] - b_f05
    print(f"BASELINE F0.5:   {b_f05:.5f}")
    print(f"BEST F0.5:       {best_overall['f05']:.5f}")
    print(f"IMPROVEMENT:     {improvement:+.5f} ({improvement/b_f05*100:+.2f}%)")
    print(f"RULE:            {best_overall['rule']}")
    print(f"THRESHOLD:       {best_overall['thresh']}")
    print(f"PARAMETERS:      {json.dumps(best_overall['params'])}")
    print(f"PRECISION:       {best_overall['prec']:.5f}")
    print(f"RECALL:          {best_overall['rec']:.5f}")
    print(f"FP:              {best_overall['fp']:,}")
    print(f"FN:              {best_overall['fn']:,}")
    print(f"Total Runtime:   {time.time() - t_start:.1f}s")
    
    # Save results to json
    res_path = os.path.join(EXP_DIR, "cracked_optimization_results.json")
    with open(res_path, "w") as f:
        json.dump({
            "baseline_f05": b_f05,
            "best_f05": best_overall["f05"],
            "improvement": improvement,
            "best_config": best_overall,
            "runtime_seconds": time.time() - t_start,
        }, f, indent=2)
    print(f"Results written to {res_path}")

if __name__ == "__main__":
    run_experiment()
