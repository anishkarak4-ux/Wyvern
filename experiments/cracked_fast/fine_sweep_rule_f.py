"""
fine_sweep_rule_f.py - Fine-grained grid sweep around Rule F and Rule B
Using cached val_p2_scored_candidates.json
"""

import json
import time
import os
import sys

BASE_DIR = r"C:\Users\KIIT\Desktop\Wyvern"
sys.path.insert(0, BASE_DIR)

from src.evaluation.entity_level_metrics import evaluate_population_entity_f05

EXP_DIR = os.path.join(BASE_DIR, "experiments", "cracked_fast")
CACHE_FILE = os.path.join(EXP_DIR, "val_p2_scored_candidates.json")

def evaluate_predictions(pred_dict, gt_dict):
    res = evaluate_population_entity_f05(pred_dict, gt_dict, beta=0.5)
    fp = sum(len(pred_dict.get(sid, set()) - gt_dict[sid]) for sid in gt_dict)
    fn = sum(len(gt_dict[sid] - pred_dict.get(sid, set())) for sid in gt_dict)
    return res["macro_f05"], res["macro_precision"], res["macro_recall"], fp, fn

def sweep():
    print(f"Loading {CACHE_FILE}...")
    with open(CACHE_FILE, "r") as f:
        data = json.load(f)
    val_data = data["val_data"]
    gt_dict = {sid: set(tids) for sid, tids in data["gt_matches"].items()}
    val_s1_list = sorted(list(val_data.keys()))
    
    # Baseline
    base_preds = {sid: {c["tid"] for c in val_data[sid]["cands"] if c["prob"] >= 0.90} for sid in val_s1_list}
    b_f05, b_p, b_r, b_fp, b_fn = evaluate_predictions(base_preds, gt_dict)
    print(f"Baseline (tau=0.90): F0.5 = {b_f05:.5f}, P = {b_p:.5f}, R = {b_r:.5f}, FP = {b_fp}, FN = {b_fn}")
    
    best = {
        "f05": 0.80462,
        "rule": "Rule F",
        "thresh": 0.89,
        "params": {"max_rank": 5, "max_gap": 0.08},
        "prec": 0.88185,
        "rec": 0.67582,
        "fp": 1853,
        "fn": 11808
    }
    
    t0 = time.time()
    
    # Sweep Rule F: base_th, max_rank, max_gap
    # Also test: what if rank 0 only requires a lower threshold (e.g. 0.88), while ranks 1+ require (p1 - p <= gap)?
    for base_th in [0.86, 0.87, 0.88, 0.89, 0.90, 0.91]:
        for max_rank in [2, 3, 4, 5, 6, 8, 10]:
            for max_gap in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]:
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
                if f05 > best["f05"]:
                    best.update({"f05": f05, "rule": "Rule F (fine)", "thresh": base_th, "params": {"max_rank": max_rank, "max_gap": max_gap}, "prec": p, "rec": r, "fp": fp, "fn": fn})
                    print(f"  * BETTER Rule F: base={base_th:.2f}, rank={max_rank}, gap={max_gap:.2f} -> F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f}, FP: {fp}, FN: {fn})")
                    
    # Also test Rule B fine: base_th and relative ratio
    for base_th in [0.85, 0.86, 0.87, 0.88, 0.89, 0.90]:
        for ratio in [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96]:
            preds = {}
            for sid in val_s1_list:
                cands = val_data[sid]["cands"]
                if not cands or cands[0]["prob"] < base_th:
                    preds[sid] = set()
                    continue
                p1 = cands[0]["prob"]
                preds[sid] = {c["tid"] for c in cands if c["prob"] >= base_th and (c["prob"] / p1) >= ratio}
            f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
            if f05 > best["f05"]:
                best.update({"f05": f05, "rule": "Rule B (fine)", "thresh": base_th, "params": {"ratio": ratio}, "prec": p, "rec": r, "fp": fp, "fn": fn})
                print(f"  * BETTER Rule B: base={base_th:.2f}, ratio={ratio:.2f} -> F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f}, FP: {fp}, FN: {fn})")

    # Also test Hybrid: p1 >= th1, and for rank >= 1, require p >= th2 AND (p1 - p <= gap)
    for th1 in [0.88, 0.89, 0.90]:
        for th2 in [0.90, 0.91, 0.92, 0.93]:
            for max_gap in [0.03, 0.05, 0.07, 0.10]:
                for max_rank in [3, 4, 5]:
                    preds = {}
                    for sid in val_s1_list:
                        cands = val_data[sid]["cands"]
                        if not cands or cands[0]["prob"] < th1:
                            preds[sid] = set()
                            continue
                        p1 = cands[0]["prob"]
                        res = set()
                        for rank, c in enumerate(cands):
                            if rank == 0 and c["prob"] >= th1:
                                res.add(c["tid"])
                            elif rank < max_rank and c["prob"] >= th2 and (p1 - c["prob"]) <= max_gap:
                                res.add(c["tid"])
                        preds[sid] = res
                    f05, p, r, fp, fn = evaluate_predictions(preds, gt_dict)
                    if f05 > best["f05"]:
                        best.update({"f05": f05, "rule": "Rule Hybrid", "thresh": th1, "params": {"th1": th1, "th2": th2, "gap": max_gap, "rank": max_rank}, "prec": p, "rec": r, "fp": fp, "fn": fn})
                        print(f"  * BETTER Hybrid: th1={th1}, th2={th2}, gap={max_gap}, rank={max_rank} -> F0.5 = {f05:.5f} (P: {p:.4f}, R: {r:.4f})")

    print(f"\nFine sweep complete in {time.time() - t0:.1f}s.")
    print("BEST CONFIGURATION FOUND:")
    print(json.dumps(best, indent=2))
    
    # Save best
    out_file = os.path.join(EXP_DIR, "fine_sweep_best.json")
    with open(out_file, "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    sweep()
