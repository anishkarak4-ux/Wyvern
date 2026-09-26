"""
entity_level_metrics.py - Official Competition-Style Entity-Level Macro F0.5 Evaluation
Amazon ML Challenge 2026

Evaluates entity resolution predictions at the Source 1 entity level.
For each Source 1 entity:
- Evaluates predicted matched target IDs against ground truth.
- Computes entity-level TP, FP, FN, Precision, Recall, and F0.5.
- Grants full credit (F0.5 = 1.0) when a true singleton (0 GT matches)
  is correctly predicted with 0 matches.
- Averages per-entity F0.5 across the entire S1 population (Macro F0.5).
"""

import numpy as np
from collections import defaultdict

def calc_f_beta(prec, rec, beta=0.5):
    """
    Compute F-beta score.
    For beta=0.5:
    F0.5 = (1 + 0.25) * P * R / (0.25 * P + R) = 1.25 * P * R / (0.25 * P + R)
    """
    b2 = beta ** 2
    denom = (b2 * prec) + rec
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * (prec * rec) / denom

def calculate_entity_metrics(pred_ids, gt_ids, beta=0.5):
    """
    Calculate entity-level metrics for a single Source 1 entity.
    
    Parameters:
    - pred_ids: iterable of predicted target entity IDs (e.g. ['S2-123', 'S3-456'])
    - gt_ids: iterable of ground-truth target entity IDs
    - beta: float, default 0.5 (competition metric)
    
    Returns:
    - dict with keys:
        'tp', 'fp', 'fn', 'precision', 'recall', 'f_beta', 'num_pred', 'num_gt'
    """
    pred_set = set(pred_ids) if pred_ids else set()
    gt_set = set(gt_ids) if gt_ids else set()
    
    num_pred = len(pred_set)
    num_gt = len(gt_set)
    
    # Case 1: True singleton (0 ground truth matches)
    if num_gt == 0:
        if num_pred == 0:
            # Correctly predicted zero matches -> Full credit!
            return {
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 1.0, "recall": 1.0, "f_beta": 1.0,
                "num_pred": 0, "num_gt": 0, "is_singleton": True,
            }
        else:
            # False positive predictions on a singleton -> Zero credit
            return {
                "tp": 0, "fp": num_pred, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f_beta": 0.0,
                "num_pred": num_pred, "num_gt": 0, "is_singleton": True,
            }

    # Case 2: Matching entity (>= 1 ground truth matches)
    if num_pred == 0:
        # Failed to predict any matches -> Zero credit
        return {
            "tp": 0, "fp": 0, "fn": num_gt,
            "precision": 0.0, "recall": 0.0, "f_beta": 0.0,
            "num_pred": 0, "num_gt": num_gt, "is_singleton": False,
        }

    # Case 3: Has matches and predicted matches
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    prec = tp / num_pred if num_pred > 0 else 0.0
    rec = tp / num_gt if num_gt > 0 else 0.0
    fb = calc_f_beta(prec, rec, beta=beta)
    
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f_beta": fb,
        "num_pred": num_pred, "num_gt": num_gt, "is_singleton": False,
    }

def evaluate_population_entity_f05(entity_predictions, entity_ground_truth, entity_metadata=None, beta=0.5):
    """
    Evaluate macro entity-level F0.5 across an entire Source 1 entity population.
    
    Parameters:
    - entity_predictions: dict mapping s1_id -> set/list of predicted target IDs
    - entity_ground_truth: dict mapping s1_id -> set/list of true ground-truth target IDs
    - entity_metadata: dict mapping s1_id -> dict with 'country' and other metadata (optional)
    - beta: float, default 0.5
    
    Returns:
    - dict containing population-level macro and micro metrics, match group breakdowns,
      and subgroup performance.
    """
    all_s1_ids = sorted(list(entity_ground_truth.keys()))
    n_entities = len(all_s1_ids)
    
    per_entity_results = []
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    
    pred_counts = []
    
    # Groups: 0, 1, 2-3, 4-5, 6+
    match_group_stats = defaultdict(lambda: {"f05_list": [], "prec_list": [], "rec_list": [], "pred_counts": [], "count": 0})
    
    # Country groups: US, India
    country_stats = defaultdict(lambda: {"f05_list": [], "prec_list": [], "rec_list": [], "count": 0})
    
    for sid in all_s1_ids:
        gt_ids = entity_ground_truth.get(sid, set())
        pred_ids = entity_predictions.get(sid, set())
        
        m = calculate_entity_metrics(pred_ids, gt_ids, beta=beta)
        per_entity_results.append(m)
        
        tot_tp += m["tp"]
        tot_fp += m["fp"]
        tot_fn += m["fn"]
        pred_counts.append(m["num_pred"])
        
        # Categorize match group
        num_gt = m["num_gt"]
        if num_gt == 0:
            mg = "0_matches"
        elif num_gt == 1:
            mg = "1_match"
        elif num_gt <= 3:
            mg = "2-3_matches"
        elif num_gt <= 5:
            mg = "4-5_matches"
        else:
            mg = "6+_matches"
            
        match_group_stats[mg]["f05_list"].append(m["f_beta"])
        match_group_stats[mg]["prec_list"].append(m["precision"])
        match_group_stats[mg]["rec_list"].append(m["recall"])
        match_group_stats[mg]["pred_counts"].append(m["num_pred"])
        match_group_stats[mg]["count"] += 1
        
        # Country categorization if available
        if entity_metadata and sid in entity_metadata:
            c = entity_metadata[sid].get("country", "Unknown")
            country_stats[c]["f05_list"].append(m["f_beta"])
            country_stats[c]["prec_list"].append(m["precision"])
            country_stats[c]["rec_list"].append(m["recall"])
            country_stats[c]["count"] += 1

    # Macro averages (primary competition metric)
    macro_f05 = float(np.mean([m["f_beta"] for m in per_entity_results]))
    macro_precision = float(np.mean([m["precision"] for m in per_entity_results]))
    macro_recall = float(np.mean([m["recall"] for m in per_entity_results]))
    
    # Micro metrics (aggregated pairs)
    micro_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f05 = calc_f_beta(micro_prec, micro_rec, beta=beta)
    
    # Prediction count distributions
    pred_counts.sort()
    n_p = len(pred_counts)
    
    # Match group summary
    group_summary = {}
    for mg, gdata in sorted(match_group_stats.items()):
        group_summary[mg] = {
            "entity_count": gdata["count"],
            "avg_predicted": float(np.mean(gdata["pred_counts"])),
            "median_predicted": float(np.median(gdata["pred_counts"])),
            "macro_f05": float(np.mean(gdata["f05_list"])),
            "macro_precision": float(np.mean(gdata["prec_list"])),
            "macro_recall": float(np.mean(gdata["rec_list"])),
        }
        
    # Country summary
    country_summary = {}
    for c, cdata in sorted(country_stats.items()):
        country_summary[c] = {
            "entity_count": cdata["count"],
            "macro_f05": float(np.mean(cdata["f05_list"])),
            "macro_precision": float(np.mean(cdata["prec_list"])),
            "macro_recall": float(np.mean(cdata["rec_list"])),
        }

    return {
        "n_entities": n_entities,
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f05": micro_f05,
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "total_fn": tot_fn,
        "pred_matches_distribution": {
            "total_predicted_matches": sum(pred_counts),
            "zero_predicted_entities": sum(1 for c in pred_counts if c == 0),
            "one_predicted_entities": sum(1 for c in pred_counts if c == 1),
            "multi_predicted_entities": sum(1 for c in pred_counts if c >= 2),
            "avg_predicted": sum(pred_counts) / n_p,
            "median_predicted": pred_counts[n_p // 2],
            "p95_predicted": pred_counts[int(n_p * 0.95)],
            "max_predicted": pred_counts[-1],
        },
        "match_group_summary": group_summary,
        "country_summary": country_summary,
    }
