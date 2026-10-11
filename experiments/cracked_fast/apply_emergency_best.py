"""
apply_emergency_best.py - Apply Rule F to Existing Stage 9 Test Predictions
Amazon ML Challenge 2026

Rule F:
- Filter existing Stage 9 accepted matches (p >= 0.90)
- Sort descending by probability: p1 >= p2 >= ...
- Keep rank 1 (p1)
- For rank 2..5: keep if (p1 - p) <= 0.08
- Drop rank > 5 or gap > 0.08
- Save to submissions/emergency_best/matching_results.tsv
- Align row order with test_source1.tsv
"""

import os
import sys
import csv
import time
import polars as pl

BASE_DIR = r"C:\Users\KIIT\Desktop\Wyvern"
STAGE9_PARQUET = os.path.join(BASE_DIR, "data", "processed", "test_predictions_stage9.parquet")
TEST_S1_PATH = os.path.join(BASE_DIR, "data", "raw", "test", "test_source1.tsv")
OUT_DIR = os.path.join(BASE_DIR, "submissions", "emergency_best")
OUT_MATCHING = os.path.join(OUT_DIR, "matching_results.tsv")

def apply_rule():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading test predictions from {STAGE9_PARQUET}...")
    
    # Read parquet
    df = pl.read_parquet(STAGE9_PARQUET, columns=["source1_entity_id", "matched_entity_ids", "match_probabilities"])
    print(f"Loaded {len(df):,} rows from parquet in {time.time() - t0:.1f}s.")
    
    # Read canonical order from test_source1.tsv
    print("Reading canonical S1 order...")
    canonical_s1_order = []
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            canonical_s1_order.append(r[0])
            
    print(f"Canonical S1 count: {len(canonical_s1_order):,}")
    
    # Transform rows
    print("Applying Rule F (max_rank=5, max_gap=0.08)...")
    transformed_matches = {}
    
    orig_total_matches = 0
    new_total_matches = 0
    orig_zero_matches = 0
    new_zero_matches = 0
    pruned_by_rank = 0
    pruned_by_gap = 0
    
    s1_ids = df["source1_entity_id"].to_list()
    m_ids_list = df["matched_entity_ids"].to_list()
    m_probs_list = df["match_probabilities"].to_list()
    
    for i in range(len(s1_ids)):
        sid = s1_ids[i]
        m_ids_str = m_ids_list[i]
        m_probs_str = m_probs_list[i]
        
        if not m_ids_str:
            transformed_matches[sid] = ""
            orig_zero_matches += 1
            new_zero_matches += 1
            continue
            
        tids = m_ids_str.split(",")
        probs = [float(p) for p in m_probs_str.split(",")]
        orig_total_matches += len(tids)
        
        # Sort descending by probability
        paired = sorted(zip(tids, probs), key=lambda x: x[1], reverse=True)
        p1 = paired[0][1]
        
        filtered_tids = []
        for rank, (tid, prob) in enumerate(paired):
            if rank == 0:
                filtered_tids.append(tid)
            elif rank < 5:
                if (p1 - prob) <= 0.08:
                    filtered_tids.append(tid)
                else:
                    pruned_by_gap += 1
            else:
                pruned_by_rank += 1
                
        new_total_matches += len(filtered_tids)
        if not filtered_tids:
            new_zero_matches += 1
            transformed_matches[sid] = ""
        else:
            transformed_matches[sid] = ",".join(filtered_tids)
            
    print(f"\nRule F Transformation Statistics:")
    print(f"  Original matches:      {orig_total_matches:,}")
    print(f"  New matches:           {new_total_matches:,} (Delta: {new_total_matches - orig_total_matches:,})")
    print(f"  Pruned by rank (>5):   {pruned_by_rank:,}")
    print(f"  Pruned by gap (>0.08): {pruned_by_gap:,}")
    print(f"  Original singletons:   {orig_zero_matches:,}")
    print(f"  New singletons:        {new_zero_matches:,}")
    
    # Write matching_results.tsv in canonical order
    print(f"\nWriting {OUT_MATCHING} in canonical order...")
    with open(OUT_MATCHING, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in canonical_s1_order:
            m_str = transformed_matches.get(sid, "")
            f.write(f"{sid}\t{m_str}\n")
            
    out_size_mb = os.path.getsize(OUT_MATCHING) / (1024 * 1024)
    print(f"Successfully generated {OUT_MATCHING} ({out_size_mb:.2f} MB) in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    apply_rule()
