"""
apply_rule_f.py - Rule F Post-Processing for High-Precision Entity Resolution
Team Wyvern - Amazon ML Challenge 2026

Applies Rule F to candidate predictions:
1. Filters candidates by base threshold (default: tau = 0.90).
2. Sorts candidates by probability descending: p1 >= p2 >= ...
3. Retains Rank 1 candidate (p1).
4. Retains Ranks 2-5 only if (p1 - p) <= max_gap (default: 0.08).
5. Prunes candidates with rank > 5 or gap > 0.08 to eliminate false-positive branch explosions.
6. Emits matching_results.tsv in canonical S1 order.
"""

import os
import sys
import csv
import time
import argparse

# Setup portable relative paths
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "src"))

def apply_rule_f(input_parquet_or_tsv, test_s1_path, output_matching_path, base_th=0.90, max_rank=5, max_gap=0.08):
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_matching_path)), exist_ok=True)
    
    print(f"Reading canonical S1 order from {test_s1_path}...")
    canonical_s1_order = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            canonical_s1_order.append(r[0])
    print(f"Total S1 entities required: {len(canonical_s1_order):,}")
    
    transformed_matches = {}
    
    if input_parquet_or_tsv.endswith(".parquet"):
        import polars as pl
        print(f"Loading predictions from parquet: {input_parquet_or_tsv}...")
        df = pl.read_parquet(input_parquet_or_tsv, columns=["source1_entity_id", "matched_entity_ids", "match_probabilities"])
        s1_ids = df["source1_entity_id"].to_list()
        m_ids_list = df["matched_entity_ids"].to_list()
        m_probs_list = df["match_probabilities"].to_list()
        
        for i in range(len(s1_ids)):
            sid = s1_ids[i]
            m_ids_str = m_ids_list[i]
            m_probs_str = m_probs_list[i]
            
            if not m_ids_str:
                transformed_matches[sid] = ""
                continue
                
            tids = m_ids_str.split(",")
            probs = [float(p) for p in m_probs_str.split(",")]
            paired = sorted(zip(tids, probs), key=lambda x: x[1], reverse=True)
            p1 = paired[0][1]
            
            filtered = []
            for rank, (tid, p) in enumerate(paired):
                if p < base_th:
                    continue
                if rank == 0:
                    filtered.append(tid)
                elif rank < max_rank and (p1 - p) <= max_gap:
                    filtered.append(tid)
            transformed_matches[sid] = ",".join(filtered)
    else:
        # TSV format fallback
        print(f"Reading predictions from TSV: {input_parquet_or_tsv}...")
        with open(input_parquet_or_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                sid = r[0]
                m_str = r[1] if len(r) > 1 else ""
                if not m_str:
                    transformed_matches[sid] = ""
                    continue
                tids = m_str.split(",")
                transformed_matches[sid] = ",".join(tids[:max_rank])
                
    print(f"Writing final matching results to {output_matching_path}...")
    total_matches = 0
    empty_matches = 0
    with open(output_matching_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in canonical_s1_order:
            m_str = transformed_matches.get(sid, "")
            if not m_str:
                empty_matches += 1
            else:
                total_matches += len(m_str.split(","))
            f.write(f"{sid}\t{m_str}\n")
            
    print(f"Done in {time.time() - t0:.1f}s!")
    print(f"  Total S1 rows:      {len(canonical_s1_order):,}")
    print(f"  Total matches:      {total_matches:,}")
    print(f"  Empty singletons:   {empty_matches:,}")

def main():
    parser = argparse.ArgumentParser(description="Rule F Post-Processing")
    parser.add_argument("--input", default="data/processed/test_predictions_stage9.parquet", help="Path to predictions parquet")
    parser.add_argument("--test-s1", default="data/raw/test/test_source1.tsv", help="Path to test_source1.tsv")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Path to output matching_results.tsv")
    parser.add_argument("--base-th", type=float, default=0.90, help="Base probability threshold")
    parser.add_argument("--max-rank", type=int, default=5, help="Maximum matching rank")
    parser.add_argument("--max-gap", type=float, default=0.08, help="Maximum score gap from rank 1")
    args = parser.parse_args()
    
    apply_rule_f(args.input, args.test_s1, args.output, args.base_th, args.max_rank, args.max_gap)

if __name__ == "__main__":
    main()
