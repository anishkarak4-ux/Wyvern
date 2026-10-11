"""
Unit tests for Rule F gap pruning logic
"""

import pytest

def apply_rule_f_single_entity(candidates_with_scores, base_th=0.90, max_rank=5, max_gap=0.08):
    """
    Pure reference implementation of Rule F for testing.
    candidates_with_scores: list of (entity_id, score)
    """
    eligible = [(tid, s) for tid, s in candidates_with_scores if s >= base_th]
    if not eligible:
        return []
    
    eligible.sort(key=lambda x: x[1], reverse=True)
    p1 = eligible[0][1]
    
    selected = []
    for rank, (tid, p) in enumerate(eligible):
        if rank == 0:
            selected.append(tid)
        elif rank < max_rank and (p1 - p) <= max_gap:
            selected.append(tid)
            
    return selected

def test_rule_f_empty_below_threshold():
    candidates = [("S2-001", 0.85), ("S3-002", 0.82)]
    selected = apply_rule_f_single_entity(candidates, base_th=0.90)
    assert selected == []

def test_rule_f_single_dominant_match():
    candidates = [("S2-001", 0.95), ("S2-002", 0.84), ("S3-003", 0.81)]
    selected = apply_rule_f_single_entity(candidates, base_th=0.90)
    assert selected == ["S2-001"]

def test_rule_f_multi_match_within_gap():
    candidates = [
        ("S2-001", 0.96),
        ("S3-002", 0.94),  # gap = 0.02 <= 0.08 (Keep)
        ("S2-003", 0.91),  # gap = 0.05 <= 0.08 (Keep)
        ("S3-004", 0.87),  # below base_th (Drop)
    ]
    selected = apply_rule_f_single_entity(candidates, base_th=0.90, max_gap=0.08)
    assert selected == ["S2-001", "S3-002", "S2-003"]

def test_rule_f_prunes_distractors_with_large_gap():
    candidates = [
        ("S2-001", 0.99),
        ("S2-002", 0.90),  # gap = 0.09 > 0.08 (Drop)
    ]
    selected = apply_rule_f_single_entity(candidates, base_th=0.90, max_gap=0.08)
    assert selected == ["S2-001"]

def test_rule_f_max_rank_cap():
    # 7 candidates all above 0.90 and gap <= 0.08
    candidates = [(f"S2-00{i}", 0.95 - 0.01 * i) for i in range(7)]
    selected = apply_rule_f_single_entity(candidates, base_th=0.90, max_rank=5, max_gap=0.08)
    assert len(selected) == 5
    assert selected == [f"S2-00{i}" for i in range(5)]
