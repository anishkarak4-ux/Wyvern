"""
Unit tests for src.features.pair_features
"""

import numpy as np
import pytest
from src.features.pair_features import extract_pair_features, FEATURE_NAMES

def test_extract_pair_features_dimension():
    s1_rep = {
        "country": "US",
        "norm_name": "acme corporation",
        "stripped_name": "acme",
        "first_two_tokens": "acme",
        "name_pfx8": "acme",
        "first_word": "acme",
        "norm_addr": "123 main street springfield",
        "addr_numeric": ["123"],
    }
    t_rep = {
        "country": "US",
        "norm_name": "acme incorporated",
        "stripped_name": "acme",
        "first_two_tokens": "acme",
        "name_pfx8": "acme",
        "first_word": "acme",
        "norm_addr": "123 main st springfield",
        "addr_numeric": ["123"],
    }
    feats = extract_pair_features(s1_rep, t_rep, blocking_provenance={"stripped": True}, is_s3=False)
    assert isinstance(feats, np.ndarray)
    assert len(feats) == len(FEATURE_NAMES)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

def test_extract_pair_features_exact_match():
    rep = {
        "country": "IN",
        "norm_name": "reliance retail",
        "stripped_name": "reliance retail",
        "first_two_tokens": "reliance_retail",
        "name_pfx8": "reliance",
        "first_word": "reliance",
        "norm_addr": "mumbai maharashtra 400001",
        "addr_numeric": ["400001"],
    }
    feats = extract_pair_features(rep, rep, blocking_provenance=None, is_s3=True)
    # Check exact match flags
    assert feats[FEATURE_NAMES.index("name_exact")] == 1.0
    assert feats[FEATURE_NAMES.index("addr_exact")] == 1.0
    assert feats[FEATURE_NAMES.index("is_source3")] == 1.0
    assert feats[FEATURE_NAMES.index("same_country")] == 1.0
