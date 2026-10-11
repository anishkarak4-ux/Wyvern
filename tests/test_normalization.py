"""
Unit tests for src.preprocessing.normalization
"""

import pytest
from src.preprocessing.normalization import (
    clean_text,
    strip_legal_suffixes,
    get_significant_tokens,
    get_first_two_tokens,
    get_name_prefix,
    get_first_significant_word,
    normalize_address,
    extract_address_numeric_tokens,
    extract_address_word_tokens,
)

def test_clean_text_basic():
    assert clean_text("Acme Corp, Inc.") == "acme corp inc"
    assert clean_text("  SPACED   OUT  ") == "spaced out"
    assert clean_text(None) == ""
    assert clean_text("null") == ""
    assert clean_text("NAN") == ""

def test_clean_text_unicode_nfkc():
    # Preserves accents under NFKC without crashing
    text = "Café & École SARL"
    cleaned = clean_text(text)
    assert "café" in cleaned
    assert "école" in cleaned
    assert "&" not in cleaned

def test_strip_legal_suffixes():
    assert strip_legal_suffixes("acme technologies pvt ltd") == "acme technologies"
    assert strip_legal_suffixes("google llc") == "google"
    assert strip_legal_suffixes("bistro sarl") == "bistro"
    assert strip_legal_suffixes("holdings company limited") == "holdings company limited"  # preserved if only legal words
    assert strip_legal_suffixes("") == ""

def test_get_significant_tokens():
    tokens = get_significant_tokens("tata motors private limited")
    assert tokens == ["tata", "motors"]

def test_get_first_two_tokens():
    assert get_first_two_tokens("apple computer inc") == "apple_computer"
    assert get_first_two_tokens("starbucks") == "starbucks"
    assert get_first_two_tokens("") == ""

def test_get_name_prefix():
    assert get_name_prefix("microsoft corporation", length=8) == "microsof"
    assert get_name_prefix("ab", length=8) == ""  # Too short (<3 chars)

def test_get_first_significant_word():
    assert get_first_significant_word("a bb ccc dddd", min_len=3) == "ccc"
    assert get_first_significant_word("a b", min_len=3) == ""

def test_normalize_address():
    assert normalize_address("123 Main St., Suite 400") == "123 main st suite 400"
    assert normalize_address(None) is None
    assert normalize_address("NaN") is None

def test_extract_address_numeric_tokens():
    nums = extract_address_numeric_tokens("123 main street unit 400 postal 560001")
    assert nums == ["123", "400", "560001"]
    assert extract_address_numeric_tokens("") == []
