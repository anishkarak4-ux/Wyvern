"""
Unit tests for utils.validate_submission
"""

import os
import tempfile
import pytest
from utils.validate_submission import validate_id_list_file, read_ids, MATCHING_HEADER

def test_validate_matching_file_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy S1 source file
        s1_file = os.path.join(tmpdir, "test_source1.tsv")
        with open(s1_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tname\taddress\tcountry\n")
            f.write("S1-1\tAcme\t123 Main\tUS\n")
            f.write("S1-2\tBeta\t456 Oak\tUS\n")

        required_s1 = read_ids(s1_file)

        # Create valid matching file
        match_file = os.path.join(tmpdir, "matching_results.tsv")
        with open(match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-100,S3-200\n")
            f.write("S1-2\t\n")

        valid_match_ids = {"S2-100", "S3-200"}
        errors = []
        mapping = validate_id_list_file(
            match_file,
            MATCHING_HEADER,
            "matched_entity_ids",
            required_s1,
            valid_match_ids,
            errors,
        )
        assert len(errors) == 0
        assert mapping is not None
        assert mapping["S1-1"] == {"S2-100", "S3-200"}
        assert mapping["S1-2"] == set()

def test_validate_matching_file_missing_s1():
    with tempfile.TemporaryDirectory() as tmpdir:
        required_s1 = {"S1-1", "S1-2", "S1-3"}
        match_file = os.path.join(tmpdir, "matching_results.tsv")
        with open(match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-100\n")
            f.write("S1-2\t\n")

        errors = []
        mapping = validate_id_list_file(
            match_file,
            MATCHING_HEADER,
            "matched_entity_ids",
            required_s1,
            None,
            errors,
        )
        assert any("required S1 entity(ies) missing" in e for e in errors)
