"""
pair_features.py - Reusable Pairwise Feature Engineering for Amazon ML Challenge 2026

Extracts comprehensive lexical, phonetic, token, numeric, cross-field, and metadata
features for entity pairs (Source 1 reference entity vs Target candidate entity).

All features are pure, deterministic, bounded, and handle missing values gracefully.
Strictly zero external data, web lookups, or third-party APIs used.
"""

import re
import numpy as np
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    # --- Name Lexical & Token Features (1-15) ---
    "name_exact",
    "name_stripped_exact",
    "name_len_diff",
    "name_len_ratio",
    "name_stripped_len_diff",
    "name_jaro_winkler",
    "name_levenshtein_norm",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_first_two_tokens_match",
    "name_prefix_8_match",
    "name_first_word_match",
    "name_num_shared_tokens",
    "name_token_jaccard",

    # --- Address Lexical, Token & Numeric Features (16-28) ---
    "addr_exact",
    "addr_len_diff",
    "addr_jaro_winkler",
    "addr_levenshtein_norm",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_num_shared_tokens",
    "addr_token_jaccard",
    "shared_numeric_count",
    "shared_numeric_ratio",
    "exact_numeric_match",
    "has_postal_code_pair",
    "postal_code_match",

    # --- Cross-Field Interaction Features (29-34) ---
    "name_addr_sim_product",
    "name_exact_and_addr_sim",
    "name_sim_and_shared_num",
    "both_fields_agree_strong",
    "name_only_agrees",
    "addr_only_agrees",

    # --- Metadata & Blocking Provenance Features (35-45) ---
    "same_country",
    "is_source3",
    "s1_missing_name",
    "t_missing_name",
    "s1_missing_addr",
    "t_missing_addr",
    "both_have_addr",
    "blocking_rule_stripped",
    "blocking_rule_tok2",
    "blocking_rule_pfx8",
    "blocking_rule_addr_num",
    "num_blocking_rules",
]

US_ZIP_REGEX = re.compile(r'\b\d{5}\b')
IN_PIN_REGEX = re.compile(r'\b\d{6}\b')

def extract_postal_code(cleaned_address, country):
    """
    Extract postal code from cleaned address based on country:
    - US: 5-digit ZIP code
    - India (in): 6-digit PIN code
    """
    if not cleaned_address:
        return ""
    c_lower = country.lower().strip() if country else ""
    if c_lower == "us":
        m = US_ZIP_REGEX.findall(cleaned_address)
        return m[-1] if m else ""
    elif c_lower == "in":
        m = IN_PIN_REGEX.findall(cleaned_address)
        return m[-1] if m else ""
    return ""

def extract_pair_features(s1_rep, t_rep, blocking_provenance=None, is_s3=False):
    """
    Extract 45 pairwise features for an entity pair (S1 vs Target).
    
    Parameters:
    - s1_rep: dict or RecordRepresentation of Source 1 entity
    - t_rep: dict or RecordRepresentation of Target entity
    - blocking_provenance: dict or tuple indicating which blocking rules generated this candidate:
        {'stripped': bool, 'tok2': bool, 'pfx8': bool, 'addr_num': bool}
    - is_s3: bool, True if target entity is from Source 3, False if Source 2
    
    Returns:
    - numpy array of shape (45,), dtype float32
    """
    feats = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # Unpack S1 representations
    s1_country = s1_rep.get("country", "")
    s1_name = s1_rep.get("norm_name", "")
    s1_stripped = s1_rep.get("stripped_name", "")
    s1_tok2 = s1_rep.get("first_two_tokens", "")
    s1_pfx8 = s1_rep.get("name_pfx8", "")
    s1_fw = s1_rep.get("first_word", "")
    s1_addr = s1_rep.get("norm_addr") or ""
    s1_nums = s1_rep.get("addr_numeric", [])
    s1_postal = s1_rep.get("postal_code")
    if s1_postal is None:
        s1_postal = extract_postal_code(s1_addr, s1_country)

    # Unpack Target representations
    t_country = t_rep.get("country", "")
    t_name = t_rep.get("norm_name", "")
    t_stripped = t_rep.get("stripped_name", "")
    t_tok2 = t_rep.get("first_two_tokens", "")
    t_pfx8 = t_rep.get("name_pfx8", "")
    t_fw = t_rep.get("first_word", "")
    t_addr = t_rep.get("norm_addr") or ""
    t_nums = t_rep.get("addr_numeric", [])
    t_postal = t_rep.get("postal_code")
    if t_postal is None:
        t_postal = extract_postal_code(t_addr, t_country)

    # Name lengths & missingness
    len_s1_name = len(s1_name)
    len_t_name = len(t_name)
    s1_has_name = bool(len_s1_name > 0)
    t_has_name = bool(len_t_name > 0)

    # Address lengths & missingness
    len_s1_addr = len(s1_addr)
    len_t_addr = len(t_addr)
    s1_has_addr = bool(len_s1_addr > 0)
    t_has_addr = bool(len_t_addr > 0)

    # -------------------------------------------------------------
    # 1. NAME FEATURES (Indices 0 to 14)
    # -------------------------------------------------------------
    # name_exact
    feats[0] = 1.0 if (s1_has_name and s1_name == t_name) else 0.0
    # name_stripped_exact
    feats[1] = 1.0 if (s1_stripped and s1_stripped == t_stripped) else 0.0
    # name_len_diff
    feats[2] = float(abs(len_s1_name - len_t_name))
    # name_len_ratio
    max_len = max(len_s1_name, len_t_name)
    feats[3] = (min(len_s1_name, len_t_name) / float(max_len)) if max_len > 0 else 0.0
    # name_stripped_len_diff
    feats[4] = float(abs(len(s1_stripped) - len(t_stripped)))

    if s1_has_name and t_has_name:
        # name_jaro_winkler
        name_jw = distance.JaroWinkler.similarity(s1_name, t_name)
        feats[5] = name_jw
        # name_levenshtein_norm
        feats[6] = distance.Levenshtein.normalized_similarity(s1_name, t_name)
        # name_token_sort_ratio
        feats[7] = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
        # name_token_set_ratio
        feats[8] = fuzz.token_set_ratio(s1_name, t_name) / 100.0
        # name_partial_ratio
        feats[9] = fuzz.partial_ratio(s1_name, t_name) / 100.0
    else:
        name_jw = 0.0
        feats[5] = 0.0
        feats[6] = 0.0
        feats[7] = 0.0
        feats[8] = 0.0
        feats[9] = 0.0

    # Key matches
    feats[10] = 1.0 if (s1_tok2 and s1_tok2 == t_tok2) else 0.0
    feats[11] = 1.0 if (s1_pfx8 and s1_pfx8 == t_pfx8) else 0.0
    feats[12] = 1.0 if (s1_fw and s1_fw == t_fw) else 0.0

    # Name token overlap
    s1_name_tokens = s1_rep.get("name_tokens")
    if s1_name_tokens is None:
        s1_name_tokens = set(s1_name.split())
    t_name_tokens = t_rep.get("name_tokens")
    if t_name_tokens is None:
        t_name_tokens = set(t_name.split())

    shared_name_tokens = s1_name_tokens & t_name_tokens
    union_name_tokens = s1_name_tokens | t_name_tokens
    feats[13] = float(len(shared_name_tokens))
    feats[14] = (len(shared_name_tokens) / float(len(union_name_tokens))) if union_name_tokens else 0.0

    # -------------------------------------------------------------
    # 2. ADDRESS FEATURES (Indices 15 to 27)
    # -------------------------------------------------------------
    feats[15] = 1.0 if (s1_has_addr and t_has_addr and s1_addr == t_addr) else 0.0
    feats[16] = float(abs(len_s1_addr - len_t_addr)) if (s1_has_addr and t_has_addr) else -1.0

    if s1_has_addr and t_has_addr:
        addr_jw = distance.JaroWinkler.similarity(s1_addr, t_addr)
        feats[17] = addr_jw
        feats[18] = distance.Levenshtein.normalized_similarity(s1_addr, t_addr)
        feats[19] = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
        feats[20] = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

        s1_addr_tokens = s1_rep.get("addr_tokens")
        if s1_addr_tokens is None:
            s1_addr_tokens = set(s1_addr.split())
        t_addr_tokens = t_rep.get("addr_tokens")
        if t_addr_tokens is None:
            t_addr_tokens = set(t_addr.split())

        shared_addr_tokens = s1_addr_tokens & t_addr_tokens
        union_addr_tokens = s1_addr_tokens | t_addr_tokens
        feats[21] = float(len(shared_addr_tokens))
        feats[22] = (len(shared_addr_tokens) / float(len(union_addr_tokens))) if union_addr_tokens else 0.0
    else:
        addr_jw = 0.0
        feats[17] = 0.0
        feats[18] = 0.0
        feats[19] = 0.0
        feats[20] = 0.0
        feats[21] = 0.0
        feats[22] = 0.0

    # Numeric address tokens
    s1_num_set = set(s1_nums)
    t_num_set = set(t_nums)
    shared_nums = s1_num_set & t_num_set
    union_nums = s1_num_set | t_num_set
    feats[23] = float(len(shared_nums))
    feats[24] = (len(shared_nums) / float(len(union_nums))) if union_nums else 0.0
    feats[25] = 1.0 if (s1_num_set and s1_num_set == t_num_set) else 0.0

    # Postal code agreement
    if s1_postal and t_postal:
        feats[26] = 1.0  # has_postal_code_pair
        feats[27] = 1.0 if s1_postal == t_postal else 0.0
    else:
        feats[26] = 0.0
        feats[27] = -1.0  # missing

    # -------------------------------------------------------------
    # 3. CROSS-FIELD INTERACTION FEATURES (Indices 28 to 33)
    # -------------------------------------------------------------
    feats[28] = name_jw * addr_jw
    feats[29] = feats[0] * addr_jw
    feats[30] = name_jw * (1.0 if feats[23] >= 1.0 else 0.0)
    feats[31] = 1.0 if (name_jw >= 0.85 and addr_jw >= 0.70) else 0.0
    feats[32] = 1.0 if (name_jw >= 0.85 and addr_jw < 0.35) else 0.0
    feats[33] = 1.0 if (name_jw < 0.40 and addr_jw >= 0.75) else 0.0

    # -------------------------------------------------------------
    # 4. METADATA & BLOCKING PROVENANCE (Indices 34 to 44)
    # -------------------------------------------------------------
    feats[34] = 1.0 if (s1_country and s1_country == t_country) else 0.0
    feats[35] = 1.0 if is_s3 else 0.0
    feats[36] = 1.0 if not s1_has_name else 0.0
    feats[37] = 1.0 if not t_has_name else 0.0
    feats[38] = 1.0 if not s1_has_addr else 0.0
    feats[39] = 1.0 if not t_has_addr else 0.0
    feats[40] = 1.0 if (s1_has_addr and t_has_addr) else 0.0

    if blocking_provenance:
        rule_stripped = 1.0 if blocking_provenance.get("stripped", False) else 0.0
        rule_tok2 = 1.0 if blocking_provenance.get("tok2", False) else 0.0
        rule_pfx8 = 1.0 if blocking_provenance.get("pfx8", False) else 0.0
        rule_addr_num = 1.0 if blocking_provenance.get("addr_num", False) else 0.0
        num_rules = rule_stripped + rule_tok2 + rule_pfx8 + rule_addr_num
    else:
        # Infer provenance from representations
        rule_stripped = 1.0 if (s1_stripped and s1_stripped == t_stripped) else 0.0
        rule_tok2 = 1.0 if (s1_tok2 and s1_tok2 == t_tok2) else 0.0
        rule_pfx8 = 1.0 if (s1_pfx8 and s1_pfx8 == t_pfx8) else 0.0
        rule_addr_num = 1.0 if (shared_nums and s1_rep.get("first_char") == t_rep.get("first_char")) else 0.0
        num_rules = rule_stripped + rule_tok2 + rule_pfx8 + rule_addr_num

    feats[41] = rule_stripped
    feats[42] = rule_tok2
    feats[43] = rule_pfx8
    feats[44] = rule_addr_num
    feats[45] = num_rules

    return feats
