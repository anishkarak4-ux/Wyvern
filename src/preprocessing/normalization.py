"""
normalization.py - Reusable, Deterministic Normalization for Amazon ML Challenge 2026

Provides derived representations for business names and addresses:
1. Basic normalized name (NFKC Unicode normalization, lowercase, punctuation removal)
2. Legal-suffix-stripped name (removes corporate/legal suffixes observed in data)
3. Significant-token representation (ordered significant word tokens)
4. First 2 significant tokens (for 2-token blocking)
5. Name prefix representation (first 8 characters)
6. First significant word (first token >= 3 characters)
7. Basic normalized address
8. Address numeric tokens (house/plot/PIN/postal codes)
9. Address word tokens

All functions are pure, deterministic, and handle missing/null values safely
without modifying raw inputs or generating empty/universal keys.
"""

import re
import unicodedata

# Multi-word legal suffix phrases to strip first (order matters: longest first)
MULTI_WORD_LEGAL_SUFFIXES = [
    "private limited",
    "pvt limited",
    "private ltd",
    "pvt ltd",
    "holding company",
    "holdings company",
    "holdings co",
    "holding co",
    "public limited",
    "pub limited",
    "limited liability company",
    "limited liability partnership",
    "societe a responsabilite limitee",
    "societe par actions simplifiee",
]

# Single-word legal and corporate suffixes
LEGAL_SUFFIX_TOKENS = {
    # US / General English
    "inc", "incorporated", "incorporation",
    "corp", "corporation",
    "llc", "llp", "lp", "pc", "pllc",
    "co", "company", "companies", "cie",
    "ltd", "limited",
    "dba",
    # India
    "pvt", "private",
    # France
    "sarl", "sas", "sa", "sci", "eurl", "snc", "gie", "societe", "etablissement",
}

def clean_text(text):
    """
    Standardize text:
    - Unicode NFKC normalization
    - Lowercase
    - Replace non-alphanumeric characters with spaces (preserving Unicode scripts)
    - Collapse redundant whitespace
    """
    if text is None:
        return ""
    text = str(text)
    if not text.strip() or text.strip().lower() in ("null", "nan", "none"):
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    # Replace non-alphanumeric (except whitespace) with space
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    # Replace underscores with space
    text = text.replace('_', ' ')
    # Lowercase & collapse whitespace
    return ' '.join(text.lower().split())

def strip_legal_suffixes(cleaned_name):
    """
    Strip known legal and corporate suffixes from a cleaned name.
    Preserves original tokens if stripping would result in an empty string.
    """
    if not cleaned_name:
        return ""
    
    # 1. Check multi-word suffix phrases at end or start of string
    s = cleaned_name
    for phrase in MULTI_WORD_LEGAL_SUFFIXES:
        if s.endswith(" " + phrase):
            s = s[:-len(" " + phrase)].strip()
        elif s.startswith(phrase + " "):
            s = s[len(phrase + " "):].strip()
        elif s == phrase:
            # Entire name is just the legal phrase; do not erase
            pass

    # 2. Token-level stripping for remaining single-word legal tokens
    tokens = s.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIX_TOKENS]
    
    if filtered:
        return ' '.join(filtered)
    # Fallback to cleaned name if all tokens were legal words
    return cleaned_name

def get_significant_tokens(cleaned_name):
    """Return list of significant tokens from name (after removing legal suffixes)."""
    stripped = strip_legal_suffixes(cleaned_name)
    tokens = stripped.split()
    return tokens if tokens else cleaned_name.split()

def get_first_two_tokens(cleaned_name):
    """Return tuple of first 2 significant tokens, or single token if only 1 exists."""
    sig_tokens = get_significant_tokens(cleaned_name)
    if not sig_tokens:
        return ""
    if len(sig_tokens) >= 2:
        return f"{sig_tokens[0]}_{sig_tokens[1]}"
    return sig_tokens[0]

def get_name_prefix(cleaned_name, length=8):
    """
    Return first `length` characters of cleaned name (or stripped name).
    Returns empty string if string is too short (< 3 chars) to avoid universal keys.
    """
    stripped = strip_legal_suffixes(cleaned_name)
    target = stripped if len(stripped) >= 3 else cleaned_name
    if len(target) < 3:
        return ""
    return target[:length]

def get_first_significant_word(cleaned_name, min_len=3):
    """
    Return first token with length >= min_len from significant tokens.
    Returns empty string if no such token exists.
    """
    tokens = get_significant_tokens(cleaned_name)
    for t in tokens:
        if len(t) >= min_len:
            return t
    return ""

def normalize_address(address):
    """Return basic normalized address string, or None if address is missing."""
    if address is None:
        return None
    s = str(address).strip()
    if not s or s.lower() in ("null", "nan", "none"):
        return None
    cleaned = clean_text(s)
    return cleaned if cleaned else None

def extract_address_numeric_tokens(cleaned_address):
    """
    Extract all distinct numeric tokens from address.
    Filters out extreme outliers (e.g. single digit '0' or numbers > 10 digits).
    Returns sorted list of numeric strings, or empty list if none/missing.
    """
    if not cleaned_address:
        return []
    nums = re.findall(r'\b\d+\b', cleaned_address)
    # Filter to reasonable numbers (length between 1 and 8 digits)
    distinct_nums = sorted(list({n for n in nums if 1 <= len(n) <= 8}))
    return distinct_nums

def extract_address_word_tokens(cleaned_address):
    """Return set of word tokens in cleaned address."""
    if not cleaned_address:
        return set()
    return set(cleaned_address.split())

def get_record_representations(entity_id, business_name, business_address, country):
    """
    Compute all derived normalization representations for an entity.
    Returns a dictionary of derived features.
    """
    norm_name = clean_text(business_name)
    stripped_name = strip_legal_suffixes(norm_name)
    first_two_tokens = get_first_two_tokens(norm_name)
    name_pfx8 = get_name_prefix(norm_name, length=8)
    first_word = get_first_significant_word(norm_name, min_len=3)
    
    norm_addr = normalize_address(business_address)
    addr_numeric = extract_address_numeric_tokens(norm_addr)
    
    # First char of name for constrained address blocking
    first_char = norm_name[0] if norm_name else ""
    
    return {
        "entity_id": entity_id,
        "country": country.strip() if country else "",
        "norm_name": norm_name,
        "stripped_name": stripped_name,
        "first_two_tokens": first_two_tokens,
        "name_pfx8": name_pfx8,
        "first_word": first_word,
        "norm_addr": norm_addr,
        "addr_numeric": addr_numeric,
        "first_char": first_char,
    }
