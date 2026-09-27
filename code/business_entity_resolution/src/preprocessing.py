"""
Preprocessing and text normalization module for Business Entity Resolution.
Handles:
- Transliteration of Indic scripts (Hindi, Tamil, Gujarati, Odia, Kannada, etc.) via unidecode
- Transliteration and normalization of French accented characters (é, è, ê, etc.)
- Standardization and removal of business legal suffixes across US, India, and France
- Address token cleaning, number normalization, and unit stripping
"""

import re
import unidecode

LEGAL_STOPWORDS = {
    # US / UK
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'llc', 'llp', 'lp', 'pc', 'pllc', 'plc', 'group', 'holdings',
    'enterprises', 'enterprise', 'international', 'the', 'and', 'services', 'service',
    'solutions', 'solution', 'consulting', 'associates', 'technologies', 'technology',
    'trust', 'foundation', 'center', 'centre', 'fka', 'dba', 'management',
    # India
    'private limited', 'pvt ltd', 'proprietorship',
    # France
    'sarl', 'sas', 'sa', 'sasu', 'eurl', 'sci', 'snc', 'gie', 'selarl', 'scop'
}

ADDR_STOPWORDS = {
    'street', 'road', 'avenue', 'drive', 'lane', 'unit', 'suite', 'floor', 'block',
    'near', 'beside', 'behind', 'opposite', 'opp', 'dept', 'plot', 'house', 'st', 'rd',
    'ave', 'blvd', 'dr', 'ln', 'ct', 'hno', 'flat', 'bldg', 'rue', 'boulevard', 'bd',
    'chemin', 'place', 'allee', 'impasse'
}

def clean_text(text: str) -> str:
    """Normalize text by transliterating Unicode to ASCII and removing punctuation."""
    if not isinstance(text, str) or not text:
        return ""
    # Transliterate accents and Indic scripts to phonetic Latin
    text = unidecode.unidecode(text).lower()
    # Replace non-alphanumeric with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse consecutive spaces
    return re.sub(r'\s+', ' ', text).strip()

def clean_name(name: str) -> str:
    """Clean business name."""
    return clean_text(name)

def clean_addr(addr: str) -> str:
    """Clean business address."""
    return clean_text(addr)

def get_name_tokens(cname: str):
    """Extract significant tokens from cleaned business name."""
    tokens = [t for t in cname.split() if len(t) >= 2 and t not in LEGAL_STOPWORDS]
    return tokens

def extract_address_numbers(caddr: str):
    """
    Extract significant numbers from address (house numbers, plot numbers, pin codes).
    Strips leading zeroes (e.g., '01109' -> '1109').
    """
    raw_nums = re.findall(r'\d{1,6}', caddr)
    clean_nums = set()
    for n in raw_nums:
        stripped = n.lstrip('0')
        if stripped and len(stripped) >= 2:
            clean_nums.add(stripped)
    return clean_nums

def get_address_words(caddr: str):
    """Extract significant address words, excluding common street suffixes."""
    return [w for w in caddr.split() if len(w) >= 4 and not any(c.isdigit() for c in w) 
            and w not in ADDR_STOPWORDS and w not in LEGAL_STOPWORDS]
