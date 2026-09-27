"""
Pairwise feature extraction module for candidate pair classification.
Computes fine-grained lexical, phonetic, structural, and numerical similarity metrics.
"""

import re
from rapidfuzz import fuzz, distance
from preprocessing import extract_address_numbers

FEATURE_NAMES = [
    'name_token_sort', 'name_token_set', 'name_partial', 'name_jaro_winkler', 'name_len_diff',
    'addr_token_sort', 'addr_token_set', 'addr_jaro_winkler', 'addr_len_diff',
    'num_match', 'num_mismatch',
    'comb_token_sort', 'has_landmark', 'is_s2'
]

def extract_pairwise_features(cname1: str, caddr1: str, cname2: str, caddr2: str, is_s2: bool):
    """
    Extract pairwise feature vector for a candidate pair (S1 vs S2/S3).
    All string inputs must be pre-cleaned.
    """
    # Name similarity metrics
    n_ts = fuzz.token_sort_ratio(cname1, cname2) / 100.0
    n_tset = fuzz.token_set_ratio(cname1, cname2) / 100.0
    n_part = fuzz.partial_ratio(cname1, cname2) / 100.0
    n_jw = distance.JaroWinkler.similarity(cname1, cname2)
    n_len_diff = abs(len(cname1) - len(cname2))
    
    # Address similarity metrics
    a_ts = fuzz.token_sort_ratio(caddr1, caddr2) / 100.0
    a_tset = fuzz.token_set_ratio(caddr1, caddr2) / 100.0
    a_jw = distance.JaroWinkler.similarity(caddr1, caddr2)
    a_len_diff = abs(len(caddr1) - len(caddr2))
    
    # House / plot / unit number consistency
    nums1 = extract_address_numbers(caddr1)
    nums2 = extract_address_numbers(caddr2)
    num_match = 1.0 if (nums1 and nums2 and bool(nums1.intersection(nums2))) else 0.0
    num_mismatch = 1.0 if (nums1 and nums2 and not bool(nums1.intersection(nums2))) else 0.0
    
    # Concatenated name + address similarity
    comb1 = f"{cname1} {caddr1}"
    comb2 = f"{cname2} {caddr2}"
    comb_ts = fuzz.token_sort_ratio(comb1, comb2) / 100.0
    
    # Landmark presence flag (especially relevant for Source 3 addresses)
    has_landmark = 1.0 if any(w in caddr2.split() for w in ['near', 'beside', 'behind', 'opp', 'opposite']) else 0.0
    
    return [
        n_ts, n_tset, n_part, n_jw, n_len_diff,
        a_ts, a_tset, a_jw, a_len_diff,
        num_match, num_mismatch,
        comb_ts, has_landmark, 1.0 if is_s2 else 0.0
    ]
