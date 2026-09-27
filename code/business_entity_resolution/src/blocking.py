"""
Candidate generation and blocking module for Business Entity Resolution.
Reduces the O(N * M) comparison space to high-precision, high-recall candidate sets
per Source 1 entity using multi-key inverted indexing and lightweight fuzzy pruning.
"""

import re
from collections import defaultdict
from rapidfuzz import fuzz
from preprocessing import (
    clean_name, clean_addr, get_name_tokens, 
    extract_address_numbers, get_address_words,
    LEGAL_STOPWORDS
)

def generate_blocking_keys(cname: str, caddr: str, country: str):
    """
    Generate diverse blocking keys to achieve 98%+ recall with small candidate pools.
    Handles:
    - Name tokens & prefixes
    - Name word-order transposition
    - d/b/a and f/k/a prefixes
    - Unspaced/hashtag names (#sjace)
    - House numbers + locality names
    - Locality word pairs
    """
    keys = []
    
    # Handle d/b/a, f/k/a, a/k/a, c/o in name
    name_parts = re.split(r'\b(?:d/?b/?a|f/?k/?a|a/?k/?a|c/?o)\b', cname)
    all_tokens = []
    for part in name_parts:
        all_tokens.extend(get_name_tokens(part))
        
    # 1. First significant token
    if all_tokens:
        first = all_tokens[0]
        keys.append((country, f"T1_{first}"))
        if len(first) >= 4:
            keys.append((country, f"P4_{first[:4]}"))
            
        # 2. Pair of first two tokens (sorted to handle word swaps)
        if len(all_tokens) >= 2:
            pair = "_".join(sorted([all_tokens[0], all_tokens[1]]))
            keys.append((country, f"T2_{pair}"))
            
        # 3. Compact name prefix (no spaces, catches #sjace, etc.)
        compact = re.sub(r'[^a-z0-9]', '', cname)
        if len(compact) >= 5:
            keys.append((country, f"CP_{compact[:6]}"))
            
        # 4. If d/b/a had second part, also index its first token
        if len(name_parts) > 1:
            part2_tokens = get_name_tokens(name_parts[1])
            if part2_tokens:
                keys.append((country, f"T1_{part2_tokens[0]}"))

    # 5. Robust Address Blocking:
    clean_nums = list(extract_address_numbers(caddr))
    addr_words = get_address_words(caddr)
    
    # Combine first clean number with top address words
    if clean_nums and addr_words:
        n = clean_nums[0]
        for w in addr_words[:3]:
            keys.append((country, f"NW_{n}_{w}"))
            
    # Distinct address word pair (handles addresses where one record has no house number)
    if len(addr_words) >= 2:
        pair = "_".join(sorted([addr_words[0], addr_words[1]]))
        keys.append((country, f"AW_{pair}"))

    return keys

class BlockingEngine:
    """Builds multi-key inverted index over candidate targets (S2 + S3)."""
    
    def __init__(self):
        self.index = defaultdict(list)
        self.target_names = {}
        self.target_addrs = {}
        
    def index_targets(self, targets_df):
        """Index a DataFrame of targets containing entity_id, business_name, business_address, country."""
        for _, row in targets_df.iterrows():
            eid = row['entity_id']
            cname = clean_name(str(row['business_name']))
            caddr = clean_addr(str(row['business_address']))
            country = str(row['country'])
            
            self.target_names[eid] = cname
            self.target_addrs[eid] = caddr
            
            keys = generate_blocking_keys(cname, caddr, country)
            for k in keys:
                self.index[k].append(eid)
                
    def get_candidates_for_entity(self, cname: str, caddr: str, country: str, top_k: int = 25):
        """
        Query inverted index for an S1 entity and return pruned top_k candidate IDs.
        """
        keys = generate_blocking_keys(cname, caddr, country)
        candidates = set()
        for k in keys:
            if k in self.index:
                posting = self.index[k]
                if len(posting) > 1500:
                    candidates.update(posting[:100])
                else:
                    candidates.update(posting)
                
        if not candidates:
            return []
            
        if len(candidates) <= top_k:
            return list(candidates)
            
        # Fast fuzzy ranking to take top_k
        ranked = []
        for cid in candidates:
            cn = self.target_names.get(cid, "")
            ca = self.target_addrs.get(cid, "")
            n_sim = fuzz.token_sort_ratio(cname, cn)
            a_sim = fuzz.token_sort_ratio(caddr, ca)
            ranked.append((max(n_sim, a_sim), cid))
            
        ranked.sort(reverse=True)
        return [cid for _, cid in ranked[:top_k]]
