import pandas as pd
import numpy as np
import re
import sys
import time
import unidecode
from collections import defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'llc', 'llp', 'lp', 'pc', 'pllc', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'sci', 'plc', 'services', 'service', 'solutions', 'solution', 'group',
    'holdings', 'enterprises', 'enterprise', 'international', 'the', 'and', 'fka',
    'trust', 'foundation', 'center', 'centre', 'associates', 'consulting'
}

def clean_name(name):
    if not isinstance(name, str) or not name:
        return ""
    name = unidecode.unidecode(name).lower()
    # Remove punctuation
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    return re.sub(r'\s+', ' ', name).strip()

def clean_addr(addr):
    if not isinstance(addr, str) or not addr:
        return ""
    addr = unidecode.unidecode(addr).lower()
    addr = re.sub(r'[^a-z0-9\s]', ' ', addr)
    return re.sub(r'\s+', ' ', addr).strip()

def get_name_tokens(cname):
    return [t for t in cname.split() if len(t) >= 2 and t not in LEGAL_STOPWORDS]

def generate_blocking_keys(cname, caddr, country):
    keys = []
    
    # Handle d/b/a, f/k/a, a/k/a, c/o in name
    name_parts = re.split(r'\b(?:d/?b/?a|f/?k/?a|a/?k/?a|c/?o)\b', cname)
    all_tokens = []
    for part in name_parts:
        all_tokens.extend(get_name_tokens(part))
    
    # 1. First significant token (allow length >= 2!)
    if all_tokens:
        first = all_tokens[0]
        keys.append((country, f"T1_{first}"))
        if len(first) >= 4:
            keys.append((country, f"P4_{first[:4]}"))
            
        # 2. Pair of first two tokens (sorted to handle word-order swaps)
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
    # Extract any digits sequence (even inside 31D or C/13St), strip leading zeroes
    raw_nums = re.findall(r'\d{1,6}', caddr)
    clean_nums = [n.lstrip('0') for n in raw_nums if n.lstrip('0') and len(n.lstrip('0')) >= 2]
    
    # Significant address words (skip common street stop words)
    ADDR_STOP = {'street', 'road', 'avenue', 'drive', 'lane', 'unit', 'suite', 'floor', 'block', 'near', 'beside', 'behind', 'opposite', 'dept', 'plot', 'house'}
    addr_words = [w for w in caddr.split() if len(w) >= 4 and not any(c.isdigit() for c in w) and w not in ADDR_STOP and w not in LEGAL_STOPWORDS]
    
    # Combine first clean number with top address words
    if clean_nums and addr_words:
        n = clean_nums[0]
        for w in addr_words[:3]:
            keys.append((country, f"NW_{n}_{w}"))
            
    # Distinct address word pair (handles addresses where one record has no house number, e.g. "uneeda street, madison")
    if len(addr_words) >= 2:
        pair = "_".join(sorted([addr_words[0], addr_words[1]]))
        keys.append((country, f"AW_{pair}"))

    return keys

print("Loading test slice...")
# Load S1 records first
print("Loading S1 records...")
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=5000).set_index('entity_id')

# Load ground truth matching these S1 entities
print("Loading ground truth...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=20000)
gt = gt[gt['source1_entity_id'].isin(s1_df.index)]
gt = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')]

gt_dict = {}
all_matched_ids = set()
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    m = [x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip()]
    if m:
        gt_dict[s1_id] = set(m)
        all_matched_ids.update(m)

print(f"Testing on {len(gt_dict)} S1 entities with {len(all_matched_ids)} total target IDs")

# We need the target S2/S3 records. Let's find them from the first 500k rows of S2 and S3
print("Finding target records in S2 and S3...")
s2_chunks = []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
    match = chunk[chunk['entity_id'].isin(all_matched_ids)]
    if len(match):
        s2_chunks.append(match)
    if sum(len(c) for c in s2_chunks) >= len(all_matched_ids):
        break
s2_found = pd.concat(s2_chunks) if s2_chunks else pd.DataFrame()

s3_chunks = []
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
    match = chunk[chunk['entity_id'].isin(all_matched_ids)]
    if len(match):
        s3_chunks.append(match)
    if sum(len(c) for c in s3_chunks) >= len(all_matched_ids):
        break
s3_found = pd.concat(s3_chunks) if s3_chunks else pd.DataFrame()

targets_df = pd.concat([s2_found, s3_found]).drop_duplicates('entity_id')
print(f"Found {len(targets_df)} of {len(all_matched_ids)} target records in train files.")

# Also add some random noise records to simulate realistic candidate pool
noise_s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=50000)
noise_s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=50000)
pool_df = pd.concat([targets_df, noise_s2, noise_s3]).drop_duplicates('entity_id')
print(f"Total pool size to block: {len(pool_df)} records.")

# Index pool by blocking keys
print("Indexing pool by blocking keys...")
t0 = time.time()
index = defaultdict(list)
for _, row in pool_df.iterrows():
    eid = row['entity_id']
    cname = clean_name(str(row['business_name']))
    caddr = clean_addr(str(row['business_address']))
    country = str(row['country'])
    keys = generate_blocking_keys(cname, caddr, country)
    for k in keys:
        index[k].append(eid)
print(f"Indexed in {time.time()-t0:.2f}s. Unique keys: {len(index)}")

# Precompute clean names and addresses into dict for instant O(1) lookup
print("Pre-cleaning pool strings...")
pool_names = dict(zip(pool_df['entity_id'], pool_df['business_name'].apply(clean_name)))
pool_addrs = dict(zip(pool_df['entity_id'], pool_df['business_address'].apply(clean_addr)))

# Query S1 records
print("Querying S1 entities and evaluating recall...")
total_targets_evaluated = 0
candidate_counts = []
recalled_at_k = defaultdict(int)

for s1_id, true_targets in gt_dict.items():
    # Only evaluate targets present in pool
    present_targets = true_targets.intersection(set(pool_df['entity_id']))
    if not present_targets:
        continue
    
    s1_row = s1_df.loc[s1_id]
    cname = clean_name(str(s1_row['business_name']))
    caddr = clean_addr(str(s1_row['business_address']))
    country = str(s1_row['country'])
    
    keys = generate_blocking_keys(cname, caddr, country)
    candidates = set()
    for k in keys:
        if k in index:
            candidates.update(index[k])
            
    # Pruning experiment: test recall at various top-K thresholds
    candidates_ranked = []
    for cid in candidates:
        cn = pool_names.get(cid, "")
        ca = pool_addrs.get(cid, "")
        n_sim = fuzz.token_sort_ratio(cname, cn)
        a_sim = fuzz.token_sort_ratio(caddr, ca)
        score = max(n_sim, a_sim)
        candidates_ranked.append((score, cid))
        
    candidates_ranked.sort(reverse=True)
    
    for k in [10, 15, 20, 30, 50, 100]:
        top_k_ids = set([cid for _, cid in candidates_ranked[:k]])
        recalled_at_k[k] += len(present_targets.intersection(top_k_ids))
        
    total_targets_evaluated += len(present_targets)
    candidate_counts.append(len(candidates))

print(f"\n--- PRUNING ANALYSIS (Total targets: {total_targets_evaluated}) ---")
for k in [10, 15, 20, 30, 50, 100]:
    rec_k = recalled_at_k[k] / total_targets_evaluated * 100
    print(f"Top-{k:2d} candidates per S1 -> Recall: {rec_k:.2f}% ({recalled_at_k[k]}/{total_targets_evaluated})")

print("\n--- MISSED TARGET EXAMPLES ---")
missed_count = 0
for s1_id, true_targets in gt_dict.items():
    present_targets = true_targets.intersection(set(pool_df['entity_id']))
    if not present_targets:
        continue
    s1_row = s1_df.loc[s1_id]
    cname = clean_name(str(s1_row['business_name']))
    caddr = clean_addr(str(s1_row['business_address']))
    country = str(s1_row['country'])
    keys = generate_blocking_keys(cname, caddr, country)
    candidates = set()
    for k in keys:
        if k in index:
            candidates.update(index[k])
    missed = present_targets - candidates
    for m in missed:
        t_row = pool_df[pool_df['entity_id'] == m].iloc[0]
        print(f"S1 ({s1_id}): Name='{s1_row['business_name']}', Addr='{s1_row['business_address']}'")
        print(f"MISSED ({m}): Name='{t_row['business_name']}', Addr='{t_row['business_address']}'")
        print()
        missed_count += 1
        if missed_count >= 10:
            break
    if missed_count >= 10:
        break
