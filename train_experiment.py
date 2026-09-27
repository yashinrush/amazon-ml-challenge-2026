import pandas as pd
import numpy as np
import re
import sys
import time
import unidecode
from collections import defaultdict
from rapidfuzz import fuzz, distance
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding='utf-8')

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'llc', 'llp', 'lp', 'pc', 'pllc', 'sarl', 'sas', 'sa', 'sasu',
    'eurl', 'sci', 'plc', 'services', 'service', 'solutions', 'solution', 'group',
    'holdings', 'enterprises', 'enterprise', 'international', 'the', 'and', 'fka',
    'trust', 'foundation', 'center', 'centre', 'associates', 'consulting'
}

ADDR_STOPWORDS = {
    'street', 'road', 'avenue', 'drive', 'lane', 'unit', 'suite', 'floor', 'block',
    'near', 'beside', 'behind', 'opposite', 'dept', 'plot', 'house', 'st', 'rd', 'ave'
}

def clean_name(name):
    if not isinstance(name, str) or not name:
        return ""
    name = unidecode.unidecode(name).lower()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    return re.sub(r'\s+', ' ', name).strip()

def clean_addr(addr):
    if not isinstance(addr, str) or not addr:
        return ""
    addr = unidecode.unidecode(addr).lower()
    addr = re.sub(r'[^a-z0-9\s]', ' ', addr)
    return re.sub(r'\s+', ' ', addr).strip()

def extract_features(cname1, caddr1, cname2, caddr2, is_s2):
    # Name features
    n_ts = fuzz.token_sort_ratio(cname1, cname2) / 100.0
    n_tset = fuzz.token_set_ratio(cname1, cname2) / 100.0
    n_part = fuzz.partial_ratio(cname1, cname2) / 100.0
    n_jw = distance.JaroWinkler.similarity(cname1, cname2)
    n_len_diff = abs(len(cname1) - len(cname2))
    
    # Address features
    a_ts = fuzz.token_sort_ratio(caddr1, caddr2) / 100.0
    a_tset = fuzz.token_set_ratio(caddr1, caddr2) / 100.0
    a_jw = distance.JaroWinkler.similarity(caddr1, caddr2)
    a_len_diff = abs(len(caddr1) - len(caddr2))
    
    # Numbers in address
    nums1 = set([n.lstrip('0') for n in re.findall(r'\d{1,6}', caddr1) if n.lstrip('0') and len(n.lstrip('0')) >= 2])
    nums2 = set([n.lstrip('0') for n in re.findall(r'\d{1,6}', caddr2) if n.lstrip('0') and len(n.lstrip('0')) >= 2])
    num_match = 1.0 if (nums1 and nums2 and bool(nums1.intersection(nums2))) else 0.0
    num_mismatch = 1.0 if (nums1 and nums2 and not bool(nums1.intersection(nums2))) else 0.0
    
    # Combined name + address
    comb1 = f"{cname1} {caddr1}"
    comb2 = f"{cname2} {caddr2}"
    comb_ts = fuzz.token_sort_ratio(comb1, comb2) / 100.0
    
    # Landmark flag
    has_landmark = 1.0 if any(w in caddr2.split() for w in ['near', 'beside', 'behind', 'opp', 'opposite']) else 0.0
    
    return [
        n_ts, n_tset, n_part, n_jw, n_len_diff,
        a_ts, a_tset, a_jw, a_len_diff,
        num_match, num_mismatch,
        comb_ts, has_landmark, float(is_s2)
    ]

FEATURE_NAMES = [
    'name_token_sort', 'name_token_set', 'name_partial', 'name_jaro_winkler', 'name_len_diff',
    'addr_token_sort', 'addr_token_set', 'addr_jaro_winkler', 'addr_len_diff',
    'num_match', 'num_mismatch',
    'comb_token_sort', 'has_landmark', 'is_s2'
]

def compute_f05_macro(predictions_dict, gt_dict):
    """
    Computes macro F_0.5 score per S1 entity according to the official formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons: 1.0 if correctly empty, 0.0 if any false match predicted.
    """
    scores = []
    for s1_id, true_set in gt_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        if not true_set:  # Singleton
            if not pred_set:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            if not pred_set:
                scores.append(0.0)
            else:
                tp = len(true_set.intersection(pred_set))
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if prec + rec == 0:
                    scores.append(0.0)
                else:
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                    scores.append(f05)
    return np.mean(scores)

from blocking_experiment import generate_blocking_keys

print("Loading data for LightGBM experiment...")
# Load S1 sample (including both entities with matches and singletons!)
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=2500).set_index('entity_id')
gt_raw = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
gt_raw = gt_raw[gt_raw['source1_entity_id'].isin(s1_df.index)]

gt_dict = {}
all_matched_ids = set()
for _, row in gt_raw.iterrows():
    s1_id = row['source1_entity_id']
    m = [x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip() and x.strip() != 'nan']
    gt_dict[s1_id] = set(m)
    all_matched_ids.update(m)

# Find all matching S2/S3 targets
print(f"Finding targets for {len(gt_dict)} S1 entities ({len(all_matched_ids)} target IDs)...")
s2_chunks = []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=150000):
    match = chunk[chunk['entity_id'].isin(all_matched_ids)]
    if len(match):
        s2_chunks.append(match)
    if sum(len(c) for c in s2_chunks) >= len(all_matched_ids):
        break
s2_found = pd.concat(s2_chunks) if s2_chunks else pd.DataFrame()

s3_chunks = []
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=150000):
    match = chunk[chunk['entity_id'].isin(all_matched_ids)]
    if len(match):
        s3_chunks.append(match)
    if sum(len(c) for c in s3_chunks) >= len(all_matched_ids):
        break
s3_found = pd.concat(s3_chunks) if s3_chunks else pd.DataFrame()

# Also sample background negatives
bg_s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=30000)
bg_s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=30000)

pool_df = pd.concat([s2_found, s3_found, bg_s2, bg_s3]).drop_duplicates('entity_id')
print(f"Pool created: {len(pool_df)} records.")

# Index pool
index = defaultdict(list)
pool_names = dict(zip(pool_df['entity_id'], pool_df['business_name'].apply(clean_name)))
pool_addrs = dict(zip(pool_df['entity_id'], pool_df['business_address'].apply(clean_addr)))
pool_countries = dict(zip(pool_df['entity_id'], pool_df['country'].astype(str)))

for eid in pool_df['entity_id']:
    cn = pool_names.get(eid, "")
    ca = pool_addrs.get(eid, "")
    country = pool_countries.get(eid, "")
    keys = generate_blocking_keys(cn, ca, country)
    for k in keys:
        index[k].append(eid)

print("Extracting candidate pairs and features...")
X_list = []
y_list = []
s1_groups = []
pair_meta = []  # (s1_id, cand_id)

for s1_id, true_targets in gt_dict.items():
    s1_row = s1_df.loc[s1_id]
    cname = clean_name(str(s1_row['business_name']))
    caddr = clean_addr(str(s1_row['business_address']))
    country = str(s1_row['country'])
    
    keys = generate_blocking_keys(cname, caddr, country)
    candidates = set()
    for k in keys:
        if k in index:
            candidates.update(index[k])
            
    # Rank candidates by fast heuristic and take top 25
    ranked = []
    for cid in candidates:
        cn = pool_names.get(cid, "")
        ca = pool_addrs.get(cid, "")
        score = max(fuzz.token_sort_ratio(cname, cn), fuzz.token_sort_ratio(caddr, ca))
        ranked.append((score, cid))
    ranked.sort(reverse=True)
    top_candidates = [cid for _, cid in ranked[:25]]
    
    # Also ensure true positives that are in pool are included in training
    present_targets = true_targets.intersection(set(pool_df['entity_id']))
    for pt in present_targets:
        if pt not in top_candidates:
            top_candidates.append(pt)
            
    for cid in top_candidates:
        cn = pool_names.get(cid, "")
        ca = pool_addrs.get(cid, "")
        is_s2 = 1.0 if cid.startswith('S2-') else 0.0
        label = 1 if cid in true_targets else 0
        feats = extract_features(cname, caddr, cn, ca, is_s2)
        X_list.append(feats)
        y_list.append(label)
        s1_groups.append(s1_id)
        pair_meta.append((s1_id, cid))

X = np.array(X_list)
y = np.array(y_list)
groups = np.array(s1_groups)

print(f"Dataset generated: {len(X)} pairs. Positive pairs: {sum(y)} ({np.mean(y)*100:.2f}%)")

# Train LightGBM with GroupKFold
print("Training LightGBM with GroupKFold...")
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(y))
models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    clf.fit(X_train, y_train)
    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    models.append(clf)

# Feature Importances
print("\n--- FEATURE IMPORTANCES ---")
importances = np.mean([m.feature_importances_ for m in models], axis=0)
for fn, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
    print(f"{fn:22s}: {imp:.1f}")

# Threshold Optimization for Macro F0.5
print("\n--- THRESHOLD OPTIMIZATION FOR MACRO F0.5 ---")
best_thresh = 0.5
best_f05 = 0.0

for thresh in np.arange(0.30, 0.95, 0.05):
    # Form predictions dict per S1
    preds_dict = defaultdict(set)
    for i, (prob, (s1_id, cid)) in enumerate(zip(oof_preds, pair_meta)):
        if prob >= thresh:
            preds_dict[s1_id].add(cid)
    # Include all S1s in gt_dict (even if 0 preds)
    full_preds = {s1_id: preds_dict.get(s1_id, set()) for s1_id in gt_dict.keys()}
    score = compute_f05_macro(full_preds, gt_dict)
    print(f"Threshold: {thresh:.2f} -> Macro F0.5: {score:.4f}")
    if score > best_f05:
        best_f05 = score
        best_thresh = thresh

print(f"\n>>> BEST THRESHOLD: {best_thresh:.2f} with Macro F0.5: {best_f05:.4f} <<<")
