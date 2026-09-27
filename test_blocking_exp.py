import pandas as pd
import numpy as np
import re
import sys
from collections import defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

# Let's inspect 10,000 S1 records with their true matches
print("Loading train slice for blocking experiment...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=20000)
# We need corresponding S2 and S3 for the same country/slice
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=20000)
gt_map = {}
all_target_ids = set()
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    m = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ""
    if m:
        m_list = [x.strip() for x in m.split(',') if x.strip()]
        gt_map[s1_id] = set(m_list)
        all_target_ids.update(m_list)
    else:
        gt_map[s1_id] = set()

print(f"S1 count: {len(s1)}, Ground truth mappings: {len(gt_map)}, Total true match targets: {len(all_target_ids)}")

# Let's load S2 and S3 (e.g. 100k or where IDs match)
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=200000)
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=200000)

s2_in_slice = s2[s2['entity_id'].isin(all_target_ids)]
s3_in_slice = s3[s3['entity_id'].isin(all_target_ids)]
print(f"S2 targets found in 200k: {len(s2_in_slice)}, S3 targets found in 200k: {len(s3_in_slice)}")

# Filter s1 to those where true matches are actually present in our loaded s2+s3
loaded_targets = set(s2['entity_id']).union(set(s3['entity_id']))
eval_s1 = {}
total_eval_targets = 0
for s1_id, targets in gt_map.items():
    present_targets = targets.intersection(loaded_targets)
    if present_targets:
        eval_s1[s1_id] = present_targets
        total_eval_targets += len(present_targets)

print(f"Eval S1 entities with present targets: {len(eval_s1)}, Total targets to find: {total_eval_targets}")
