import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Loading samples...")
# Load first 100k of each to find overlapping ground truth pairs
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=100000).set_index('entity_id')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=250000).set_index('entity_id')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=250000).set_index('entity_id')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=100000)
gt = gt.dropna()

print("Searching for matches...")
found = 0
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1.index:
        continue
    matches = str(row['matched_entity_ids']).split(',')
    for m in matches:
        m = m.strip()
        target = None
        source_name = None
        if m in s2.index:
            target = s2.loc[m]
            source_name = "S2"
        elif m in s3.index:
            target = s3.loc[m]
            source_name = "S3"
        if target is not None:
            s1_row = s1.loc[s1_id]
            print(f"=== MATCH: {s1_id} <-> {m} ({source_name}, Country: {s1_row['country']}) ===")
            print(f"  S1: [Name] {s1_row['business_name']} | [Addr] {s1_row['business_address']}")
            print(f"  {source_name}: [Name] {target['business_name']} | [Addr] {target['business_address']}")
            print("-" * 60)
            found += 1
            if found >= 20:
                break
    if found >= 20:
        break
