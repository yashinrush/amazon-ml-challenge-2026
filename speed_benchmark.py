import time
import os
import sys
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.path.append('code/business_entity_resolution/src')
from preprocessing import clean_name, clean_addr
from blocking import BlockingEngine
from features import extract_pairwise_features
from model import EntityMatchingModel

print("Loading 1,000 test entities...")
t1_df = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', nrows=1000)
country = t1_df['country'].iloc[0]
print(f"Country: {country}")

# Load 50,000 targets for that country
t2_chunks = []
for chunk in pd.read_csv('dataset/test/test_source2.tsv', sep='\t', chunksize=100000):
    sub = chunk[chunk['country'] == country]
    if len(sub):
        t2_chunks.append(sub)
    if sum(len(c) for c in t2_chunks) >= 50000:
        break
targets = pd.concat(t2_chunks).head(50000)

print(f"Indexing {len(targets)} targets...")
t0 = time.time()
engine = BlockingEngine()
engine.index_targets(targets)
print(f"Indexed in {time.time()-t0:.2f}s")

# Load model
model = EntityMatchingModel()
model.load('code/business_entity_resolution/models/model.pkl')

print("Benchmarking 1,000 S1 queries + feature extraction + inference...")
t0 = time.time()
pairs_count = 0
for _, row in t1_df.iterrows():
    cname1 = clean_name(str(row['business_name']))
    caddr1 = clean_addr(str(row['business_address']))
    cands = engine.get_candidates_for_entity(cname1, caddr1, country, top_k=20)
    pairs_count += len(cands)
    for cid in cands:
        cn2 = engine.target_names.get(cid, "")
        ca2 = engine.target_addrs.get(cid, "")
        feats = extract_pairwise_features(cname1, caddr1, cn2, ca2, cid.startswith('S2-'))

dt = time.time() - t0
print(f"Completed 1,000 entities in {dt:.2f}s ({len(t1_df)/dt:.1f} entities/sec). Total candidate pairs: {pairs_count}")
