import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load all files
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

ts1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t')
ts2 = pd.read_csv('dataset/test/test_source2.tsv', sep='\t')
ts3 = pd.read_csv('dataset/test/test_source3.tsv', sep='\t')

print('=== TRAINING SET SIZES ===')
print(f'Source 1: {len(s1):,} rows')
print(f'Source 2: {len(s2):,} rows')
print(f'Source 3: {len(s3):,} rows')
print(f'Ground Truth: {len(gt):,} rows')
print()

print('=== TEST SET SIZES ===')
print(f'Test Source 1: {len(ts1):,} rows')
print(f'Test Source 2: {len(ts2):,} rows')
print(f'Test Source 3: {len(ts3):,} rows')
print()

print('=== COUNTRY DISTRIBUTION (TRAIN) ===')
print('Source 1:', s1['country'].value_counts().to_dict())
print('Source 2:', s2['country'].value_counts().to_dict())
print('Source 3:', s3['country'].value_counts().to_dict())
print()

print('=== COUNTRY DISTRIBUTION (TEST) ===')
print('Test Source 1:', ts1['country'].value_counts().to_dict())
print('Test Source 2:', ts2['country'].value_counts().to_dict())
print('Test Source 3:', ts3['country'].value_counts().to_dict())
print()

# Ground truth stats
print('=== GROUND TRUTH ANALYSIS ===')
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
gt['n_matches'] = gt['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
singletons = (gt['n_matches'] == 0).sum()
print(f'Singletons (0 matches): {singletons:,} ({singletons/len(gt)*100:.1f}%)')
print(f'With matches: {len(gt)-singletons:,} ({(len(gt)-singletons)/len(gt)*100:.1f}%)')
print('Match count distribution:')
print(gt['n_matches'].value_counts().sort_index().to_string())
print()
print(f'Mean matches per entity: {gt["n_matches"].mean():.2f}')
print(f'Max matches per entity: {gt["n_matches"].max()}')

# Check for S2-only, S3-only, both
def categorize_matches(ids_str):
    if not ids_str:
        return 'singleton'
    ids = ids_str.split(',')
    has_s2 = any(i.startswith('S2-') for i in ids)
    has_s3 = any(i.startswith('S3-') for i in ids)
    if has_s2 and has_s3:
        return 'both'
    elif has_s2:
        return 's2_only'
    else:
        return 's3_only'

gt['match_type'] = gt['matched_entity_ids'].apply(categorize_matches)
print()
print('=== MATCH TYPE DISTRIBUTION ===')
print(gt['match_type'].value_counts().to_string())

# Nulls
print()
print('=== NULL VALUES ===')
for name, df in [('Train S1', s1), ('Train S2', s2), ('Train S3', s3),
                  ('Test S1', ts1), ('Test S2', ts2), ('Test S3', ts3)]:
    nn = df['business_name'].isna().sum()
    na = df['business_address'].isna().sum()
    nc = df['country'].isna().sum()
    print(f'{name}: name_null={nn} ({nn/len(df)*100:.2f}%), addr_null={na} ({na/len(df)*100:.2f}%), country_null={nc}')

# Address length stats
print()
print('=== ADDRESS LENGTH STATS ===')
for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    lengths = df['business_address'].dropna().str.len()
    print(f'{name}: mean={lengths.mean():.0f}, median={lengths.median():.0f}, max={lengths.max():.0f}, min={lengths.min():.0f}')

# Name length stats
print()
print('=== NAME LENGTH STATS ===')
for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    lengths = df['business_name'].dropna().str.len()
    print(f'{name}: mean={lengths.mean():.0f}, median={lengths.median():.0f}, max={lengths.max():.0f}, min={lengths.min():.0f}')

# Country-wise ground truth
print()
print('=== COUNTRY-WISE GROUND TRUTH ===')
gt_merged = gt.merge(s1[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
for country in gt_merged['country'].unique():
    sub = gt_merged[gt_merged['country'] == country]
    sing = (sub['n_matches'] == 0).sum()
    print(f'{country}: total={len(sub):,}, singletons={sing:,} ({sing/len(sub)*100:.1f}%), avg_matches={sub["n_matches"].mean():.2f}')
