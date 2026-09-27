"""
Main pipeline execution script for Business Entity Resolution.
Supports:
1. --mode train: Generates training pairs, trains LightGBM, optimizes F_0.5 threshold, and saves model.
2. --mode predict: Runs country-partitioned candidate generation, feature extraction, and prediction on test set.
   Generates output/candidate_pairs.tsv and output/matching_results.tsv.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from rapidfuzz import fuzz

from preprocessing import clean_name, clean_addr
from blocking import BlockingEngine, generate_blocking_keys
from features import extract_pairwise_features
from model import EntityMatchingModel

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

def train_pipeline(data_dir: str, model_path: str, n_train_s1: int = 15000):
    """
    Train matching model on training dataset with mined hard negatives from blocking.
    """
    print(f"\n==========================================")
    print(f"       STARTING TRAINING PIPELINE         ")
    print(f"==========================================\n")
    
    # 1. Load S1 and Ground Truth
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    
    print(f"Loading {n_train_s1} S1 training entities...")
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=n_train_s1).set_index("entity_id")
    
    print("Loading ground truth matching labels...")
    gt_df = pd.read_csv(gt_path, sep="\t", nrows=n_train_s1 * 3)
    gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_df.index)]
    
    gt_dict = {}
    all_target_ids = set()
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        m = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if m and m != "nan":
            targets = [x.strip() for x in m.split(",") if x.strip()]
            gt_dict[s1_id] = set(targets)
            all_target_ids.update(targets)
        else:
            gt_dict[s1_id] = set()
            
    # Include any remaining S1 entities as singletons
    for s1_id in s1_df.index:
        if s1_id not in gt_dict:
            gt_dict[s1_id] = set()
            
    print(f"Total S1 entities in train split: {len(gt_dict)} ({sum(len(v)==0 for v in gt_dict.values())} singletons)")
    print(f"Searching for {len(all_target_ids)} matching target records across S2 and S3...")
    
    # Load targets from S2 and S3 chunks
    s2_targets = []
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=250000):
        match = chunk[chunk["entity_id"].isin(all_target_ids)]
        if len(match):
            s2_targets.append(match)
        if sum(len(c) for c in s2_targets) >= len(all_target_ids):
            break
            
    s3_targets = []
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=250000):
        match = chunk[chunk["entity_id"].isin(all_target_ids)]
        if len(match):
            s3_targets.append(match)
        if sum(len(c) for c in s3_targets) >= len(all_target_ids):
            break
            
    # Add random background negative records
    bg_s2 = pd.read_csv(s2_path, sep="\t", nrows=50000)
    bg_s3 = pd.read_csv(s3_path, sep="\t", nrows=50000)
    
    pool_df = pd.concat(s2_targets + s3_targets + [bg_s2, bg_s3]).drop_duplicates("entity_id")
    print(f"Target pool created: {len(pool_df)} records.")
    
    # Build Inverted Index
    print("Building blocking index on training pool...")
    engine = BlockingEngine()
    engine.index_targets(pool_df)
    
    # Generate Training Candidate Pairs & Features
    print("Generating candidate pairs and extracting pairwise features...")
    X_list, y_list, groups, pair_meta = [], [], [], []
    
    for s1_id, true_targets in tqdm(gt_dict.items(), desc="Processing S1 entities"):
        s1_row = s1_df.loc[s1_id]
        cname1 = clean_name(str(s1_row["business_name"]))
        caddr1 = clean_addr(str(s1_row["business_address"]))
        country = str(s1_row["country"])
        
        cands = engine.get_candidates_for_entity(cname1, caddr1, country, top_k=25)
        
        # Ensure true targets present in pool are included in training
        present_targets = true_targets.intersection(set(pool_df["entity_id"]))
        for pt in present_targets:
            if pt not in cands:
                cands.append(pt)
                
        for cid in cands:
            cname2 = engine.target_names.get(cid, "")
            caddr2 = engine.target_addrs.get(cid, "")
            is_s2 = cid.startswith("S2-")
            label = 1 if cid in true_targets else 0
            
            feats = extract_pairwise_features(cname1, caddr1, cname2, caddr2, is_s2)
            X_list.append(feats)
            y_list.append(label)
            groups.append(s1_id)
            pair_meta.append((s1_id, cid))
            
    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups)
    
    print(f"\nTraining set size: {len(X)} candidate pairs. Positives: {sum(y)} ({np.mean(y)*100:.2f}%)")
    
    # Train Model and Calibrate Threshold
    model = EntityMatchingModel()
    model.train(X, y, groups, gt_dict, pair_meta)
    
    # Save Model
    model.save(model_path)
    print(f"\nTraining complete. Model saved to: {model_path}\n")

def predict_pipeline(data_dir: str, output_dir: str, model_path: str):
    """
    Run candidate generation, feature extraction, and inference on the test set.
    Processes data country-by-country to ensure high speed and low memory footprint.
    """
    print(f"\n==========================================")
    print(f"       STARTING PREDICTION PIPELINE       ")
    print(f"==========================================\n")
    
    os.makedirs(output_dir, exist_ok=True)
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # Load Model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run --mode train first.")
    model = EntityMatchingModel()
    model.load(model_path)
    print(f"Using decision threshold: {model.threshold:.2f}")
    
    # Load Test S1
    t1_path = os.path.join(data_dir, "test", "test_source1.tsv")
    t2_path = os.path.join(data_dir, "test", "test_source2.tsv")
    t3_path = os.path.join(data_dir, "test", "test_source3.tsv")
    
    print("Loading test reference dataset (Source 1)...")
    t1_df = pd.read_csv(t1_path, sep="\t")
    print(f"Test Source 1 loaded: {len(t1_df):,} entities.")
    
    countries = list(t1_df["country"].unique())
    print(f"Test countries detected: {countries}")
    
    print("\nPreparing output files for streaming...", flush=True)
    f_match = open(matching_file, "w", encoding="utf-8")
    f_cand = open(candidate_file, "w", encoding="utf-8")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.flush()
    f_cand.flush()
    
    total_written = 0
    total_matches_count = 0
    
    # Process country-by-country with isolated memory footprint
    for country in countries:
        print(f"\n---------------------------------------------", flush=True)
        print(f"   Processing Country Partition: {country}   ", flush=True)
        print(f"---------------------------------------------", flush=True)
        
        country_s1 = t1_df[t1_df["country"] == country]
        
        # Stream and collect S2 and S3 targets for this specific country only
        print(f"Reading S2 and S3 records for {country}...", flush=True)
        country_targets_list = []
        for chunk in pd.read_csv(t2_path, sep="\t", chunksize=250000):
            sub = chunk[chunk["country"] == country]
            if len(sub):
                country_targets_list.append(sub)
        for chunk in pd.read_csv(t3_path, sep="\t", chunksize=250000):
            sub = chunk[chunk["country"] == country]
            if len(sub):
                country_targets_list.append(sub)
                
        country_targets = pd.concat(country_targets_list, ignore_index=True)
        del country_targets_list
        print(f"Country {country}: S1={len(country_s1):,}, Targets(S2+S3)={len(country_targets):,}", flush=True)
        
        # Build index for this country
        print(f"Building blocking index for {country}...", flush=True)
        engine = BlockingEngine()
        engine.index_targets(country_targets)
        
        # Process S1 entities in batches
        batch_size = 25000
        for start_idx in range(0, len(country_s1), batch_size):
            batch_s1 = country_s1.iloc[start_idx : start_idx + batch_size]
            
            # Step 1: Candidate generation
            batch_candidates = {}
            pairs_to_score = []
            
            for _, s1_row in batch_s1.iterrows():
                s1_id = s1_row["entity_id"]
                cname1 = clean_name(str(s1_row["business_name"]))
                caddr1 = clean_addr(str(s1_row["business_address"]))
                
                cands = engine.get_candidates_for_entity(cname1, caddr1, country, top_k=20)
                batch_candidates[s1_id] = cands
                
                for cid in cands:
                    cname2 = engine.target_names.get(cid, "")
                    caddr2 = engine.target_addrs.get(cid, "")
                    is_s2 = cid.startswith("S2-")
                    pairs_to_score.append((s1_id, cid, cname1, caddr1, cname2, caddr2, is_s2))
                    
            matched_by_s1 = defaultdict(list)
            if pairs_to_score:
                # Step 2: Feature extraction for candidate pairs
                X_batch = [
                    extract_pairwise_features(cname1, caddr1, cname2, caddr2, is_s2)
                    for (s1_id, cid, cname1, caddr1, cname2, caddr2, is_s2) in pairs_to_score
                ]
                X_batch = np.array(X_batch)
                
                # Step 3: Model scoring
                probs = model.predict_proba(X_batch)
                
                # Step 4: High-precision thresholding
                for (s1_id, cid, _, _, _, _, _), prob in zip(pairs_to_score, probs):
                    if prob >= model.threshold:
                        matched_by_s1[s1_id].append(cid)
                        
            # Write batch directly to disk
            for _, s1_row in batch_s1.iterrows():
                s1_id = s1_row["entity_id"]
                c_list = batch_candidates.get(s1_id, [])
                m_list = matched_by_s1.get(s1_id, [])
                
                c_dedup = list(dict.fromkeys(c_list))
                m_dedup = list(dict.fromkeys(m_list))
                
                f_cand.write(f"{s1_id}\t{','.join(c_dedup)}\n")
                f_match.write(f"{s1_id}\t{','.join(m_dedup)}\n")
                if m_dedup:
                    total_matches_count += 1
                    
            f_cand.flush()
            f_match.flush()
            total_written += len(batch_s1)
            pct = total_written / len(t1_df) * 100
            print(f"[{country}] {start_idx + len(batch_s1):,}/{len(country_s1):,} ({min(100.0, (start_idx + len(batch_s1))/len(country_s1)*100):.1f}%) | Total: {total_written:,}/{len(t1_df):,} ({pct:.1f}%) | Matched entities: {total_matches_count:,}", flush=True)
            
        del country_targets, engine
        import gc
        gc.collect()
        print(f"Completed country {country}.", flush=True)
        
    f_match.close()
    f_cand.close()
    print(f"\nSuccessfully generated {matching_file} and {candidate_file} ({total_written:,} rows total)!", flush=True)
    
    # Validate with official validator script
    print("\nValidating generated submission files with official validator...")
    val_script = os.path.join(data_dir, "..", "utils", "validate_submission.py")
    test_dir = os.path.join(data_dir, "test")
    cmd = [
        sys.executable, val_script,
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n>>> VALIDATION SUCCESSFUL: SUBMISSION FILES ARE 100% COMPLIANT! <<<\n")
    else:
        print("\n>>> WARNING: Validation reported issues. Check output above. <<<\n")

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all",
                        help="Execution mode: train, predict, or all")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset"),
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "output"),
                        help="Path to output directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Path to save/load trained model")
    parser.add_argument("--n-train", type=int, default=20000,
                        help="Number of S1 training records to use")
                        
    args = parser.parse_args()
    
    # Normalize paths
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.model_path = os.path.abspath(args.model_path)
    
    if args.mode in ["train", "all"]:
        train_pipeline(args.data_dir, args.model_path, n_train_s1=args.n_train)
        
    if args.mode in ["predict", "all"]:
        predict_pipeline(args.data_dir, args.output_dir, args.model_path)

if __name__ == "__main__":
    main()
