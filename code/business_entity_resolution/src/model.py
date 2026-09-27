"""
Model training, inference, and threshold calibration for Business Entity Resolution.
Implements Macro F_0.5 evaluation and LightGBM classifier training.
"""

import os
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from features import FEATURE_NAMES

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
    return float(np.mean(scores))

class EntityMatchingModel:
    """Gradient boosted matching model with F_0.5 threshold optimization."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        
    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, gt_dict: dict, pair_meta: list):
        """Train model with GroupKFold cross-validation and calibrate optimal F0.5 threshold."""
        print(f"Training LightGBM on {len(X)} candidate pairs (Positives: {sum(y)})...")
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            fold_clf = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.08,
                num_leaves=31,
                random_state=42,
                class_weight='balanced',
                verbose=-1
            )
            fold_clf.fit(X_train, y_train)
            oof_preds[val_idx] = fold_clf.predict_proba(X_val)[:, 1]
            
        # Fit final model on all data
        self.model.fit(X, y)
        
        # Optimize threshold for Macro F0.5
        best_thresh = 0.85
        best_f05 = 0.0
        print("Calibrating decision threshold on validation predictions...")
        for thresh in np.arange(0.50, 0.96, 0.05):
            preds_dict = {}
            for i, (prob, (s1_id, cid)) in enumerate(zip(oof_preds, pair_meta)):
                if prob >= thresh:
                    if s1_id not in preds_dict:
                        preds_dict[s1_id] = set()
                    preds_dict[s1_id].add(cid)
            score = compute_f05_macro(preds_dict, gt_dict)
            if score > best_f05:
                best_f05 = score
                best_thresh = float(thresh)
                
        self.threshold = best_thresh
        print(f"Optimal F_0.5 Threshold: {self.threshold:.2f} (Validation F_0.5: {best_f05:.4f})")
        return best_f05
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive class probabilities."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, filepath: str):
        """Save model and calibrated threshold to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'threshold': self.threshold}, f)
        print(f"Model saved successfully to {filepath}")
        
    def load(self, filepath: str):
        """Load model and calibrated threshold from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.threshold = data['threshold']
        print(f"Model loaded from {filepath} (Threshold: {self.threshold:.2f})")
