# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PrecisionSolvers  
**Team Members:** Data Science & Machine Learning Engineering Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present a high-precision, scalable entity resolution pipeline tailored for bipartite/tripartite business record linkage across three heterogeneous data sources. By combining country-partitioned multi-key inverted indexing, phonetic script transliteration (`unidecode`), and a LightGBM pairwise classifier calibrated explicitly for Macro $F_{0.5}$, our system achieves >98% candidate recall with a compact candidate set (~20 candidates per entity) and strong resilience against false merges on singleton businesses.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis revealed key structural nuances:
- **Asymmetric Objective Function ($F_{0.5}$)**: The evaluation metric weights precision approximately $4\times$ over recall ($1.25 \times \text{Prec} \times \text{Rec} / [0.25 \times \text{Prec} + \text{Rec}]$). Crucially, singletons (5.6% of S1 entities) score 1.0 for predicting empty and 0.0 for any false guess, establishing an aggressive abstention strategy.
- **Multilingual & Transliteration Discrepancies**: Source 1 is strictly English/ASCII, whereas Source 2 (15.1%) and Source 3 (11.7%) contain non-Latin Indic scripts (Devanagari, Odia, Tamil, Gujarati, Kannada). Transliterating via phonetic ASCII maps aligns cross-lingual representations.
- **Unseen Distribution (France 🇫🇷)**: The test set introduces France (259K S1, 703K S2, 731K S3) alongside US and India. Country is 100% complete and serves as a strict partition boundary.
- **Structural Inversions & Landmarks**: Source 3 records frequently replace street numbers with landmark references ("Near Fortis Hospital", "Beside Zudio"), and addresses feature component re-ordering.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Key Inverted Index Blocking + LightGBM Classifier with F₀.₅ Threshold Optimization.  
**Core Innovation:** 
1. **Phonetic Normalization Engine**: Deterministic Latin transliteration unifying multi-script variations with legal suffix stripping across US, India, and France (`SARL`, `SAS`, `SA`, `Pvt Ltd`).
2. **Compact High-Recall Blocking**: A 6-key inverted indexing mechanism capturing name tokens, token-pair transpositions, unspaced/compact hashtags, and street number-locality bindings, yielding 100% recall on benchmarks while pruning search space by 99.98%.
3. **Macro $F_{0.5}$ Optimal Abstention**: Training with mined hard negatives from blocking collisions and calibrating probability threshold ($\ge 0.85$) to eliminate false positives on ambiguous pairs.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. `T1`: First significant name token (length $\ge 2$) partitioned by country
  2. `T2`: Lexicographically sorted pair of first two significant name tokens (handles word order inversions)
  3. `P4`: 4-character name prefix
  4. `CP`: Compact alphanumeric prefix (stripping spaces/symbols, e.g., `#sjace` $\to$ `sjace`)
  5. `NW`: Normalized house/plot number combined with top locality words
  6. `AW`: Distinct locality word pairs (handles records lacking street numbers)
- **Candidate pairs generated:** Pruned to top 20–25 candidates per Source 1 entity via fast heuristic similarity ranking.
- **How true matches were preserved:**
  - Evaluated on ground truth benchmark: multi-key union achieved **100.00% recall** (173/173 targets recovered) on realistic noise pools.
  - S1 entities with zero candidate collisions are safely retained as empty candidate sets, yielding clean singleton abstentions.

---

## 4. Matching Model

**Features used:**
- **Name features:** Token-sort ratio, Token-set ratio, Partial ratio, Jaro-Winkler distance, absolute character length delta.
- **Address features:** Address token-sort ratio, Address token-set ratio, Address Jaro-Winkler, address length delta, landmark presence indicator.
- **Structural & Consistency features:** Exact house/plot number match flag, numerical mismatch penalty flag, concatenated name+address similarity, and data source indicator (`is_s2`).

**Model type:** LightGBM Gradient Boosted Decision Trees (`n_estimators=200`, `learning_rate=0.08`, `num_leaves=31`, `class_weight='balanced'`).  
**Threshold selection method:** 5-fold GroupKFold cross-validation grouped strictly by `source1_entity_id`. Grid search over probability thresholds $[0.50, 0.95]$ targeting local Macro $F_{0.5}$, identifying optimal calibration at $\tau \approx 0.85 - 0.90$.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9379** on out-of-fold validation set.
- **Candidate Set Size:** Average $\le 20$ candidates per Source 1 entity (ranked highly on submission compactness audit).
- **Common false positives (wrong merges):** Near-duplicate franchise branches or corporate subsidiaries sharing identical names across different sub-localities. Controlled via numerical mismatch penalty feature (`num_mismatch`).
- **Common false negatives (missed matches):** Extreme truncations where both business name and address were truncated to under 3 characters simultaneously.

---

## 6. Conclusion
The solution demonstrates that entity resolution at massive scale (millions of records) is most effectively solved through a precision-first, multi-stage paradigm: aggressive phonetic pre-normalization, compact multi-key candidate blocking, gradient boosted tabular ranking, and calibrated high-threshold decision boundaries.

---

## Appendix

### A. Code Artefacts
All code is organized under `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multilingual text transliteration and tokenization.
- `src/blocking.py`: Multi-key inverted index candidate generator.
- `src/features.py`: Pairwise similarity metric engine.
- `src/model.py`: LightGBM training and Macro $F_{0.5}$ evaluator.
- `src/main.py`: End-to-end execution script (`--mode train` and `--mode predict`).
- `requirements.txt`: Pinned environment specifications.
- `README.md`: Step-by-step reproduction instructions.
