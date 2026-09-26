# Apex — Progress Log

## Amazon ML Challenge 2026: Business Entity Resolution

**Challenge Window**: Sept 25, 2026 12:00 AM IST → Sept 27, 2026 11:59 PM IST  
**Submissions**: Max 5/day (15 total)

---

## Day 1 — Sept 25, 2026

### 21:14 IST — Problem Study
- Read both PDFs: problem statement + guidelines
- Explored dataset structure and file sizes
- Key findings:
  - Scale: 2.2M S1 vs 10.3M S2+S3 (train), 1.7M vs 10M (test)
  - Metric: F₀.₅ (precision-heavy)
  - Countries: US + India (train), + France (test, unseen)
  - Model limit: ≤8B params, open license

### 21:23 IST — Strategy Decision
- Chose **Option A**: Lightweight pipeline first, optimize later
- Rejected heavy transformer approach due to time + scale constraints
- Plan: String similarity + LightGBM, no embeddings in v1

### 21:44 IST — Pipeline Design
- Designed 5-stage architecture: Preprocess → Block → Features → Match → Threshold
- Created detailed plan with all feature specs and blocking strategy

### 21:55 IST — Pipeline Implementation (v1)
- Created all source modules:
  - `src/config.py` — Central configuration
  - `src/preprocess.py` — Unicode normalization, abbreviation expansion
  - `src/blocking.py` — Sparse matrix token-overlap blocking (per country)
  - `src/features.py` — 15 string-similarity features (rapidfuzz)
  - `src/evaluate.py` — F₀.₅ scorer + blocking recall
  - `src/matcher.py` — LightGBM with auto class-weight + threshold tuning
  - `src/pipeline.py` — End-to-end orchestration
  - `run_train.py` / `run_predict.py` — Entry points
- All imports verified clean
- Dependencies: pandas, numpy, lightgbm, rapidfuzz, scikit-learn, tqdm, scipy

### 23:58 IST — Debugging Performance Bottleneck
- **Breakthrough 1 (O(N*M) bug in Set Lookup):** Training pipeline was hanging. Found that `set(sample_ids)` was being created inside a list comprehension looping over 2.2 million `ground_truth` rows. Extracted the set creation to a variable before the loop, saving ~30-40 minutes of execution time.
- **Breakthrough 2 (Pandas Iterrows Speed):** Found a massive bottleneck where `df.iterrows()` was iterating over 10.3 million rows in `build_lookup()`. Replaced it with a vectorized `.tolist()` zip method, achieving a 1000x speedup (bringing lookup time down from >20 mins to ~2 seconds).
- Temporarily reduced `TRAIN_SAMPLE_SIZE` from 100K to 10K for faster interactive verification of the end-to-end pipeline.

### 00:13 IST — Unicode Encoding Bug Fix
- The training finished completely in ~5 minutes but crashed on the very last line during threshold tuning because the Windows terminal (`cp1252` charmap) couldn't encode the unicode arrow `→` in the print statement. 
- Replaced all unicode arrows (`→`, `◀`) with ASCII (`->`, `<--`) across `matcher.py` and `pipeline.py`.

### 00:14 IST — Final Training Run (v1)
- Started the final fixed `run_train.py`.
- Status: ✅ **Completed** (Took 5.6 minutes)
- **Validation F₀.₅:** 0.5524 (at optimized threshold 0.65)
- **Artifacts Saved:** 
  - `models/lgbm_model.txt`
  - `models/threshold.txt`

### 00:26 IST — Final Prediction Run (v1)
- Ran `run_predict.py` across the entire 11.7 million row test set.
- Status: ✅ **Completed** (Took 35.7 minutes)
- **Results:**
  - Total S1 Entities: 1,732,544
  - Entities with matches: 1,016,864
  - Total links found: 2,758,543
  - Singletons (no match): 715,680
- **Submission Artifacts Generated:**
  - `output/matching_results.tsv`
  - `output/candidate_pairs.tsv`

## Day 2 — Sept 26, 2026

### 02:00 IST — Feature & Sample Size Upgrades
- **v2 Implemented**: Added phonetic features (Soundex, Metaphone) using `jellyfish`.
- **v3 Implemented**: Added specific parsing and matching for Zip Codes (5-6 digits) and PO Box numbers.
- **Config**: Increased `TRAIN_SAMPLE_SIZE` from 10k to 100k to let the model generalize better. Increased LightGBM rounds from 500 to 1500.

### 02:15 IST — The Blocking Breakthrough
- **Discovery**: Realized that the initial training run had a `Blocking recall` of only **0.43**. We were throwing away 57% of true positive matches before LightGBM even saw them!
- **Fix**: Modified `src/preprocess.py` to include `addr_clean` tokens in the blocking index. Changed `BLOCKING_TOP_K` from 20 to 100 in `config.py`.
- **Result**: Validation F₀.₅ score skyrocketed from **0.5609** to **0.9481**! Model peaked at exactly 1500 rounds with threshold 0.85. Address features (`addr_jaccard`) dominate the importance list.
- **Test Predictions Completed**: Ran `run_predict.py` with the 94.8% model. Processed 1.7M S1 entities against 10M S2/S3. Found matches for 1,662,879 entities (6.6M total links) leaving only 69k singletons. Output saved to `output/matching_results.tsv`.

### 13:00 IST — V4 (Semantic Embeddings) Success!
- **Implementation**: Bypassed TensorFlow conflicts and integrated `all-MiniLM-L6-v2` via `sentence-transformers`.
- **Caching**: Built an LRU-style dictionary cache in `src/features.py` to extract embeddings *only* for unique strings, saving days of CPU time.
- **Result**: `name_embedding_cosine` successfully entered the top 10 most important features! 
- **Validation Score**: Peaked at **0.9488** (up from 0.9482) at threshold 0.80. Model training took 43.8 minutes on CPU.

---

## Submissions Log

| # | Time | F₀.₅ (Public) | Notes |
|---|---|---|---|
| 2 | 26 Sep 26, 11:54 AM IST | **0.788** | v3 pipeline (100k samples, improved blocking, 1500 rounds) |
| 1 | 26 Sep 26, 01:13 AM IST | **0.480** | v1 baseline (LightGBM + Lexical features only) |

## Analysis: Validation vs Public LB Drop (0.948 → 0.788)
The massive jump from 0.480 to 0.788 confirms our blocking fixes worked, but the drop from 0.948 validation to 0.788 public test is significant. Hypotheses for the gap:
1. **The "France" Problem (Zero-Shot):** The test set includes a massive amount of data from France, which was entirely unseen in our training data. Our LightGBM model learned thresholds optimized for English/Hindi distributions (US/India).
2. **French Stopwords in Blocking:** Our `BLOCKING_STOPWORDS` in `preprocess.py` only contains English words! Common French words (le, la, de, societe) are not being filtered, causing false positives in the blocking phase and pushing true matches out of the top 100 candidates.
3. **Abbreviation Bias:** Our `ADDRESS_ABBREVIATIONS` dictionary expands "st" to "street", but completely misses French address abbreviations (e.g., "rue", "blvd", "av").

---

## Upgrade Roadmap

- [x] **v1**: String similarity + LightGBM (Score: 0.480)
- [x] **v2**: Add phonetic features (Soundex, Metaphone)
- [x] **v3**: Better address parsing, component-level matching
- [x] **v4**: Small encoder embeddings as additional features (Score: 0.9488 Val)
- [ ] **v5**: Ensemble / stacking / threshold refinement
