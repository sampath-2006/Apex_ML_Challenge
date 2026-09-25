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

---

## Submissions Log

| # | Time | F₀.₅ (Public) | Notes |
|---|---|---|---|
| 1 | — | — | v1 baseline (pending) |

---

## Upgrade Roadmap

- [ ] **v1**: String similarity + LightGBM (current)
- [ ] **v2**: Add phonetic features (Soundex, Metaphone)
- [ ] **v3**: Better address parsing, component-level matching
- [ ] **v4**: Small encoder embeddings as additional features
- [ ] **v5**: Ensemble / stacking / threshold refinement
