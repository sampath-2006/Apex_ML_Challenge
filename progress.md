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

### 21:58 IST — Training Started
- Kicked off `run_train.py`
- Config: 100K S1 sample, 80/20 train/val split, top-20 candidates, LightGBM 500 rounds
- Status: ⏳ Running...

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
