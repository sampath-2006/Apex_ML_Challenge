# Apex — Amazon ML Challenge 2026

## Business Entity Resolution

Given business records from 3 independent data sources with noisy, inconsistent fields, determine which records across sources refer to the **same real-world business entity**.

- **Source 1** = deduplicated reference (anchor)
- **Source 2 & 3** = noisy candidate sources
- **Goal**: For each S1 entity, find all matching S2/S3 records

### 🏆 Current Leaderboard (Public)
- **V1 Baseline (LightGBM + Lexical):** `0.480` F₀.₅ Score
- **V3 Pipeline (100k + Improved Blocking):** `0.948` F₀.₅ Score (Validation)

## Pipeline Architecture

```
Raw TSV Data → Preprocessing → Blocking → Feature Engineering → LightGBM Classifier → Threshold Tuning → Output
```

### Stage 1: Preprocessing (`src/preprocess.py`)
- Unicode NFKD normalization (handles Hindi, French, accented chars)
- Lowercase + punctuation removal
- Abbreviation expansion (Corp→Corporation, Pvt→Private, Rd→Road, etc.)
- Stopword removal for blocking tokens

### Stage 2: Blocking (`src/blocking.py`)
- **Per-country** partitioning (only compare within same country)
- Sparse binary token-document matrix via `CountVectorizer`
- Token overlap via sparse matrix dot product (chunked for memory)
- Top-K candidates per S1 entity by overlap count
- Frequency filtering: ignores tokens in >0.5% of docs (too common)

### Stage 3: Feature Engineering (`src/features.py`)
19 string-similarity features, **zero embeddings**:

| Category | Features |
|---|---|
| **Name (10)** | fuzz.ratio, partial_ratio, token_sort_ratio, token_set_ratio, Jaro-Winkler, Jaccard, shared token count, length ratio, Soundex match, Metaphone match |
| **Address (9)** | has_both_addresses, fuzz.ratio, partial_ratio, token_sort_ratio, Jaccard, shared token count, numeric token Jaccard, Zip Code match, PO Box match |

### Stage 4: Matching (`src/matcher.py`)
- LightGBM binary classifier
- Auto class-weight balancing (scale_pos_weight)
- Early stopping on validation loss

### Stage 5: Threshold Tuning
- Sweep thresholds 0.10–0.95
- Maximize **F₀.₅** (precision-heavy) on validation split
- Conservative threshold → fewer false merges

## Evaluation Metric

```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Macro-averaged per S1 entity. Precision weighted **2× over recall**.

## Dataset Scale

| Split | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

Countries: US, India (train+test), **France** (test only — unseen).

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Train
```bash
python run_train.py
```
Samples 100K S1 entities, runs blocking, trains LightGBM, tunes threshold.
Saves model to `models/`.

### Predict
```bash
python run_predict.py
```
Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### Validate
```bash
python dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test
```

## Project Structure

```
ML_Challenge/
├── src/
│   ├── config.py          # Paths, hyperparameters
│   ├── preprocess.py      # Text normalization
│   ├── blocking.py        # Candidate generation
│   ├── features.py        # String similarity features
│   ├── evaluate.py        # F₀.₅ scorer
│   ├── matcher.py         # LightGBM wrapper
│   └── pipeline.py        # End-to-end orchestration
├── run_train.py            # Training entry point
├── run_predict.py          # Prediction entry point
├── requirements.txt        # Dependencies
├── output/                 # Generated submission files
├── models/                 # Saved model + threshold
└── dataset/                # Data (gitignored)
```

## Constraints
- Model: ≤ 8B parameters, MIT/Apache 2.0 license
- No external APIs, databases, or geocoding services
- Max 5 submissions per day

## Team
**Apex**
