"""End-to-end entity resolution pipeline: train and predict."""
import os
import time
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import config
from .preprocess import preprocess_dataframe
from .blocking import generate_candidates
from .features import compute_features_batch, FEATURE_COLUMNS
from .evaluate import evaluate_predictions, evaluate_blocking_recall
from .matcher import EntityMatcher


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_sources(s1_path, s2_path, s3_path, gt_path=None):
    """Load source TSVs and optionally the ground-truth file."""
    print("  Loading data...")
    s1 = pd.read_csv(s1_path, sep='\t', dtype=str)
    s2 = pd.read_csv(s2_path, sep='\t', dtype=str)
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    print(f"    S1: {len(s1):,}   S2+S3: {len(s23):,}")

    gt = None
    if gt_path:
        gt = pd.read_csv(gt_path, sep='\t', dtype=str)
        print(f"    Ground truth: {len(gt):,} entries")

    return s1, s23, gt


def parse_ground_truth(gt_df):
    """Convert ground-truth DF → {s1_id: set(matched_ids)}."""
    result = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        raw = row.get('matched_entity_ids', '')
        if pd.isna(raw) or str(raw).strip() == '':
            result[s1_id] = set()
        else:
            result[s1_id] = set(str(raw).split(','))
    return result


def build_lookup(df):
    """Build {entity_id: {'name_clean': …, 'addr_clean': …}} for feature computation."""
    names = df['name_clean'].tolist()
    addrs = df['addr_clean'].tolist()
    ids = df['entity_id'].tolist()
    return {eid: {'name_clean': n, 'addr_clean': a} for eid, n, a in zip(ids, names, addrs)}


def create_training_pairs(candidates, ground_truth, neg_ratio=None):
    """Generate labelled (s1_id, s23_id, label) pairs from blocking output.

    Positives = candidates that ARE in ground truth.
    Negatives = candidates that are NOT, capped per entity.
    """
    if neg_ratio is None:
        neg_ratio = config.NEGATIVE_RATIO

    pairs = []
    for s1_id, cand_ids in candidates.items():
        true_ids = ground_truth.get(s1_id, set())

        pos = [c for c in cand_ids if c in true_ids]
        neg = [c for c in cand_ids if c not in true_ids]

        for c in pos:
            pairs.append((s1_id, c, 1))

        # Cap negatives relative to positives (at least 1 neg per entity with candidates)
        n_neg = min(len(neg), max(neg_ratio * max(len(pos), 1), 1))
        np.random.shuffle(neg)
        for c in neg[:int(n_neg)]:
            pairs.append((s1_id, c, 0))

    return pairs


# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ═════════════════════════════════════════════════════════════════════════════

def train_pipeline():
    """Full training pipeline: load → preprocess → block → features → LightGBM → threshold."""
    t0 = time.time()
    print("=" * 65)
    print("  ENTITY RESOLUTION PIPELINE - TRAINING")
    print("=" * 65)

    # 1 ── Load ────────────────────────────────────────────────────────────
    s1_full, s23, gt_df = load_sources(
        config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3, config.TRAIN_GT,
    )
    ground_truth = parse_ground_truth(gt_df)
    del gt_df

    # 2 ── Sample S1 entities ──────────────────────────────────────────────
    sample_n = min(config.TRAIN_SAMPLE_SIZE, len(s1_full))
    np.random.seed(42)
    sample_ids = np.random.choice(s1_full['entity_id'].values, size=sample_n, replace=False)
    n_val  = int(sample_n * config.VAL_RATIO)
    val_ids   = set(sample_ids[:n_val])
    train_ids = set(sample_ids[n_val:])
    print(f"\n  Sampled {sample_n:,} S1 entities  (train {len(train_ids):,} / val {len(val_ids):,})")

    s1 = s1_full[s1_full['entity_id'].isin(sample_ids)].copy()
    del s1_full
    gc.collect()

    # 3 ── Preprocess ──────────────────────────────────────────────────────
    print("\n  Preprocessing...")
    s1  = preprocess_dataframe(s1)
    s23 = preprocess_dataframe(s23)
    print(f"    Done ({time.time()-t0:.0f}s elapsed)")

    # 4 ── Blocking ────────────────────────────────────────────────────────
    print("\n  Blocking / Candidate Generation...")
    candidates = generate_candidates(s1, s23)

    sample_ids_set = set(sample_ids)
    sample_gt = {k: v for k, v in ground_truth.items() if k in sample_ids_set}
    br = evaluate_blocking_recall(candidates, sample_gt)
    print(f"\n  Blocking recall (sample): {br:.4f}")

    # 5 ── Lookups ─────────────────────────────────────────────────────────
    print("\n  Building entity lookups...")
    s1_lookup  = build_lookup(s1)
    s23_lookup = build_lookup(s23)
    del s23; gc.collect()

    # 6 ── Training pairs ──────────────────────────────────────────────────
    print("\n  Creating training pairs...")
    train_cands = {k: v for k, v in candidates.items() if k in train_ids}
    val_cands   = {k: v for k, v in candidates.items() if k in val_ids}
    train_gt    = {k: v for k, v in ground_truth.items() if k in train_ids}
    val_gt      = {k: v for k, v in ground_truth.items() if k in val_ids}

    train_pairs = create_training_pairs(train_cands, train_gt)
    val_pairs   = create_training_pairs(val_cands,   val_gt)

    n_tp = sum(1 for _,_,l in train_pairs if l == 1)
    n_vp = sum(1 for _,_,l in val_pairs   if l == 1)
    print(f"    Train pairs: {len(train_pairs):,}  ({n_tp:,} pos)")
    print(f"    Val   pairs: {len(val_pairs):,}  ({n_vp:,} pos)")

    # 7 ── Features ────────────────────────────────────────────────────────
    print("\n  Computing features...")
    train_xy = [(s, c) for s, c, _ in train_pairs]
    val_xy   = [(s, c) for s, c, _ in val_pairs]
    y_train  = np.array([l for _, _, l in train_pairs], dtype=np.float32)
    y_val    = np.array([l for _, _, l in val_pairs],   dtype=np.float32)

    X_train = compute_features_batch(train_xy, s1_lookup, s23_lookup)
    X_val   = compute_features_batch(val_xy,   s1_lookup, s23_lookup)
    print(f"    X_train {X_train.shape}   X_val {X_val.shape}")

    # 8 ── Train model ─────────────────────────────────────────────────────
    print("\n  Training LightGBM...")
    matcher = EntityMatcher()
    matcher.train(X_train, y_train, X_val, y_val)

    # 9 ── Threshold tuning ────────────────────────────────────────────────
    print("\n  Tuning threshold (F_0.5 on validation)...")
    val_s1  = [s for s, _, _ in val_pairs]
    val_s23 = [c for _, c, _ in val_pairs]
    matcher.tune_threshold(X_val, y_val, val_s1, val_s23, val_gt)

    # 10 ── Final validation score ─────────────────────────────────────────
    print("\n  Final validation evaluation...")
    probs = matcher.predict(X_val)
    preds = {}
    for i, (s1_id, s23_id) in enumerate(zip(val_s1, val_s23)):
        if probs[i] >= matcher.threshold:
            preds.setdefault(s1_id, set()).add(s23_id)
    for s1_id in val_gt:
        preds.setdefault(s1_id, set())
    f05 = evaluate_predictions(preds, val_gt)
    print(f"\n  * Validation F_0.5 = {f05:.4f}")

    # 11 ── Save ───────────────────────────────────────────────────────────
    print("\n  Saving model...")
    matcher.save()

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"{'=' * 65}")
    return matcher


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ═════════════════════════════════════════════════════════════════════════════

def predict_pipeline():
    """Full prediction pipeline: load → preprocess → block → features → predict → write."""
    t0 = time.time()
    print("=" * 65)
    print("  ENTITY RESOLUTION PIPELINE - PREDICTION")
    print("=" * 65)

    # 1 ── Load model ──────────────────────────────────────────────────────
    print("\n  Loading model...")
    matcher = EntityMatcher()
    matcher.load()

    # 2 ── Load test data ──────────────────────────────────────────────────
    s1, s23, _ = load_sources(config.TEST_S1, config.TEST_S2, config.TEST_S3)
    all_s1_ids = set(s1['entity_id'].values)

    # 3 ── Preprocess ──────────────────────────────────────────────────────
    print("\n  Preprocessing...")
    s1  = preprocess_dataframe(s1)
    s23 = preprocess_dataframe(s23)

    # 4 ── Blocking ────────────────────────────────────────────────────────
    print("\n  Blocking / Candidate Generation...")
    candidates = generate_candidates(s1, s23)

    # 5 ── Lookups ─────────────────────────────────────────────────────────
    print("\n  Building entity lookups...")
    s1_lookup  = build_lookup(s1)
    s23_lookup = build_lookup(s23)
    del s1, s23; gc.collect()

    # 6 ── Feature + Predict (chunked) ─────────────────────────────────────
    print("\n  Computing features & predicting...")
    final_matches = {}
    candidate_out = {}

    ids_with_cands    = [sid for sid in all_s1_ids if candidates.get(sid)]
    ids_without_cands = [sid for sid in all_s1_ids if not candidates.get(sid)]

    for sid in ids_without_cands:
        final_matches[sid] = set()
        candidate_out[sid] = []

    print(f"    With candidates:    {len(ids_with_cands):,}")
    print(f"    Singletons (block): {len(ids_without_cands):,}")

    BATCH = 50_000
    for bstart in tqdm(range(0, len(ids_with_cands), BATCH), desc="    Predict", unit="batch"):
        batch_ids = ids_with_cands[bstart : bstart + BATCH]

        pairs, p_s1, p_s23 = [], [], []
        for s1_id in batch_ids:
            cands = candidates.get(s1_id, [])
            candidate_out[s1_id] = cands
            for cid in cands:
                pairs.append((s1_id, cid))
                p_s1.append(s1_id)
                p_s23.append(cid)

        if not pairs:
            for s1_id in batch_ids:
                final_matches.setdefault(s1_id, set())
            continue

        X = compute_features_batch(pairs, s1_lookup, s23_lookup, show_progress=False)
        probs = matcher.predict(X)

        for i, (s1_id, cid) in enumerate(pairs):
            if probs[i] >= matcher.threshold:
                final_matches.setdefault(s1_id, set()).add(cid)

        for s1_id in batch_ids:
            final_matches.setdefault(s1_id, set())

    # 7 ── Write outputs ───────────────────────────────────────────────────
    print("\n  Writing output files...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    sorted_s1 = sorted(all_s1_ids)

    with open(config.MATCHING_OUTPUT, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for sid in sorted_s1:
            m = final_matches.get(sid, set())
            f.write(f"{sid}\t{','.join(sorted(m)) if m else ''}\n")

    with open(config.CANDIDATE_OUTPUT, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for sid in sorted_s1:
            c = candidate_out.get(sid, [])
            f.write(f"{sid}\t{','.join(sorted(c)) if c else ''}\n")

    n_matched = sum(1 for v in final_matches.values() if v)
    n_total   = sum(len(v) for v in final_matches.values())
    print(f"\n    Total S1:     {len(all_s1_ids):,}")
    print(f"    With matches: {n_matched:,}")
    print(f"    Total links:  {n_total:,}")
    print(f"    Singletons:   {len(all_s1_ids) - n_matched:,}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  Prediction complete in {elapsed/60:.1f} min")
    print(f"  -> {config.MATCHING_OUTPUT}")
    print(f"  -> {config.CANDIDATE_OUTPUT}")
    print(f"{'=' * 65}")
