"""Configuration for the Entity Resolution Pipeline."""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "student_resource", "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── File Paths ───────────────────────────────────────────────────────────────
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.txt")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")

# ─── Blocking Parameters ─────────────────────────────────────────────────────
BLOCKING_TOP_K = 20            # Max candidates per S1 entity
BLOCKING_MAX_DF_RATIO = 0.005  # Ignore tokens in >0.5% of docs (too common)
BLOCKING_CHUNK_SIZE = 5000     # S1 entities per chunk during blocking

# ─── Training Parameters ─────────────────────────────────────────────────────
TRAIN_SAMPLE_SIZE = 10000      # S1 entities to sample for training
VAL_RATIO = 0.2                # Fraction held out for validation
NEGATIVE_RATIO = 3             # Max negative pairs per positive pair

# ─── LightGBM Hyperparameters ────────────────────────────────────────────────
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}
LGBM_NUM_ROUNDS = 500
LGBM_EARLY_STOPPING = 50

# ─── Threshold ────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.5
