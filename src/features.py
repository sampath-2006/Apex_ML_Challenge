"""String-similarity feature engineering for entity pairs."""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import jellyfish

import os
# Force transformers to only use PyTorch and avoid loading TensorFlow (which causes Protobuf crashes)
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

try:
    from sentence_transformers import SentenceTransformer
    # Load a tiny, fast model for CPU
    _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    _EMB_CACHE = {}  # Cache string -> embedding to save massive CPU time
except ImportError:
    _ST_MODEL = None
    _EMB_CACHE = {}


# ─── Feature column order (must be consistent between train & predict) ────────
FEATURE_COLUMNS = [
    # Name features (11)
    'name_ratio', 'name_partial_ratio', 'name_token_sort_ratio',
    'name_token_set_ratio', 'name_jaro_winkler', 'name_jaccard',
    'name_shared_count', 'name_len_ratio',
    'name_soundex_match', 'name_metaphone_match',
    'name_embedding_cosine',
    # Address features (9)
    'has_address_both', 'addr_ratio', 'addr_partial_ratio',
    'addr_token_sort_ratio', 'addr_jaccard', 'addr_shared_count',
    'addr_number_jaccard', 'addr_zip_match', 'addr_pobox_match'
]


def _safe_jaccard(set1, set2):
    """Jaccard similarity: |A∩B| / |A∪B|."""
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def _safe_len_ratio(a, b):
    """min(len) / max(len), 0 when both empty."""
    la, lb = len(a), len(b)
    mx = max(la, lb)
    return min(la, lb) / mx if mx else 0.0


_NUM_RE = re.compile(r'\d+')
_ZIP_RE = re.compile(r'\b\d{5,6}\b')
_POBOX_RE = re.compile(r'\b(?:po box|p o box|box)\s*(\d+)\b')

def _match_score(set1, set2):
    """0.5 if missing, 1.0 if match, 0.0 if conflict."""
    if not set1 or not set2:
        return 0.5
    if set1 & set2:
        return 1.0
    return 0.0


def compute_pair_features(name1, name2, addr1, addr2):
    """Compute the full feature vector for one (S1, S2/S3) pair.

    All inputs should already be preprocessed (lowercase, normalised, etc.).
    Returns a dict keyed by FEATURE_COLUMNS names.
    """
    n1 = name1 or ''
    n2 = name2 or ''

    feats = {}

    # ── Name character-level ──────────────────────────────────────────────
    feats['name_ratio']           = fuzz.ratio(n1, n2) / 100.0
    feats['name_partial_ratio']   = fuzz.partial_ratio(n1, n2) / 100.0
    feats['name_token_sort_ratio'] = fuzz.token_sort_ratio(n1, n2) / 100.0
    feats['name_token_set_ratio']  = fuzz.token_set_ratio(n1, n2) / 100.0
    feats['name_jaro_winkler']    = JaroWinkler.similarity(n1, n2)

    if n1 and n2:
        feats['name_soundex_match']   = 1.0 if jellyfish.soundex(n1) == jellyfish.soundex(n2) else 0.0
        feats['name_metaphone_match'] = 1.0 if jellyfish.metaphone(n1) == jellyfish.metaphone(n2) else 0.0
    else:
        feats['name_soundex_match']   = 0.0
        feats['name_metaphone_match'] = 0.0

    # ── Name token-level ──────────────────────────────────────────────────
    t1 = set(n1.split()) if n1 else set()
    t2 = set(n2.split()) if n2 else set()
    feats['name_jaccard']      = _safe_jaccard(t1, t2)
    feats['name_shared_count'] = float(len(t1 & t2))
    feats['name_len_ratio']    = _safe_len_ratio(n1, n2)

    # ── Address ───────────────────────────────────────────────────────────
    a1 = str(addr1) if addr1 and pd.notna(addr1) else ''
    a2 = str(addr2) if addr2 and pd.notna(addr2) else ''
    both = bool(a1) and bool(a2)
    feats['has_address_both'] = float(both)

    if both:
        feats['addr_ratio']           = fuzz.ratio(a1, a2) / 100.0
        feats['addr_partial_ratio']   = fuzz.partial_ratio(a1, a2) / 100.0
        feats['addr_token_sort_ratio'] = fuzz.token_sort_ratio(a1, a2) / 100.0
        at1 = set(a1.split())
        at2 = set(a2.split())
        feats['addr_jaccard']      = _safe_jaccard(at1, at2)
        feats['addr_shared_count'] = float(len(at1 & at2))
        nums1 = set(_NUM_RE.findall(a1))
        nums2 = set(_NUM_RE.findall(a2))
        feats['addr_number_jaccard'] = _safe_jaccard(nums1, nums2)
        
        zips1 = set(_ZIP_RE.findall(a1))
        zips2 = set(_ZIP_RE.findall(a2))
        feats['addr_zip_match'] = _match_score(zips1, zips2)
        
        pobox1 = set(_POBOX_RE.findall(a1))
        pobox2 = set(_POBOX_RE.findall(a2))
        feats['addr_pobox_match'] = _match_score(pobox1, pobox2)
    else:
        feats['addr_ratio'] = 0.0
        feats['addr_partial_ratio'] = 0.0
        feats['addr_token_sort_ratio'] = 0.0
        feats['addr_jaccard'] = 0.0
        feats['addr_shared_count'] = 0.0
        feats['addr_number_jaccard'] = 0.0
        feats['addr_zip_match'] = 0.5
        feats['addr_pobox_match'] = 0.5

    return feats


def compute_features_batch(pairs, s1_lookup, s23_lookup, show_progress=True):
    """Compute feature matrix for a list of (s1_id, s23_id) pairs.

    Args:
        pairs:       list of (s1_entity_id, s23_entity_id)
        s1_lookup:   {entity_id: {'name_clean': str, 'addr_clean': str}}
        s23_lookup:  same structure
        show_progress: show tqdm bar

    Returns:
        np.ndarray of shape (len(pairs), len(FEATURE_COLUMNS))
    """
    n = len(pairs)
    X = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)

    # 1. Pre-compute missing embeddings for the batch
    if _ST_MODEL is not None:
        unique_names = set()
        for s1_id, s23_id in pairs:
            if n1 := s1_lookup.get(s1_id, {}).get('name_clean', ''): unique_names.add(n1)
            if n2 := s23_lookup.get(s23_id, {}).get('name_clean', ''): unique_names.add(n2)
        
        missing = list(unique_names - set(_EMB_CACHE.keys()))
        if missing:
            embs = _ST_MODEL.encode(missing, batch_size=256, show_progress_bar=False)
            for name, emb in zip(missing, embs):
                _EMB_CACHE[name] = emb

    iterator = enumerate(pairs)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc="    Features", unit="pair")

    for idx, (s1_id, s23_id) in iterator:
        s1  = s1_lookup.get(s1_id, {})
        s23 = s23_lookup.get(s23_id, {})
        n1 = s1.get('name_clean', '')
        n2 = s23.get('name_clean', '')
        
        feats = compute_pair_features(
            n1, n2,
            s1.get('addr_clean', ''),
            s23.get('addr_clean', ''),
        )
        
        # Add embedding cosine
        cos_sim = 0.0
        if n1 and n2 and _ST_MODEL is not None:
            e1 = _EMB_CACHE.get(n1)
            e2 = _EMB_CACHE.get(n2)
            if e1 is not None and e2 is not None:
                denom = np.linalg.norm(e1) * np.linalg.norm(e2)
                if denom > 0:
                    cos_sim = float(np.dot(e1, e2) / denom)
        feats['name_embedding_cosine'] = cos_sim

        X[idx] = [feats[c] for c in FEATURE_COLUMNS]

    return X


# ─── Import tqdm at module level (used in batch function) ─────────────────────
from tqdm import tqdm
