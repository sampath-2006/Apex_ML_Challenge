"""String-similarity feature engineering for entity pairs."""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


# ─── Feature column order (must be consistent between train & predict) ────────
FEATURE_COLUMNS = [
    # Name features (8)
    'name_ratio', 'name_partial_ratio', 'name_token_sort_ratio',
    'name_token_set_ratio', 'name_jaro_winkler', 'name_jaccard',
    'name_shared_count', 'name_len_ratio',
    # Address features (7)
    'has_address_both', 'addr_ratio', 'addr_partial_ratio',
    'addr_token_sort_ratio', 'addr_jaccard', 'addr_shared_count',
    'addr_number_jaccard',
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
    else:
        feats['addr_ratio'] = 0.0
        feats['addr_partial_ratio'] = 0.0
        feats['addr_token_sort_ratio'] = 0.0
        feats['addr_jaccard'] = 0.0
        feats['addr_shared_count'] = 0.0
        feats['addr_number_jaccard'] = 0.0

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

    iterator = enumerate(pairs)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc="    Features", unit="pair")

    for idx, (s1_id, s23_id) in iterator:
        s1  = s1_lookup.get(s1_id, {})
        s23 = s23_lookup.get(s23_id, {})
        feats = compute_pair_features(
            s1.get('name_clean', ''),
            s23.get('name_clean', ''),
            s1.get('addr_clean', ''),
            s23.get('addr_clean', ''),
        )
        X[idx] = [feats[c] for c in FEATURE_COLUMNS]

    return X


# ─── Import tqdm at module level (used in batch function) ─────────────────────
from tqdm import tqdm
