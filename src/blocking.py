"""Candidate generation via token-overlap blocking with sparse matrices."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from . import config


def build_blocking_index(texts, min_df=2, max_df=None):
    """Build a CountVectorizer and transform texts to a binary sparse matrix.

    Each row is one S2/S3 record; each column is a vocabulary token.
    A cell is 1 if the token appears in that record's blocking text.
    """
    if max_df is None:
        max_df = config.BLOCKING_MAX_DF_RATIO

    vectorizer = CountVectorizer(
        binary=True,
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        min_df=min_df,
        max_df=max_df,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def find_candidates_chunk(s1_chunk, s23_matrix, s1_ids_chunk, s23_ids, top_k):
    """Find top-K candidates for a chunk of S1 entities via sparse dot product.

    Returns dict {s1_id: [candidate_ids]}.
    """
    # Token overlap count = dot product of binary vectors
    overlap = s1_chunk.dot(s23_matrix.T)  # shape: (chunk_size, n_s23)

    candidates = {}
    for i in range(overlap.shape[0]):
        row = overlap.getrow(i)
        if row.nnz == 0:
            candidates[s1_ids_chunk[i]] = []
            continue

        indices = row.indices
        data = row.data

        if len(indices) > top_k:
            top_idx = np.argsort(-data)[:top_k]
            selected = indices[top_idx]
        else:
            selected = indices

        candidates[s1_ids_chunk[i]] = [s23_ids[j] for j in selected]

    return candidates


def generate_candidates(s1_df, s23_df, top_k=None, chunk_size=None):
    """Generate candidate matches using per-country token-overlap blocking.

    For each country:
      1. Build a vocabulary from S2+S3 blocking texts (rare-enough tokens only).
      2. Represent each S1 and S2/S3 record as a binary sparse vector.
      3. Compute token overlap via sparse matrix multiplication (in chunks).
      4. Keep the top-K S2/S3 candidates per S1 entity by overlap count.

    Args:
        s1_df:  Preprocessed S1 dataframe (needs 'entity_id', 'country', 'blocking_text').
        s23_df: Preprocessed S2+S3 dataframe (same columns).
        top_k:  Max candidates per S1 entity (default from config).
        chunk_size: S1 rows per dot-product chunk (default from config).

    Returns:
        dict  {s1_entity_id: [candidate_s23_entity_ids]}
    """
    if top_k is None:
        top_k = config.BLOCKING_TOP_K
    if chunk_size is None:
        chunk_size = config.BLOCKING_CHUNK_SIZE

    all_candidates = {}
    countries = sorted(s1_df['country'].dropna().unique())

    for country in countries:
        print(f"\n  [Blocking] Country: {country}")

        s1_c = s1_df[s1_df['country'] == country].reset_index(drop=True)
        s23_c = s23_df[s23_df['country'] == country].reset_index(drop=True)

        if len(s23_c) == 0:
            print(f"    No S2/S3 records — all {len(s1_c)} S1 entities become singletons.")
            for eid in s1_c['entity_id']:
                all_candidates[eid] = []
            continue

        print(f"    S1: {len(s1_c):,}   S2/S3: {len(s23_c):,}")

        # Build sparse matrices
        s23_texts = s23_c['blocking_text'].fillna('').tolist()
        s23_ids = s23_c['entity_id'].tolist()

        try:
            vectorizer, s23_matrix = build_blocking_index(s23_texts)
        except ValueError:
            # Fallback if all tokens are too common or too rare
            print("    Vectorizer failed with default params — using min_df=1, max_df=1.0")
            vectorizer, s23_matrix = build_blocking_index(s23_texts, min_df=1, max_df=1.0)

        s1_texts = s1_c['blocking_text'].fillna('').tolist()
        s1_ids = s1_c['entity_id'].tolist()
        s1_matrix = vectorizer.transform(s1_texts)

        vocab_size = len(vectorizer.vocabulary_)
        print(f"    Vocabulary: {vocab_size:,} tokens")

        # Chunked sparse dot product
        n_s1 = len(s1_ids)
        for start in tqdm(range(0, n_s1, chunk_size),
                          desc=f"    {country}", unit="chunk"):
            end = min(start + chunk_size, n_s1)
            chunk_cands = find_candidates_chunk(
                s1_matrix[start:end], s23_matrix,
                s1_ids[start:end], s23_ids, top_k,
            )
            all_candidates.update(chunk_cands)

        # Free memory
        del s23_matrix, s1_matrix, vectorizer

    # Ensure every S1 entity has an entry
    for eid in s1_df['entity_id']:
        if eid not in all_candidates:
            all_candidates[eid] = []

    return all_candidates
