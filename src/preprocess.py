"""Text preprocessing and normalization for entity resolution."""
import re
import unicodedata
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# ─── Abbreviation Maps ───────────────────────────────────────────────────────

# Business name abbreviations → canonical forms
NAME_ABBREVIATIONS = {
    'corp': 'corporation', 'inc': 'incorporated', 'ltd': 'limited',
    'pvt': 'private', 'co': 'company', 'intl': 'international',
    'natl': 'national', 'assoc': 'association', 'dept': 'department',
    'mfg': 'manufacturing', 'tech': 'technology', 'svcs': 'services',
    'svc': 'service', 'mgmt': 'management', 'grp': 'group',
    'hldgs': 'holdings', 'bros': 'brothers', 'jr': 'junior',
    'sr': 'senior', 'engg': 'engineering', 'engr': 'engineering',
    'eng': 'engineering', 'mkt': 'market', 'mktg': 'marketing',
    'dist': 'distribution', 'distrib': 'distribution',
    'fin': 'financial', 'govt': 'government', 'hosp': 'hospital',
    'pharm': 'pharmaceutical', 'pharma': 'pharmaceutical',
    'univ': 'university', 'edu': 'education',
}

# Address abbreviations → canonical forms
ADDRESS_ABBREVIATIONS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'cir': 'circle', 'hwy': 'highway', 'pkwy': 'parkway',
    'apt': 'apartment', 'ste': 'suite', 'fl': 'floor',
    'bldg': 'building', 'dept': 'department',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast',
    'sw': 'southwest', 'no': 'number',
}

# Tokens to REMOVE from business names (too generic for matching/blocking)
NAME_REMOVE_TOKENS = {
    'llc', 'llp', 'plc', 'sa', 'sarl', 'sas', 'gmbh', 'ag',
    'dba', 'fka', 'aka',
}

# Stopwords for blocking (removed from blocking keys, kept in clean name)
BLOCKING_STOPWORDS = {
    'the', 'of', 'and', 'a', 'an', 'in', 'for', 'to', 'at', 'by',
    'on', 'with', 'from', 'or', 'its', 'is', 'as', 'it', 'be',
    'corporation', 'incorporated', 'limited', 'private', 'company',
    'enterprises', 'solutions', 'services', 'group', 'holdings',
    'industries', 'international', 'association', 'foundation',
    'partners', 'consulting', 'consultants', 'management', 'systems',
    'technologies', 'global', 'india', 'us', 'usa', 'america',
}


def normalize_text(text):
    """Normalize text: lowercase, unicode NFKD, strip punctuation."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return ''
    # Unicode normalize (decomposes accented chars, normalizes width variants)
    text = unicodedata.normalize('NFKD', text)
    # Lowercase
    text = text.lower().strip()
    # Replace & with and
    text = text.replace('&', ' and ')
    # Remove punctuation but keep unicode letters/digits and spaces
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def expand_abbreviations(text, abbrev_map):
    """Expand known abbreviations in text."""
    if not text:
        return text
    tokens = text.split()
    expanded = []
    for token in tokens:
        if token in abbrev_map:
            expanded.append(abbrev_map[token])
        elif token not in NAME_REMOVE_TOKENS:
            expanded.append(token)
    return ' '.join(expanded)


def preprocess_name(name):
    """Full preprocessing for business names."""
    text = normalize_text(name)
    text = expand_abbreviations(text, NAME_ABBREVIATIONS)
    return text


def preprocess_address(address):
    """Full preprocessing for business addresses."""
    text = normalize_text(address)
    text = expand_abbreviations(text, ADDRESS_ABBREVIATIONS)
    return text


def get_blocking_tokens(name):
    """Extract tokens for blocking (stopwords and short tokens removed)."""
    if not name:
        return []
    tokens = name.split()
    return [t for t in tokens if t not in BLOCKING_STOPWORDS and len(t) > 1]


def preprocess_dataframe(df):
    """Preprocess all entity records in a dataframe.

    Adds columns:
        - name_clean: normalized business name
        - addr_clean: normalized business address
        - blocking_text: space-joined blocking tokens (for CountVectorizer)
    """
    df = df.copy()
    print(f"    Preprocessing {len(df)} records...")

    df['name_clean'] = df['business_name'].apply(preprocess_name)
    df['addr_clean'] = df['business_address'].apply(preprocess_address)

    # Blocking tokens (name + address)
    df['blocking_text'] = (df['name_clean'].fillna('') + ' ' + df['addr_clean'].fillna('')).apply(
        lambda x: ' '.join(get_blocking_tokens(x))
    )

    return df
