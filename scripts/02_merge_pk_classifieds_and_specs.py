"""02_merge_pk_classifieds_and_specs.py

---------------------------------
Joins normalized Pakistani mobile classifieds (OLX) with clean device specifications.
- Handles slugs and bare identifiers ('16-pro-max', 's25 ultra', 'pixel-8a')
- Prepends brand/make to models missing maker context
- Applies regex normalization and token-sort fuzzy matching via rapidfuzz
- Preserves local Pakistani market variables (PTA, JV status, spatial coordinates)
"""

import re

import pandas as pd
from rapidfuzz import fuzz, process


def normalize_device_key(text: str) -> str:
    """Normalizes device names for deterministic entity resolution."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Strip parenthetical annotations e.g. '(China)', '(Global)'
    text = re.sub(r"\(.*?\)", "", text)
    # Remove cellular/connectivity suffixes that vary across listings
    text = re.sub(r"\b(5g|4g|lte)\b", "", text)
    # Replace separators with whitespace
    text = re.sub(r"[\-_\/]", " ", text)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def clean_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).lower().strip()


def build_classifieds_lookup_key(make: str, model: str) -> str:
    """Constructs a composite searchable string.

    If 'model' is '16-pro-max' and 'make' is 'apple', resolves to 'apple 16 pro
    max'.
    If 'model' is already 'pixel 9 pro xl' and 'make' is 'google', resolves to
    'google pixel 9 pro xl'.
    """
    make_str = clean_str(make)
    model_str = clean_str(model)

    if model_str.startswith(make_str):
        combined = model_str
    else:
        combined = f"{make_str} {model_str}".strip()

    return normalize_device_key(combined)


def merge_pk_classifieds_with_specs(
    classifieds_path: str = "../data/pak_mobile_classifieds_2026.csv",
    specs_path: str = "../data/device_specs_reference.csv",
    output_path: str = "../data/pak_mobile_classifieds_augmented_2026.csv",
    fuzzy_cutoff: float = 80.0,
):
    print(f"Loading classifieds from: {classifieds_path}")
    df_cls = pd.read_csv(classifieds_path)

    print(f"Loading specs from: {specs_path}")
    df_specs = pd.read_csv(specs_path)

    # Generate matching keys
    print("Building composite lookup keys...")
    df_cls["match_key"] = [
        build_classifieds_lookup_key(m, mod)
        for m, mod in zip(df_cls["make"], df_cls["model"])
    ]

    df_specs["match_key"] = df_specs["model_name"].apply(normalize_device_key)

    # Deduplicate specs to guarantee clean 1-to-1 matching
    df_specs_unique = df_specs.drop_duplicates(subset=["match_key"]).copy()

    cls_unique_keys = df_cls["match_key"].unique()
    specs_candidates = df_specs_unique["match_key"].tolist()

    print(
        f"Unique classified entities: {len(cls_unique_keys)} | Unique specs"
        f" cataloged: {len(specs_candidates)}"
    )

    # Key mapping resolution
    mapping = {}
    matched_exact = 0
    matched_fuzzy = 0
    unmatched = 0

    for c_key in cls_unique_keys:
        if not c_key:
            mapping[c_key] = None
            unmatched += 1
            continue

        # 1. Exact normalized match
        if c_key in specs_candidates:
            mapping[c_key] = c_key
            matched_exact += 1
        # 2. Fuzzy match fallback
        elif process is not None:
            match = process.extractOne(
                c_key,
                specs_candidates,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=fuzzy_cutoff,
            )
            if match:
                best_match, score, _ = match
                mapping[c_key] = best_match
                matched_fuzzy += 1
            else:
                mapping[c_key] = None
                unmatched += 1
        else:
            mapping[c_key] = None
            unmatched += 1

    print(
        f"Resolution stats: {matched_exact} exact, {matched_fuzzy} fuzzy (cutoff"
        f" >={fuzzy_cutoff}), {unmatched} unmatched."
    )

    # Merge
    df_cls["spec_key"] = df_cls["match_key"].map(mapping)

    df_merged = df_cls.merge(
        df_specs_unique,
        left_on="spec_key",
        right_on="match_key",
        how="left",
        suffixes=("", "_specs"),
    )

    # Drop temporary merge helper keys
    drop_cols = ["match_key", "spec_key", "match_key_specs"]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # Diagnostics
    total_listings = len(df_merged)
    augmented_count = df_merged["chipset"].notna().sum()
    pct = (augmented_count / total_listings * 100) if total_listings > 0 else 0.0

    print(
        f"Enrichment Complete: {augmented_count}/{total_listings} listings"
        f" ({pct:.1f}%) augmented with hardware specs."
    )

    df_merged.to_csv(output_path, index=False)
    print(f"Saved augmented dataset to: {output_path}")


if __name__ == "__main__":
    merge_pk_classifieds_with_specs()
