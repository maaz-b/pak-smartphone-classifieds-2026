"""
01_create_normalized_dataset.py
----------------------------
Pipeline to sanitize, de-identify, and normalize raw OLX mobile classifieds.
- Hashes listing IDs with SHA-256 for irreversible anonymization.
- Converts epoch timestamps to ISO UTC datetime and human-readable dates.
- Preserves and rounds spatial coordinates (lat/long) for spatial ML.
- Strips unstructured raw text (titles, descriptions) to prevent PII leakage.
- Enforces strict numeric types and exports a clean benchmark CSV.
"""

import hashlib

import pandas as pd


def hash_ad_id(val, salt: str = "pk_classifieds_2026") -> str:
    """Creates a deterministic, irreversible truncated SHA-256 hash."""
    if pd.isna(val):
        return ""
    raw_str = f"{str(val).strip()}_{salt}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def normalize_classifieds(
    input_path: str = "../data/raw/raw_market_listings.csv",
    output_path: str = "../data/pak_mobile_classifieds_2026.csv",
    salt: str = "pk_classifieds_2026",
):
    print(f"Loading raw classifieds from: {input_path}")
    df = pd.read_csv(input_path)
    total_raw = len(df)

    # 1. Salted Hash for ad_id (De-identification)
    print("Hashing ad_ids...")
    df["ad_id"] = df["ad_id"].apply(lambda x: hash_ad_id(x, salt=salt))

    # 2. Date conversion from created_at_epoch
    print("Converting timestamps...")
    if "created_at_epoch" in df.columns:
        dt_series = pd.to_datetime(
            df["created_at_epoch"], unit="s", errors="coerce", utc=True
        )
        df["listing_date"] = dt_series.dt.strftime("%Y-%m-%d")
        df["listing_datetime_utc"] = dt_series.dt.strftime("%Y-%m-%d %H:%M:%S")

    # 3. Spatial Coords (Preserve & round to ~11m precision to prevent residential pinning)
    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(4)
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(4)

    # 4. Standardize text categories
    df["make"] = df["make"].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.lower().str.strip()
    df["condition"] = df["condition"].astype(str).str.lower().str.strip()
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # 5. Numeric and binary casting
    for col in ["pta_approved", "is_jv", "hardware_repaired"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["storage_gb", "battery_health", "cycle_count", "price_pkr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Filter Outliers (installment down-payments or test posts under Rs 10,000)
    valid_mask = (df["price_pkr"] >= 10000) & (df["price_pkr"] <= 2000000)
    df_clean = df[valid_mask].copy()

    # 7. Select finalized columns (dropping raw_title, raw_description, raw created_at_epoch)
    final_cols = [
        "ad_id",
        "make",
        "model",
        "condition",
        "pta_approved",
        "city",
        "latitude",
        "longitude",
        "storage_gb",
        "battery_health",
        "cycle_count",
        "is_jv",
        "hardware_repaired",
        "listing_date",
        "listing_datetime_utc",
        "price_pkr",
    ]

    available_cols = [c for c in final_cols if c in df_clean.columns]
    df_clean = df_clean[available_cols]

    print(f"Sanitization Complete: Retained {len(df_clean)}/{total_raw} listings.")
    df_clean.to_csv(output_path, index=False)
    print(f"Saved normalized dataset to: {output_path}")


if __name__ == "__main__":
    normalize_classifieds()
