"""03_extract_anonymized_descriptions.py

----------------------------------
Extracts listing titles and descriptions from raw classifieds, applies robust
regex-based de-identification (scrubbing Pakistani mobile numbers, CNICs,
social media handles, and personal names/URLs), links them to the exact
anonymized ad_id hash, and exports an NLP-ready text corpus.
"""

import hashlib
import re

import pandas as pd


def hash_ad_id(val, salt: str = "pk_classifieds_2026") -> str:
    """Creates a deterministic, irreversible truncated SHA-256 hash matching the tabular pipeline."""
    if pd.isna(val):
        return ""
    raw_str = f"{str(val).strip()}_{salt}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def scrub_pii(text: str) -> str:
    """Removes personal identifiable information (PII) common in Pakistani classifieds:

    - Mobile numbers (03xx, +923xx, spaces/dashes variations)
    - CNIC patterns (13 digits with or without dashes)
    - URLs and domains
    - WhatsApp / Social callouts and handles
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    s = text

    # 1. Standardize whitespace
    s = re.sub(r"[\r\t]+", " ", s)

    # 2. Scrub Pakistani phone numbers (+92 3xx, 00923xx, 03xx-xxxxxxx, etc.)
    phone_pattern = r"(?:\+?92|0092|0)?\s*3[\d\s\.\-]{8,14}\d"
    s = re.sub(phone_pattern, "[PHONE_REDACTED]", s)

    # Secondary sweep for 10-12 continuous digit blocks
    s = re.sub(r"\b\d{10,12}\b", "[PHONE_REDACTED]", s)

    # 3. Scrub Pakistani CNIC numbers (13 digits or xxxxx-xxxxxxx-x)
    cnic_pattern = r"\b\d{5}[-–]\d{7}[-–]\d\b"
    s = re.sub(cnic_pattern, "[CNIC_REDACTED]", s)

    # 4. Scrub URLs and Email Addresses
    s = re.sub(r"https?://\S+|www\.\S+", "[URL_REDACTED]", s)
    s = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", s)

    # 5. Scrub WhatsApp / Contact Prompts
    s = re.sub(
        r"(?i)\b(whatsapp|wa\.me|call/sms|contact|rabta)\s*[:\-]?\s*(\[PHONE_REDACTED\]|\S+)?",
        r"\1 [CONTACT_REDACTED]",
        s,
    )

    # 6. Normalize leftover whitespace and consecutive redaction tokens
    s = re.sub(r"(\[PHONE_REDACTED\]\s*)+", "[PHONE_REDACTED] ", s)
    return re.sub(r" +", " ", s).strip()


def build_text_corpus(
    input_path: str = "../data/raw/raw_market_listings.csv",
    output_path: str = "../data/pak_mobile_descriptions_2026.csv",
    salt: str = "pk_classifieds_2026",
):
    print(f"Loading raw classifieds from: {input_path}")
    df = pd.read_csv(input_path)
    total_rows = len(df)

    # 1. Generate the identical deterministic ad_id hash
    print("Generating anonymized ad_id hashes...")
    df["ad_id"] = df["ad_id"].apply(lambda x: hash_ad_id(x, salt=salt))

    # 2. De-identify title and description
    print("De-identifying titles and descriptions...")
    df["clean_title"] = df["raw_title"].fillna("").astype(str).apply(scrub_pii)
    df["clean_description"] = (
        df["raw_description"].fillna("").astype(str).apply(scrub_pii)
    )

    # 3. Drop records with empty text
    df["has_text"] = (df["clean_title"].str.len() > 0) | (
        df["clean_description"].str.len() > 0
    )
    df_corpus = df[df["has_text"]].copy()

    # 4. Select export columns
    export_cols = ["ad_id", "make", "model", "clean_title", "clean_description"]
    available_cols = [c for c in export_cols if c in df_corpus.columns]
    df_corpus = df_corpus[available_cols]

    print(
        f"Extraction Complete: Exporting {len(df_corpus)}/{total_rows} sanitized"
        " text records."
    )
    df_corpus.to_csv(output_path, index=False)
    print(f"Saved anonymized descriptions to: {output_path}")


if __name__ == "__main__":
    build_text_corpus()
