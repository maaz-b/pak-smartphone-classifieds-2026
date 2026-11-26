"""sample_olx_scraper.py

---------------------
DISCLAIMER:
Automated requests should comply with platform policies and rate limits.
This sample collector is provided strictly for academic reproducibility and
methodological transparency. Do not hammer public endpoints; respect reasonable
delays and rate boundaries.
"""

import re
import sys
import time

import pandas as pd
import requests


# -------------------------------------------------------------------------
# Regex Pattern Matchers for Secondary Market Listing Text
# -------------------------------------------------------------------------
def extract_battery_health(text: str):
    """Matches battery health percentages (e.g.

    'bh 88%', 'battery 100', 'health 93%').
    """
    if not isinstance(text, str):
        return None
    match = re.search(
        r"(?:battery\s*health|battery|bh)\s*[:=]?\s*(\d{2,3})\s*%?", text, re.IGNORECASE
    )
    if match:
        val = float(match.group(1))
        if 50.0 <= val <= 100.0:
            return val
    return None


def extract_cycle_count(text: str):
    """Matches battery cycle counts (e.g. 'cycle count 140', 'cc 427')."""
    if not isinstance(text, str):
        return None
    match = re.search(
        r"(?:cycle\s*count|cycles?|cc)\s*[:=]?\s*(\d{1,4})\b", text, re.IGNORECASE
    )
    if match:
        val = float(match.group(1))
        if val <= 2500:
            return val
    return None


def detect_jv_status(text: str) -> int:
    """Detects carrier-locked / factory-locked / JV phones (e.g., 'jv', 'joint venture', 'gevey')."""
    if not isinstance(text, str):
        return 0
    if re.search(r"\b(jv|joint\s*venture|gevey|chip\s*sim)\b", text, re.IGNORECASE):
        return 1
    return 0


def detect_hardware_repair(text: str) -> int:
    """Detects screen replacement, bypass, or body repair disclosures."""
    if not isinstance(text, str):
        return 0
    pattern = r"\b(panel\s*changed|screen\s*change|glass\s*change|repaired|bypass|face\s*id\s*off|truetone\s*off)\b"
    if re.search(pattern, text, re.IGNORECASE):
        return 1
    return 0


# -------------------------------------------------------------------------
# Sample Polling Engine
# -------------------------------------------------------------------------
def fetch_sample_batch(
    page_limit: int = 1,
    category_id: int = 830,
    output_path: str = "data/raw/olx_scraped_sample.csv",
):
    """Fetches a small sample batch of mobile phone listings.

    Demonstrates data ingestion, parameter unnesting, and regex feature
    extraction.
    """
    # NOTE: Replace with your actual authorized ingestion endpoint or gateway.
    # A dummy placeholder endpoint is used here to prevent exposing internal production APIs.
    base_url = "https://api.example-marketplace.com/relevance/v4/search"

    headers = {
        "User-Agent": (
            "AcademicResearchBot/1.0"
            " (+https://github.com/syed-maaz-bukhari/pak-mobile-classifieds-2026;"
            " academic-dataset-curation)"
        ),
        "Accept": "application/json",
    }

    records = []

    print(
        f"Polling {page_limit} page(s) from marketplace endpoint (Category:"
        f" {category_id})..."
    )

    for page in range(page_limit):
        params = {
            "category": category_id,
            "facet_limit": 100,
            "page": page,
            "split": "iso",
        }

        try:
            time.sleep(1.5)

            response = requests.get(
                base_url, headers=headers, params=params, timeout=12
            )
            if response.status_code != 200:
                print(
                    f"Request terminated (HTTP {response.status_code}). If using"
                    " placeholder endpoint, mock data or configure actual gateway."
                )
                break

            payload = response.json()
            data_items = payload.get("data", [])

            if not data_items:
                print(
                    "No data items returned. Reached end of results or empty payload."
                )
                break

            for item in data_items:
                ad_id = item.get("id")
                raw_title = item.get("title", "")
                raw_desc = item.get("description", "")
                created_at_epoch = item.get("created_at")

                location = item.get("locations_resolved", {})
                city = (
                    location.get("ADMIN_LEVEL_3_name")
                    or location.get("SUBLOCALITY_LEVEL_1_name")
                    or location.get("COUNTRY_name")
                )
                lat = location.get("point", {}).get("latitude")
                lon = location.get("point", {}).get("longitude")

                price_dict = item.get("price", {})
                price_val = price_dict.get("value", {}).get("raw")
                if price_val is None:
                    continue

                params_list = item.get("parameters", [])
                param_dict = {
                    p.get("key"): p.get("value_name") for p in params_list if "key" in p
                }

                make = param_dict.get("make") or ""
                model = param_dict.get("model") or ""
                condition = param_dict.get("condition") or "used"

                pta_raw = str(param_dict.get("pta_status", "")).lower()
                pta_approved = (
                    1 if "approved" in pta_raw or "registered" in pta_raw else 0
                )

                storage_raw = param_dict.get("storage") or ""
                storage_match = re.search(
                    r"(\d+)\s*(?:gb|tb)", f"{storage_raw} {raw_title}", re.IGNORECASE
                )
                storage_gb = None
                if storage_match:
                    storage_val = float(storage_match.group(1))
                    storage_gb = storage_val * 1024 if storage_val <= 2 else storage_val

                full_text = f"{raw_title} {raw_desc}"
                battery_health = extract_battery_health(full_text)
                cycle_count = extract_cycle_count(full_text)
                is_jv = detect_jv_status(full_text)
                hardware_repaired = detect_hardware_repair(full_text)

                records.append(
                    {
                        "ad_id": ad_id,
                        "make": make,
                        "model": model,
                        "condition": condition,
                        "pta_approved": pta_approved,
                        "city": city,
                        "latitude": lat,
                        "longitude": lon,
                        "storage_gb": storage_gb,
                        "battery_health": battery_health,
                        "cycle_count": cycle_count,
                        "is_jv": is_jv,
                        "hardware_repaired": hardware_repaired,
                        "raw_title": raw_title,
                        "raw_description": raw_desc,
                        "created_at_epoch": created_at_epoch,
                        "price_pkr": float(price_val),
                    }
                )

        except Exception as e:
            print(f"Connection notice for page {page}: {e}")
            break

    df_out = pd.DataFrame(records)
    print(
        f"Sample Collection Complete: Extracted {len(df_out)} listings across"
        f" {page_limit} page(s)."
    )

    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
        print(f"Saved sample raw data to: {output_path}")

    return df_out


if __name__ == "__main__":
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out_file = sys.argv[2] if len(sys.argv) > 2 else "data/raw/olx_scraped_sample.csv"
    fetch_sample_batch(page_limit=pages, output_path=out_file)
