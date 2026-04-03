"""
================================================================================
SCRIPT 01 — TARGET SELECTION & CROSS-REFERENCING
================================================================================
Thesis: Thermodynamic Analysis and Modeling of Blackbody Radiation in
        Eclipsing Binary Systems Using Space-Based Photometry

Methodology Step 4.1:
    Queries the Kreiner (2004) eclipsing binary database and cross-references
    candidates against the GCVS and VSX to identify "neglected" short-period
    systems that have been observed by TESS.

Dependencies:
    pip install requests beautifulsoup4 astroquery astropy lightkurve

Outputs:
    neglected_targets.csv  — final list of selected target systems
================================================================================
"""

import requests
import time
import csv
import warnings
from datetime import datetime, timezone

from astroquery.vizier import Vizier
import astropy.units as u
import lightkurve as lk

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TARGET_TYPES        = ["EW", "EB"]
MAX_PERIOD_DAYS     = 1.0
MIN_YEARS_NEGLECTED = 10       # skip systems with timing updates more recent
SAMPLE_SIZE_MAX     = 5
CURRENT_YEAR        = datetime.now(timezone.utc).year

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: QUERY KREINER (2004) DATABASE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Querying Kreiner (2004) eclipsing binary database via VizieR")
print("=" * 70)

Vizier.ROW_LIMIT = 5000
Vizier.columns   = ["**"]

kreiner_catalog = "J/AcA/54/207"

try:
    # 1. Download Database
    result = Vizier.get_catalogs(kreiner_catalog)
    kreiner_table = result[0]
    print(f"  Retrieved {len(kreiner_table)} systems from Kreiner (2004).")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: EXTRACT PERIOD-FILTERED CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

    print("\nSTEP 2: Filtering by orbital period...")
    
    # 2. Filter strictly by period (Kreiner lacks a 'Type' column)
    valid_period_mask = kreiner_table['Per'] < MAX_PERIOD_DAYS
    filtered_table = kreiner_table[valid_period_mask]
    print(f"  Kept {len(filtered_table)} systems with P < {MAX_PERIOD_DAYS} days.")

    # 3. Build candidate list (Type will be verified via GCVS later)
    candidates = [
        {
            "name":   " ".join(str(row["Name"]).split()),
            "type":   "Pending", # GCVS will populate this
            "period": float(row["Per"]),
        }
        for row in filtered_table
    ]
    print(f"  Successfully built list of {len(candidates)} candidates to cross-reference.")

except Exception as e:
    print(f"  Error querying or processing VizieR data: {e}")
    print("  Falling back to demonstration candidates.")
    candidates = [
        {"name": "V523 Cas", "type": "EW", "period": 0.2337},
        {"name": "AW Lac",   "type": "EW", "period": 0.3087},
        {"name": "V776 Cas", "type": "EW", "period": 0.4405},
        {"name": "SW Lac",   "type": "EW", "period": 0.3207},
        {"name": "V1191 Cyg","type": "EW", "period": 0.3133},
    ]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CROSS-REFERENCE HELPERS (GCVS + VSX)
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 3: Preparing GCVS and VSX cross-reference functions...")

gcvs_catalog = "B/gcvs/gcvs_cat"

def query_gcvs(star_name):
    """
    Query the GCVS for a given star name.
    Returns the first matching astropy Row, or None.
    """
    try:
        result = Vizier.query_object(
            star_name, catalog=gcvs_catalog, radius=5 * u.arcsec
        )
        if result and len(result[0]) > 0:
            return result[0][0]
        return None
    except Exception:
        return None

def query_vsx(star_name):
    """
    Query the AAVSO VSX API for a given star name.
    Returns a dict with VSX fields, or None if not found.
    """
    url = (
        "https://www.aavso.org/vsx/index.php"
        f"?view=api.object&format=json&name={requests.utils.quote(star_name)}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            vsx_obj = resp.json().get("VSXObject", None)
            
            if isinstance(vsx_obj, list) and len(vsx_obj) > 0:
                return vsx_obj[0]
            elif isinstance(vsx_obj, dict):
                return vsx_obj
                
        return None
    except Exception:
        return None

def is_neglected(star_name, vsx_data):
    """
    Determine whether a system qualifies as 'neglected'.
    """
    if vsx_data is None:
        return True, "No VSX entry — no recent timing data recorded."

    period_vsx = vsx_data.get("Period", None)
    epoch_str  = vsx_data.get("Epoch",  None)      

    if period_vsx is None and epoch_str is None:
        return True, "VSX has no period or epoch — effectively neglected."

    if epoch_str:
        try:
            epoch_clean = float(str(epoch_str).split("(")[0])
            epoch_year  = 1900 + (epoch_clean - 2415020.31352) / 365.25
            years_since = CURRENT_YEAR - epoch_year
            
            if years_since < MIN_YEARS_NEGLECTED:
                return (False, f"Last epoch ~{epoch_year:.0f} ({years_since:.0f} yr ago) — not neglected.")
            return (True, f"Last epoch ~{epoch_year:.0f} ({years_since:.0f} yr ago) — qualifies as neglected.")
        except (ValueError, TypeError):
            pass  

    return True, "Could not verify recency of timing data — assuming neglected."

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TESS AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 4: Verifying TESS sector coverage via lightkurve MAST query...")

def check_tess_coverage(star_name):
    """
    Search MAST for TESS Target Pixel Files for a given star.
    """
    try:
        search = lk.search_targetpixelfile(star_name, mission="TESS")
        if len(search) > 0:
            sectors = list(set(str(s) for s in search.table["sequence_number"]))
            return search, sectors
        return None, []
    except Exception:
        return None, []

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: APPLY ALL CRITERIA AND BUILD FINAL TARGET LIST
# ─────────────────────────────────────────────────────────────────────────────

print("\nSTEP 5: Applying all selection criteria and building final target list...")

final_targets = []

for candidate in candidates:
    if len(final_targets) >= SAMPLE_SIZE_MAX:
        break

    name   = candidate["name"]
    period = candidate["period"]

    print(f"\n  Evaluating: {name} (P = {period:.4f} d)")

    # --- Criterion (b): TESS coverage ---
    tess_result, sectors = check_tess_coverage(name)
    if tess_result is None:
        print(f"    SKIP — No TESS data found for {name}.")
        time.sleep(0.5)
        continue
    print(f"    TESS sectors available: {', '.join(sectors)}")

    # --- Criterion (a): Neglected status via VSX ---
    vsx_data   = query_vsx(name)
    neglected, reason = is_neglected(name, vsx_data)
    print(f"    VSX check: {reason}")
    if not neglected:
        print(f"    SKIP — System has recent timing data; not neglected.")
        time.sleep(1.0)
        continue

    # --- GCVS cross-reference: confirm variable star classification ---
    gcvs_row = query_gcvs(name)
    if gcvs_row is not None:
        gcvs_type = str(gcvs_row["VarType"]).strip() if "VarType" in gcvs_row.colnames else "?"
        print(f"    GCVS entry confirmed. Reported type: {gcvs_type}")
        
        # Check if the GCVS type matches our target criteria
        if not any(t in gcvs_type for t in TARGET_TYPES):
            print(f"    -> REJECTED: Type '{gcvs_type}' is not in {TARGET_TYPES}.")
            time.sleep(1.0)
            continue
            
        ctype = gcvs_type # Lock in the confirmed true type
        
    else:
        print(f"    GCVS: no entry found within 5\" — skipping as morphological type is unconfirmed.")
        time.sleep(1.0)
        continue

    # Record confirmed target
    final_targets.append({
        "name":    name,
        "type":    ctype,
        "period":  period,
        "sectors": ";".join(sectors),
    })
    print(f"    ✓ SELECTED: {name}")

    time.sleep(1.0)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: OUTPUT FINAL TARGET LIST TO CSV
# ─────────────────────────────────────────────────────────────────────────────

output_file = "neglected_targets.csv"
fieldnames  = ["name", "type", "period", "sectors"]

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_targets)

print("\n" + "=" * 70)
print(f"TARGET SELECTION COMPLETE")
print(f"  {len(final_targets)} systems written to: {output_file}")
print("=" * 70)

for t in final_targets:
    print(f"  {t['name']:<20} Type: {t['type']:<5} "
          f"P = {t['period']:.4f} d   TESS sectors: {t['sectors']}")
