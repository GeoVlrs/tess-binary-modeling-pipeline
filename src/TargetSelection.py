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

    A system is defined as "neglected" if ALL of the following are true:
      (a) Its most recent timing record in VSX is older than MIN_YEARS_NEGLECTED
      (b) TESS has observed it in at least one sector
      (c) GCVS confirms its morphological type as EW (W UMa) or EB (Algol)

Dependencies:
    pip install requests beautifulsoup4 astroquery astropy lightkurve

Outputs:
    neglected_targets.csv   — final list of SELECTED target systems
    rejected_targets.csv    — full log of all REJECTED candidates with reasons
                              (intended for inclusion as a thesis addendum)
================================================================================
"""

# ── Standard library imports ──────────────────────────────────────────────────
import requests       # HTTP requests to the AAVSO VSX API
import time           # sleep() calls between API requests (polite rate limiting)
import csv            # reading and writing CSV files
import warnings       # suppress noisy astroquery/astropy deprecation warnings
from datetime import datetime, timezone   # for computing "years since last epoch"

# ── Third-party astronomy library imports ─────────────────────────────────────
from astroquery.vizier import Vizier   # programmatic access to the VizieR catalogue service
import astropy.units as u              # unit handling for coordinate queries
import lightkurve as lk               # MAST archive interface for TESS data

# Suppress all non-critical warnings so terminal output stays readable
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# All parameters that control the selection logic are defined here in one place.
# Adjust these values to change the stringency of the target selection criteria.

# Morphological types that qualify for inclusion.
# "EW" = W Ursae Majoris (contact) type.
# "EB" = Algol / beta-Lyrae (semi-detached or detached) type.
# These are the standard GCVS type codes.
TARGET_TYPES = ["EW", "EB"]

# Upper bound on orbital period (days).
# Short-period systems are the focus of this thesis; longer-period systems
# are excluded because their eclipse timings tend to be better maintained.
MAX_PERIOD_DAYS = 1.0

# A system is considered "neglected" if its most recently recorded timing
# is older than this many years. Lowering this value increases the pool
# of candidates; raising it makes the selection more conservative.
MIN_YEARS_NEGLECTED = 10

# Maximum number of systems to include in the final sample.
# The thesis methodology calls for a controlled sample of 3-5 systems.
SAMPLE_SIZE_MAX = 5

# Capture the current calendar year once at startup.
# This is used to calculate how many years have elapsed since a system's
# last recorded timing measurement.
CURRENT_YEAR = datetime.now(timezone.utc).year


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: QUERY KREINER (2004) DATABASE
# ─────────────────────────────────────────────────────────────────────────────
# The Kreiner (2004) catalogue (Acta Astronomica, 54, 207) provides up-to-date
# linear elements (reference epoch T0 and period P) for a large sample of
# eclipsing binaries. It is hosted on VizieR under the identifier J/AcA/54/207.
#
# We retrieve the full catalogue and then filter it locally by period,
# rather than using VizieR's server-side filtering, to avoid truncation issues.

print("=" * 70)
print("STEP 1: Querying Kreiner (2004) eclipsing binary database via VizieR")
print("=" * 70)

# Tell VizieR to return up to 5000 rows (the catalogue has ~3000 entries)
# and to include all available columns ("**" is VizieR's wildcard).
Vizier.ROW_LIMIT = 5000
Vizier.columns   = ["**"]

# VizieR catalogue identifier for Kreiner (2004)
kreiner_catalog = "J/AcA/54/207"

try:
    # Download the full Kreiner catalogue from VizieR.
    # result is an astropy TableList; we want the first (and only) table.
    result        = Vizier.get_catalogs(kreiner_catalog)
    kreiner_table = result[0]
    print(f"  Retrieved {len(kreiner_table)} systems from Kreiner (2004).")


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: EXTRACT PERIOD-FILTERED CANDIDATES
    # ─────────────────────────────────────────────────────────────────────────
    # The Kreiner catalogue does not include a morphological type column;
    # that information must be retrieved separately from the GCVS (Section 4).
    # Here we only filter by period, which is always available.

    print("\nSTEP 2: Filtering by orbital period...")

    # Boolean mask: True for rows where the orbital period is within our limit.
    # kreiner_table['Per'] contains the period in days as a float column.
    valid_period_mask = kreiner_table["Per"] < MAX_PERIOD_DAYS
    filtered_table    = kreiner_table[valid_period_mask]
    print(f"  Kept {len(filtered_table)} systems with P < {MAX_PERIOD_DAYS} days.")

    # Build a plain list of dicts for easier handling in the loop below.
    # The 'type' field is populated later from the GCVS; "Pending" is a
    # placeholder that is replaced or causes rejection if GCVS lookup fails.
    candidates = [
        {
            "name":   " ".join(str(row["Name"]).split()),   # normalise whitespace
            "type":   "Pending",                             # filled in by GCVS
            "period": float(row["Per"]),
        }
        for row in filtered_table
    ]
    print(f"  Built candidate list of {len(candidates)} systems for cross-referencing.")

except Exception as e:
    # If VizieR is unreachable or the catalogue format has changed,
    # fall back to a small hardcoded demonstration list so the rest
    # of the script can still be tested offline.
    print(f"  Error querying VizieR: {e}")
    print("  Falling back to demonstration candidate list.")
    candidates = [
        {"name": "V523 Cas",  "type": "EW", "period": 0.2337},
        {"name": "AW Lac",    "type": "EW", "period": 0.3087},
        {"name": "V776 Cas",  "type": "EW", "period": 0.4405},
        {"name": "SW Lac",    "type": "EW", "period": 0.3207},
        {"name": "V1191 Cyg", "type": "EW", "period": 0.3133},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CROSS-REFERENCE HELPER FUNCTIONS (GCVS + VSX)
# ─────────────────────────────────────────────────────────────────────────────
# Three helper functions are defined here:
#   query_gcvs()   — looks up the star in the General Catalogue of Variable Stars
#   query_vsx()    — looks up the star in the AAVSO International Variable Star Index
#   is_neglected() — applies the neglected-status logic to the VSX result

print("\nSTEP 3: Preparing GCVS and VSX cross-reference functions...")

# VizieR identifier for the GCVS (Samus et al. 2017, version 5.1)
gcvs_catalog = "B/gcvs/gcvs_cat"


def query_gcvs(star_name):
    """
    Look up a star in the GCVS via VizieR and return its catalogue entry.

    The query performs a cone search (radius = 5 arcseconds) centred on
    the coordinates resolved from the star's name. This narrow radius
    prevents accidental matches with nearby unrelated variables.

    Parameters
    ----------
    star_name : str
        The common variable star designation (e.g. "V523 Cas").

    Returns
    -------
    astropy Row or None
        The first matching row from the GCVS table, or None if no match
        is found within 5 arcseconds.
    """
    try:
        # Vizier.query_object() resolves the name to sky coordinates via Simbad,
        # then performs a positional cross-match against the GCVS table.
        result = Vizier.query_object(
            star_name,
            catalog=gcvs_catalog,
            radius=5 * u.arcsec    # 5-arcsecond search cone
        )
        # result is a TableList; check whether the GCVS table is non-empty.
        if result and len(result[0]) > 0:
            return result[0][0]    # return the first (closest) match
        return None
    except Exception:
        # Network errors, name-resolution failures, etc. — treat as no match.
        return None


def query_vsx(star_name):
    """
    Query the AAVSO Variable Star Index (VSX) API by star name.

    The VSX API returns JSON data including the star's period, reference
    epoch, and variability type. We use the epoch to assess how recently
    the star's timing has been updated, which is the primary indicator
    of "neglected" status.

    Parameters
    ----------
    star_name : str
        The common variable star designation (e.g. "V523 Cas").

    Returns
    -------
    dict or None
        A dictionary of VSX fields for the star, or None if not found.
    """
    # Construct the VSX REST API URL. The name is URL-encoded to handle
    # spaces and special characters in variable star designations.
    url = (
        "https://www.aavso.org/vsx/index.php"
        f"?view=api.object&format=json&name={requests.utils.quote(star_name)}"
    )
    try:
        resp = requests.get(url, timeout=10)   # 10-second timeout
        if resp.status_code == 200:
            vsx_obj = resp.json().get("VSXObject", None)

            # The VSX API sometimes returns a list (multiple matches) and
            # sometimes a dict (single match). Handle both cases.
            if isinstance(vsx_obj, list) and len(vsx_obj) > 0:
                return vsx_obj[0]   # take the first (best-name) match
            elif isinstance(vsx_obj, dict):
                return vsx_obj

        return None
    except Exception:
        # Any connection error or JSON parsing failure — treat as no match.
        return None


def is_neglected(star_name, vsx_data):
    """
    Determine whether a star qualifies as "neglected" based on its VSX entry.

    A star is neglected if its most recent timing record (epoch) in VSX
    is older than MIN_YEARS_NEGLECTED years, or if no epoch is recorded
    at all (implying the star has never been precisely timed).

    The epoch in VSX is stored as a Julian Date (JD). We convert it to
    a calendar year using the standard JD-to-year formula:
        year ≈ 1900 + (JD - 2415020.31352) / 365.25
    where 2415020.31352 is JD at J1900.0.

    Parameters
    ----------
    star_name : str
        Used only for logging purposes.
    vsx_data : dict or None
        The dictionary returned by query_vsx(), or None if VSX has no entry.

    Returns
    -------
    is_neglected : bool
        True if the star qualifies as neglected.
    reason : str
        A human-readable explanation of the decision, written for inclusion
        in the rejection/acceptance log.
    """
    # Case 1: No VSX entry at all.
    # If the star is not in VSX, it has no recorded modern timing data
    # by definition, so it is maximally neglected.
    if vsx_data is None:
        return True, "No VSX entry found — no modern timing data on record."

    # Extract the period and epoch fields from the VSX response dict.
    # Both fields may be absent (None) for poorly documented systems.
    period_vsx = vsx_data.get("Period", None)
    epoch_str  = vsx_data.get("Epoch",  None)

    # Case 2: VSX has an entry but records neither a period nor an epoch.
    # This means no precision timing has ever been published to VSX.
    if period_vsx is None and epoch_str is None:
        return True, "VSX entry exists but contains no period or epoch — effectively neglected."

    # Case 3: An epoch is recorded. Parse it and compute the elapsed time.
    if epoch_str:
        try:
            # VSX sometimes appends an uncertainty in parentheses, e.g. "2459001.3(5)".
            # Split on "(" and take the numeric part before the bracket.
            epoch_clean = float(str(epoch_str).split("(")[0])

            # Convert from Julian Date to approximate calendar year.
            # 2415020.31352 = JD at the start of the Besselian epoch B1900.0
            epoch_year  = 1900 + (epoch_clean - 2415020.31352) / 365.25
            years_since = CURRENT_YEAR - epoch_year

            if years_since < MIN_YEARS_NEGLECTED:
                # The system has been updated recently — it does NOT qualify.
                return (
                    False,
                    f"Last recorded epoch ~{epoch_year:.0f} "
                    f"({years_since:.0f} years ago) — within the "
                    f"{MIN_YEARS_NEGLECTED}-year threshold; not neglected."
                )
            else:
                # The system's timing is outdated — it qualifies as neglected.
                return (
                    True,
                    f"Last recorded epoch ~{epoch_year:.0f} "
                    f"({years_since:.0f} years ago) — exceeds the "
                    f"{MIN_YEARS_NEGLECTED}-year threshold; qualifies as neglected."
                )

        except (ValueError, TypeError):
            # The epoch string could not be parsed as a number.
            # Treat conservatively as neglected rather than silently dropping.
            pass

    # Case 4: VSX has an entry and a period, but no parseable epoch.
    # Without an epoch we cannot assess recency, so we treat as neglected.
    return True, "VSX epoch could not be parsed — recency of timing unverifiable; assuming neglected."


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TESS AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
# For each candidate that passes the GCVS and VSX checks, we verify that
# TESS has actually observed it. Without TESS data, the thesis pipeline
# (Scripts 02-05) cannot proceed. lightkurve.search_targetpixelfile()
# queries the MAST archive by name and returns a SearchResult object.

print("\nSTEP 4: Verifying TESS sector coverage via lightkurve MAST query...")


def check_tess_coverage(star_name):
    """
    Search the MAST archive for TESS Target Pixel Files (TPFs) for a star.

    lightkurve resolves the star name to sky coordinates via Simbad and
    then queries the MAST TESS archive for all available TPF products.
    If data exists, the sector numbers are extracted from the search table.

    Parameters
    ----------
    star_name : str
        The common variable star designation.

    Returns
    -------
    search_result : lightkurve.SearchResult or None
        The full search result object (passed to Script 02 for download),
        or None if no TESS data is available.
    sectors : list of str
        The TESS sector numbers in which the star was observed,
        as a deduplicated list of strings (e.g. ["14", "40", "41"]).
    """
    try:
        # Query MAST for all TESS TPFs associated with this star name.
        # mission="TESS" restricts results to TESS (excludes Kepler/K2).
        search = lk.search_targetpixelfile(star_name, mission="TESS")

        if len(search) > 0:
            # Extract unique sector numbers from the search result table.
            # "sequence_number" is the TESS sector index in the MAST metadata.
            sectors = list(set(str(s) for s in search.table["sequence_number"]))
            return search, sectors

        # No results found in MAST for this target.
        return None, []

    except Exception:
        # Name resolution failure, MAST timeout, etc. — treat as no data.
        return None, []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: APPLY ALL CRITERIA AND BUILD FINAL TARGET LIST
# ─────────────────────────────────────────────────────────────────────────────
# This is the main selection loop. For each candidate from Section 3, we apply
# three criteria in order:
#
#   Criterion (b) — TESS coverage      : must have >= 1 TESS sector
#   Criterion (a) — Neglected status   : last epoch must be >= MIN_YEARS_NEGLECTED ago
#   Criterion (c) — Morphological type : GCVS type must be EW or EB
#
# Any candidate that fails a criterion is immediately rejected. The rejection
# is logged with a specific reason string for inclusion in the thesis addendum.

print("\nSTEP 5: Applying all selection criteria and building final target list...")

# Accumulates all systems that pass every criterion.
final_targets = []

# Accumulates every rejected system with its rejection reason.
# This list is written to rejected_targets.csv at the end of the script.
rejected_candidates = []


for candidate in candidates:

    # Stop once we have reached the desired sample size.
    if len(final_targets) >= SAMPLE_SIZE_MAX:
        break

    name   = candidate["name"]
    period = candidate["period"]

    print(f"\n  Evaluating: {name}  (P = {period:.4f} d)")

    # ── CRITERION (b): TESS SECTOR COVERAGE ──────────────────────────────────
    # Checked first because it requires a remote API call that is fast to fail.
    # If no TESS data exists, there is no point checking the other criteria.

    tess_result, sectors = check_tess_coverage(name)

    if tess_result is None:
        # Construct a precise rejection reason for the log.
        rejection_reason = (
            "REJECTED — Criterion (b) TESS coverage: "
            "No TESS Target Pixel Files found in the MAST archive. "
            "Without TESS data the computational pipeline cannot proceed."
        )
        print(f"    x {rejection_reason}")
        rejected_candidates.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (b) — TESS coverage",
            "rejection_reason": rejection_reason,
        })
        time.sleep(0.5)
        continue   # move on to the next candidate

    print(f"    v Criterion (b) passed — TESS sectors: {', '.join(sectors)}")

    # ── CRITERION (a): NEGLECTED STATUS VIA VSX ───────────────────────────────
    # Query the AAVSO VSX API for timing recency information.
    # If the system has been recently updated (within MIN_YEARS_NEGLECTED years),
    # it does not qualify as neglected and is excluded from the study.

    vsx_data = query_vsx(name)
    neglected, neglect_reason = is_neglected(name, vsx_data)
    print(f"    VSX assessment: {neglect_reason}")

    if not neglected:
        rejection_reason = (
            f"REJECTED — Criterion (a) neglected status: {neglect_reason} "
            f"Only systems with no timing update in >={MIN_YEARS_NEGLECTED} years qualify."
        )
        print(f"    x {rejection_reason}")
        rejected_candidates.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (a) — Neglected status",
            "rejection_reason": rejection_reason,
        })
        time.sleep(1.0)
        continue   # move on to the next candidate

    print(f"    v Criterion (a) passed — system qualifies as neglected.")

    # ── CRITERION (c): MORPHOLOGICAL TYPE VIA GCVS ───────────────────────────
    # Query the GCVS to obtain the official variability type classification.
    # Only EW (W UMa / contact) and EB (Algol / semi-detached) types are
    # relevant to the short-period eclipsing binary science of this thesis.

    gcvs_row = query_gcvs(name)

    if gcvs_row is None:
        # The star is not in the GCVS within 5 arcseconds.
        # Without an official morphological classification we cannot confirm
        # the system type, so it is excluded conservatively.
        rejection_reason = (
            "REJECTED — Criterion (c) morphological type: "
            "No GCVS entry found within 5 arcseconds of the resolved coordinates. "
            "Morphological classification is unconfirmed; system excluded conservatively."
        )
        print(f"    x {rejection_reason}")
        rejected_candidates.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (c) — GCVS morphological type",
            "rejection_reason": rejection_reason,
        })
        time.sleep(1.0)
        continue

    # Extract the variability type string from the GCVS row.
    gcvs_type = (
        str(gcvs_row["VarType"]).strip()
        if "VarType" in gcvs_row.colnames
        else "UNKNOWN"
    )
    print(f"    GCVS entry found — reported type: '{gcvs_type}'")

    # Check whether the GCVS type contains any of our accepted type codes.
    # The 'any(...in...)' construction handles composite types like "EW+EA".
    if not any(t in gcvs_type for t in TARGET_TYPES):
        rejection_reason = (
            f"REJECTED — Criterion (c) morphological type: "
            f"GCVS reports type '{gcvs_type}', which is not in the accepted "
            f"type list {TARGET_TYPES}. Only short-period contact (EW) or "
            f"semi-detached (EB) systems are included in this study."
        )
        print(f"    x {rejection_reason}")
        rejected_candidates.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (c) — GCVS morphological type",
            "rejection_reason": rejection_reason,
        })
        time.sleep(1.0)
        continue

    print(f"    v Criterion (c) passed — GCVS type '{gcvs_type}' is accepted.")

    # ── ALL CRITERIA PASSED: ADD TO FINAL SELECTION ───────────────────────────
    # The confirmed GCVS type overwrites the "Pending" placeholder.
    final_targets.append({
        "name":    name,
        "type":    gcvs_type,
        "period":  period,
        "sectors": ";".join(sectors),
    })
    print(f"    vv SELECTED: {name} — all three criteria satisfied.")

    # Polite delay between iterations to avoid hammering remote APIs.
    time.sleep(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: WRITE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
# Two CSV files are written:
#
#   neglected_targets.csv  — the selected systems, passed to Script 02.
#
#   rejected_targets.csv   — a complete, auditable log of every rejected
#                            candidate with the exact criterion that failed
#                            and a plain-English reason. This file is intended
#                            for inclusion as an addendum to the thesis,
#                            demonstrating that the selection process was
#                            systematic and transparent.

# ── Write selected targets ────────────────────────────────────────────────────

selected_file = "neglected_targets.csv"
with open(selected_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "type", "period", "sectors"])
    writer.writeheader()
    writer.writerows(final_targets)

# ── Write rejection log ───────────────────────────────────────────────────────

rejected_file = "rejected_targets.csv"
with open(rejected_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["name", "period_days", "rejection_stage", "rejection_reason"]
    )
    writer.writeheader()
    writer.writerows(rejected_candidates)

# ── Terminal summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("TARGET SELECTION COMPLETE")
print("=" * 70)

print(f"\n  SELECTED ({len(final_targets)} systems) -> {selected_file}")
for t in final_targets:
    print(f"    {t['name']:<20} Type: {t['type']:<10} "
          f"P = {t['period']:.4f} d   Sectors: {t['sectors']}")

print(f"\n  REJECTED ({len(rejected_candidates)} systems) -> {rejected_file}")
for r in rejected_candidates:
    # Print only the stage label in the terminal summary;
    # the full reason is available in the CSV for the thesis addendum.
    print(f"    {r['name']:<20} Failed: {r['rejection_stage']}")

print("\n  The rejection log is formatted for direct inclusion as a")
print("  thesis addendum documenting the transparency of target selection.")
print("=" * 70)