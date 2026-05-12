"""
================================================================================
SCRIPT 03 — ORBITAL TIMING & O–C ANALYSIS
================================================================================
Thesis: Thermodynamic Analysis and Modeling of Blackbody Radiation in
        Eclipsing Binary Systems Using Space-Based Photometry

Methodology Step 4.3:
    Determines precise modern Times of Minimum (ToM) using the Kwee–van Woerden 
    algorithm. Refines orbital ephemerides by calculating secular drift 
    relative to historical baselines (Kreiner 2004). Implements a three-panel 
    visualization to resolve 20-year scale compression in O–C residuals.

Dependencies:
    pip install lightkurve numpy scipy matplotlib astropy astroquery

Inputs:
    benchmark_targets.csv — Curated target list.
    lc_<star>_clean.fits  — Detrended light curves from Script 02.

Outputs:
    tominima_<star>.csv   — Database of measured eclipse times.
    ephem_<star>.txt      — Refined T0 and Period (Orbital Anchor for Script 04).
    plots/oc_<star>.png   — Enhanced three-panel O–C diagram.
================================================================================
"""

# ── Standard library imports ──────────────────────────────────────────────────
import csv
import os
import warnings

# ── Third-party scientific library imports ────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize  import curve_fit
from scipy.signal    import argrelmin
from scipy.ndimage   import uniform_filter1d
import lightkurve as lk
from astroquery.vizier import Vizier

# Suppress non-critical runtime warnings
warnings.filterwarnings("ignore")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCALE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TESS_BTJD_OFFSET = 2457000.0  # BTJD (Mission Time) → BJD (Universal Time)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: KWEE–VAN WOERDEN (KvW) CORE
# ─────────────────────────────────────────────────────────────────────────────

def kwee_van_woerden(time, flux, t_trial, half_window=0.05):
    """
    Calculates the ToM through geometric mirror-branch minimization.
    Identifies the nadir of the eclipse by minimizing the variance (RSS) 
    between the ascending and descending branches without an assumed profile.
    """
    mask  = (time >= t_trial - half_window) & (time <= t_trial + half_window)
    t_win = time[mask]
    f_win = flux[mask]

    if len(t_win) < 6:
        raise ValueError("Insufficient data points within eclipse window for KvW.")

    t_grid  = np.linspace(t_win.min(), t_win.max(), 500)
    rss_arr = np.zeros(len(t_grid))

    for idx, t_test in enumerate(t_grid):
        dt    = t_win - t_test
        pos   = dt >= 0
        neg   = dt <  0
        dt_p  = dt[pos]
        dt_n  = -dt[neg]
        f_p   = f_win[pos]
        f_n   = f_win[neg]

        if len(dt_p) < 2 or len(dt_n) < 2:
            rss_arr[idx] = np.inf
            continue

        f_n_interp = np.interp(dt_p, dt_n[::-1], f_n[::-1], left=np.nan, right=np.nan)
        valid = np.isfinite(f_n_interp)
        if valid.sum() < 2:
            rss_arr[idx] = np.inf
            continue

        rss_arr[idx] = np.sum((f_p[valid] - f_n_interp[valid]) ** 2)

    best_idx = np.argmin(rss_arr)
    t_min    = t_grid[best_idx]

    # ── Classical KvW uncertainty: fit a parabola to the RSS trough ──────────
    # The curvature of the parabola at its minimum directly translates to the 
    # formal uncertainty of the measurement.
    lo = max(best_idx - 15, 0)
    hi = min(best_idx + 16, len(t_grid))
    finite_mask = np.isfinite(rss_arr[lo:hi])

    t_err = 1e-4   # fallback if parabola fit fails
    if finite_mask.sum() >= 3:
        try:
            parabola_coeffs = np.polyfit(
                t_grid[lo:hi][finite_mask],
                rss_arr[lo:hi][finite_mask],
                2
            )
            a_curv = parabola_coeffs[0]   
            if a_curv > 0:
                n_pairs = max(1, finite_mask.sum())
                t_err   = np.sqrt(rss_arr[best_idx] / (n_pairs * a_curv))
        except np.linalg.LinAlgError:
            pass

    return t_min, t_err

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: POLYNOMIAL MINIMUM FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def polynomial_minimum(time, flux, t_trial, half_window=0.05, deg=4):
    """
    Fallback method: Fit a polynomial to the eclipse bottom and return its 
    mathematical derivative minimum as the ToM.
    """
    mask  = (time >= t_trial - half_window) & (time <= t_trial + half_window)
    t_win = time[mask]
    f_win = flux[mask]

    if len(t_win) < deg + 2:
        raise ValueError("Insufficient data for polynomial fit.")

    t_c    = t_win - t_trial
    coeffs = np.polyfit(t_c, f_win, deg)
    poly   = np.poly1d(coeffs)

    t_fine = np.linspace(t_c.min(), t_c.max(), 10000)
    f_fine = poly(t_fine)
    t_min  = t_trial + t_fine[np.argmin(f_fine)]

    residuals = f_win - poly(t_c)
    rms       = np.std(residuals)
    deriv2    = np.polyder(poly, 2)
    curv      = abs(float(deriv2(t_fine[np.argmin(f_fine)])))
    t_err     = rms / curv if curv > 0 else 1e-4

    return t_min, t_err

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: O–C CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_oc(t_obs_bjd, t_obs_err, T0_hjd, P_kreiner):
    """
    Compute O–C (Observed minus Calculated) residuals against the Kreiner (2004) 
    historical baseline ephemeris.
    """
    # [IMPROVEMENT] Cycle-Count Validation (Preventing Cycle Jumps)
    e_calc = (t_obs_bjd - T0_hjd) / P_kreiner
    epochs = np.round(e_calc).astype(int)
    
    # Check for potential cycle jump (residual > 0.25 phase)
    for i in range(len(epochs)):
        res = e_calc[i] - epochs[i]
        if abs(res) > 0.25:
            e_test = np.array([epochs[i]-1, epochs[i], epochs[i]+1])
            res_test = np.abs(e_calc[i] - e_test)
            epochs[i] = e_test[np.argmin(res_test)]

    t_calc = T0_hjd + P_kreiner * epochs
    oc     = t_obs_bjd - t_calc
    oc_err = t_obs_err
    return epochs, oc, oc_err

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: REFINED LINEAR EPHEMERIS
# ─────────────────────────────────────────────────────────────────────────────

def refine_ephemeris(epochs, t_obs_bjd, t_obs_err):
    """
    Fit T_obs = T0_new + P_new × E via weighted least squares to calculate 
    the updated reference epoch and orbital period.
    """
    def linear(E, T0, P):
        return T0 + P * E

    # FIX: Use polyfit to generate a mathematically sound initial guess
    # regardless of how large the epoch 'E' has become since the historical baseline.
    coeffs = np.polyfit(epochs, t_obs_bjd, 1)
    p0 = [coeffs[1], coeffs[0]]  # [Intercept (T0), Slope (P)]

    try:
        popt, pcov = curve_fit(
            linear, epochs, t_obs_bjd,
            p0=p0, sigma=t_obs_err, absolute_sigma=True
        )
        perr = np.sqrt(np.diag(pcov))
        return popt[0], popt[1], perr[0], perr[1]
    except RuntimeError:
        return coeffs[1], coeffs[0], np.nan, np.nan

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: KREINER (2004) DYNAMIC FETCH
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Fetching historical baseline ephemerides from Kreiner (2004)...")
KREINER_EPHEMERIDES = {}
try:
    Vizier.ROW_LIMIT = 5000
    result = Vizier.get_catalogs("J/AcA/54/207")
    kreiner_table = result[0]
    for row in kreiner_table:
        # Match formatting from Script 01 to remove internal spaces
        star_name = " ".join(str(row["Name"]).split())
        KREINER_EPHEMERIDES[star_name] = (float(row["M0"]), float(row["Per"]))
    print(f"  Successfully loaded {len(KREINER_EPHEMERIDES)} historical ephemerides.")
except Exception as e:
    print(f"  ERROR fetching Kreiner database: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

targets = []
try:
    with open("neglected_targets.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append({"name": row["name"], "period": float(row["period"])})
except FileNotFoundError:
    print("ERROR: benchmark_targets.csv not found.")
    exit()

for target in targets:
    name   = target["name"]
    period = target["period"]
    safe   = name.replace(" ", "_")

    print("\n" + "=" * 70)
    print(f"TIME-SERIES ANALYSIS: {name}")
    print("=" * 70)

    lc_file = f"lc_{safe}_clean.fits"
    if not os.path.exists(lc_file):
        print(f"  WARNING: {lc_file} not found. Skipping.")
        continue

    lc   = lk.read(lc_file)
    time = lc.time.value    # Array is originally BTJD
    flux = lc.flux.value

    # Convert BTJD → standard BJD to align with historical O-C analysis
    time_bjd = time + TESS_BTJD_OFFSET

    print(f"  Loaded {len(lc)} data points.")
    print(f"  Time range: BTJD {time.min():.2f} – {time.max():.2f}  "
          f"(BJD {time_bjd.min():.2f} – {time_bjd.max():.2f})")

    # ── 6A: IDENTIFY ECLIPSE MINIMA ────────────────────────────────────────
    print("\n  [6A] Identifying eclipse minima candidates...")

    # Uniform smooth mitigates localized noise to prevent false-positive minima detection
    flux_smooth = uniform_filter1d(flux, size=15)
    
    # FIX: Dynamically calculate the localized search order based on actual time array
    cadence_days = np.median(np.diff(time))
    points_per_period = period / cadence_days
    order = max(5, int(points_per_period * 0.2)) # Local minima must be at least 20% of a period apart

    min_indices = argrelmin(flux_smooth, order=order)[0]

    print(f"  Found {len(min_indices)} candidate minima (order = {order})")

    if len(min_indices) == 0:
        print("  ERROR: No minima detected. Check light curve quality.")
        continue

    # ── 6B: PRECISE ToM VIA KWEE–VAN WOERDEN ─────────────────────────────
    print("\n  [6B] Computing precise Times of Minimum (Kwee–van Woerden)...")

    half_win = period * 0.15
    tom_list = []
    MAX_ERR_DAYS = 0.015  # REVISED: Allow slightly higher error for surgical curves

    for idx in min_indices:
        t_trial_btjd = time[idx]
        t_trial_bjd  = time_bjd[idx]
        try:
            t_min, t_err = kwee_van_woerden(
                time_bjd, flux, t_trial_bjd, half_window=half_win
            )
            if t_err <= MAX_ERR_DAYS:
                tom_list.append((t_min, t_err))
                print(f"    ToM = {t_min:.6f} BJD  ±{t_err:.6f} d  (KvW)")
            else:
                print(f"    REJECTED: ToM at {t_min:.4f} has error ±{t_err:.2f} d")
        except ValueError:
            try:
                t_min, t_err = polynomial_minimum(
                    time_bjd, flux, t_trial_bjd, half_window=half_win
                )
                if t_err <= MAX_ERR_DAYS:
                    tom_list.append((t_min, t_err))
                    print(f"    ToM = {t_min:.6f} BJD  ±{t_err:.6f} d  (poly)")
                else:
                    print(f"    REJECTED: ToM at {t_min:.4f} has error ±{t_err:.2f} d")
            except ValueError as e:
                print(f"    Skipping minimum at BJD {t_trial_bjd:.4f}: {e}")

    if len(tom_list) == 0:
        print("  ERROR: No valid ToM values passed the quality filter. Skipping O–C analysis.")
        continue

    t_obs_arr = np.array([t for t, _ in tom_list])
    t_err_arr = np.array([e for _, e in tom_list])

    # ── 6C: RETRIEVE KREINER BASELINE AND COMPUTE O–C ─────────────────────
    print("\n  [6C] Computing O–C residuals against Kreiner (2004) baseline...")

    if name in KREINER_EPHEMERIDES:
        T0_kr, P_kr = KREINER_EPHEMERIDES[name]
        print(f"    Historical Baseline found: T0 = {T0_kr}, P = {P_kr}")
    else:
        print(f"  WARNING: No Kreiner ephemeris for {name}. Using first ToM as T0.")
        T0_kr = t_obs_arr[0]
        P_kr  = period

    epochs, oc, oc_err = compute_oc(t_obs_arr, t_err_arr, T0_kr, P_kr)

    # ── 6D: REFINE THE LINEAR EPHEMERIS ───────────────────────────────────
    print("\n  [6D] Fitting refined linear ephemeris...")

    T0_new, P_new, T0_err, P_err = refine_ephemeris(epochs, t_obs_arr, t_err_arr)

    print(f"  Refined ephemeris (BJD):")
    print(f"    T_min = {T0_new:.6f} (±{T0_err:.6f}) "
          f"+ {P_new:.8f} (±{P_err:.8f}) × E")

    with open(f"ephem_{safe}.txt", "w") as f:
        f.write(f"# Refined linear ephemeris for {name}\n")
        f.write(f"# T_min (BJD) = T0_new + P_new * E\n")
        f.write(f"# NOTE: Kreiner T0 is HJD; ToM are in BJD. Residual\n")
        f.write(f"#       HJD-to-BJD correction (<4 s) is absorbed into T0_new.\n")
        f.write(f"T0_new  = {T0_new:.8f}  # BJD\n")
        f.write(f"T0_err  = {T0_err:.8f}  # BJD\n")
        f.write(f"P_new   = {P_new:.10f}  # days\n")
        f.write(f"P_err   = {P_err:.10f}  # days\n")
        f.write(f"T0_kreiner = {T0_kr:.8f}  # HJD (Kreiner 2004)\n")
        f.write(f"P_kreiner  = {P_kr:.10f}  # days (Kreiner 2004)\n")

    with open(f"tominima_{safe}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Star", "T_min_BJD", "T_min_err_days", "Epoch_E", "OC_days"])
        for i in range(len(t_obs_arr)):
            writer.writerow([
                name,
                f"{t_obs_arr[i]:.8f}",
                f"{t_err_arr[i]:.8f}",
                epochs[i],
                f"{oc[i]:.8f}"
            ])

    print(f"  Saved: tominima_{safe}.csv")
    print(f"  Saved: ephem_{safe}.txt")

    # ── 6E: ENHANCED O–C DIAGRAM PLOT (WITH TESS ZOOM) ─────────────────────
    print(f"  Generating enhanced diagnostic plot for {safe}...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'hspace': 0.4})
    fig.suptitle(f"{name} — O–C Analysis & Ephemeris Refinement (BJD)", fontsize=16, fontweight='bold')

    # --- PANEL A: KREINER 2004 BASELINE (Historical Trend) ---
    axes[0].errorbar(
        epochs, oc * 1440, yerr=oc_err * 1440,
        fmt="ko", ms=5, capsize=3, elinewidth=1.2, label="TESS Measurements"
    )
    axes[0].axhline(0, color="r", lw=1.5, ls="--", label="Kreiner Prediction (2004)")
    axes[0].set_ylabel("O–C (minutes)")
    axes[0].set_title(f"(a) Global O–C Trend vs. Historical Catalog (Epoch 0 to {max(epochs)})", loc='left', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    info_a = (
        "• Global Scale: Shows the drift accumulated over ~20 years.\n"
        "• TESS Data: Bunched on the right due to the massive epoch range."
    )
    axes[0].text(0.015, 0.05, info_a, transform=axes[0].transAxes, fontsize=10, linespacing=1.4,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.85))
    axes[0].legend(loc="upper left")

    # --- PANEL B: REFINED EPHEMERIS (Linear Fit) ---
    oc_refined = t_obs_arr - (T0_new + P_new * epochs)
    axes[1].errorbar(
        epochs, oc_refined * 1440, yerr=oc_err * 1440,
        fmt="bs", ms=5, capsize=3, elinewidth=1.2,
        label="Refined Residuals"
    )
    axes[1].axhline(0, color="r", lw=1.5, ls="--", label="Refined Ephemeris")
    axes[1].set_ylabel("O–C (minutes)")
    axes[1].set_title(f"(b) Residuals vs. Refined Linear Ephemeris (P = {P_new:.7f} d)", loc='left', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left")

    # --- PANEL C: TESS ERA ZOOM (Revealing all points) ---
    axes[2].errorbar(
        epochs, oc_refined * 1440, yerr=oc_err * 1440,
        fmt="ro", ms=6, capsize=4, elinewidth=1.5, mfc='white', mew=1.5,
        label="Individual TESS Eclipses"
    )
    axes[2].axhline(0, color="k", lw=1.0, ls="-")
    axes[2].set_xlabel("Orbital Epoch (E)")
    axes[2].set_ylabel("O–C (minutes)")
    
    # Calculate a nice zoom range around the TESS epochs
    e_min, e_max = min(epochs), max(epochs)
    e_range = e_max - e_min if e_max > e_min else 10
    axes[2].set_xlim(e_min - 0.1*e_range - 1, e_max + 0.1*e_range + 1)
    
    axes[2].set_title(f"(c) Zoom View: TESS Observations ({len(epochs)} individual points)", loc='left', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.4, linestyle='--')
    
    info_c = (
        f"• TESS Zoom: Revealing all {len(epochs)} Measured Times of Minimum.\n"
        "• Scatter: Vertical distribution represents the physical timing uncertainty.\n"
        "• Purpose: Proves the modern data provides a dense, consistent orbital solution."
    )
    axes[2].text(0.015, 0.05, info_c, transform=axes[2].transAxes, fontsize=10, linespacing=1.4,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="honeydew", edgecolor="gray", alpha=0.85))
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"oc_{safe}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Enhanced O–C plot saved: {plot_path}")

print("\n" + "=" * 70)
print("SCRIPT 03 COMPLETE — All O–C analyses done.")
print("=" * 70)