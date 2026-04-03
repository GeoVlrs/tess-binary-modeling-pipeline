"""
================================================================================
SCRIPT 03 — TIME-SERIES ANALYSIS & O–C DIAGRAM GENERATION
================================================================================
Thesis: Thermodynamic Analysis and Modeling of Blackbody Radiation in
        Eclipsing Binary Systems Using Space-Based Photometry

Methodology Step 4.3:
    Determines precise Times of Minimum (ToM) from the detrended TESS light
    curves using the Kwee–van Woerden algorithm or polynomial fitting.
    Computes O–C residuals against the Kreiner (2004) historical ephemeris
    baseline and constructs O–C diagrams to assess orbital period stability.

Dependencies:
    pip install lightkurve numpy scipy matplotlib astropy astroquery

Inputs:
    neglected_targets.csv          — target list from Script 01
    lc_<starname>_clean.fits       — detrended light curves from Script 02

Outputs:
    tominima_<starname>.csv        — table of measured ToM values with errors
    ephem_<starname>.txt           — refined linear ephemeris
    plots/oc_<starname>.png        — O–C diagram
================================================================================
"""

import csv
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize  import curve_fit
from scipy.signal    import argrelmin
from scipy.ndimage   import uniform_filter1d
import lightkurve as lk
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TIME-SCALE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TESS_BTJD_OFFSET = 2457000.0    # add this to BTJD to get BJD

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: KWEE–VAN WOERDEN ALGORITHM (CLASSICAL UNCERTAINTY)
# ─────────────────────────────────────────────────────────────────────────────

def kwee_van_woerden(time, flux, t_trial, half_window=0.05):
    """
    Compute the Time of Minimum using the Kwee–van Woerden (1956) method.
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
    Fit a polynomial to the eclipse bottom and return its minimum as ToM.
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
    Compute O–C residuals against the Kreiner (2004) baseline ephemeris.
    """
    epochs = np.round((t_obs_bjd - T0_hjd) / P_kreiner).astype(int)
    t_calc = T0_hjd + P_kreiner * epochs
    oc     = t_obs_bjd - t_calc
    oc_err = t_obs_err
    return epochs, oc, oc_err

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: REFINED LINEAR EPHEMERIS
# ─────────────────────────────────────────────────────────────────────────────

def refine_ephemeris(epochs, t_obs_bjd, t_obs_err):
    """
    Fit T_obs = T0_new + P_new × E (weighted least squares).
    """
    def linear(E, T0, P):
        return T0 + P * E

    # FIX: Use polyfit to generate a mathematically sound initial guess
    # regardless of how large the epoch 'E' has become.
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
    print("ERROR: neglected_targets.csv not found. Run Script 01 first.")
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
    time = lc.time.value    # BTJD
    flux = lc.flux.value

    # Convert BTJD → BJD
    time_bjd = time + TESS_BTJD_OFFSET

    print(f"  Loaded {len(lc)} data points.")
    print(f"  Time range: BTJD {time.min():.2f} – {time.max():.2f}  "
          f"(BJD {time_bjd.min():.2f} – {time_bjd.max():.2f})")

    # ── 6A: IDENTIFY ECLIPSE MINIMA ────────────────────────────────────────
    print("\n  [6A] Identifying eclipse minima candidates...")

    flux_smooth = uniform_filter1d(flux, size=15)
    
    # FIX: Dynamically calculate the search order based on actual time array
    cadence_days = np.median(np.diff(time))
    points_per_period = period / cadence_days
    order = max(5, int(points_per_period * 0.2)) # Must be 20% of a period apart

    min_indices = argrelmin(flux_smooth, order=order)[0]

    print(f"  Found {len(min_indices)} candidate minima (order = {order})")

    if len(min_indices) == 0:
        print("  ERROR: No minima detected. Check light curve quality.")
        continue

    # ── 6B: PRECISE ToM VIA KWEE–VAN WOERDEN ─────────────────────────────
    print("\n  [6B] Computing precise Times of Minimum (Kwee–van Woerden)...")

    half_win = period * 0.15
    tom_list = []
    MAX_ERR_DAYS = 0.005  # STRICT QUALITY FILTER: Reject errors > ~7.2 minutes

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

    # ── 6E: PLOT O–C DIAGRAM ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"{name} — O–C Diagram (Kreiner 2004 baseline, BJD)",
        fontsize=12
    )

    axes[0].errorbar(
        epochs, oc * 1440, yerr=oc_err * 1440,
        fmt="ko", ms=5, capsize=3, elinewidth=1.2, label="O–C (Kreiner baseline)"
    )
    axes[0].axhline(0, color="r", lw=1.0, ls="--", label="Zero reference")
    axes[0].set_ylabel("O–C (minutes)")
    axes[0].legend(fontsize=9)
    axes[0].set_title("(a) O–C residuals vs. Kreiner (2004) ephemeris")
    axes[0].grid(alpha=0.3)

    oc_refined = t_obs_arr - (T0_new + P_new * epochs)
    axes[1].errorbar(
        epochs, oc_refined * 1440, yerr=oc_err * 1440,
        fmt="bs", ms=5, capsize=3, elinewidth=1.2,
        label="O–C (refined ephemeris)"
    )
    axes[1].axhline(0, color="r", lw=1.0, ls="--")
    axes[1].set_xlabel("Epoch E")
    axes[1].set_ylabel("O–C residual (minutes)")
    axes[1].set_title("(b) Residuals after refined linear ephemeris fit")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"oc_{safe}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  O–C plot saved: {plot_path}")

print("\n" + "=" * 70)
print("SCRIPT 03 COMPLETE — All O–C analyses done.")
print("=" * 70)
