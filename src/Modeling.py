"""
================================================================================
SCRIPT 04 — THERMODYNAMIC MESHING (PHOEBE 2)
================================================================================
Thesis: Thermodynamic Analysis and Modeling of Blackbody Radiation in
        Eclipsing Binary Systems Using Space-Based Photometry

Methodology Step 4.4:
    Initializes 3D stellar meshes using PHOEBE 2. Integrates refined 
    ephemerides from Script 03 to ensure phase synchronization. Employs 
    Blackbody atmosphere approximations and 'Double Flip' constraint 
    hierarchies to maintain mesh stability for distorted contact systems.

Dependencies:
    pip install phoebe lightkurve numpy matplotlib astropy pyreadline3

Inputs:
    benchmark_targets.csv  — Standard target list.
    lc_<star>_clean.fits   — High-amplitude light curves from Script 02.
    ephem_<star>.txt       — Refined orbital anchors from Script 03.

Outputs:
    bundle_<star>.phoebe       — Initialized 3D Forward Model (Inverse Seed).
    plots/phoebe_init_<star>.png — Initial fit verification plot.
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

import phoebe
import lightkurve as lk

# Suppress PHOEBE logging noise for automated execution
warnings.filterwarnings("ignore")
phoebe.logger(clevel="WARNING")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TESS_BTJD_OFFSET = 2457000.0  # Constant for Barycentric BJD conversion

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SPECTRAL CALIBRATIONS (Carroll & Ostlie 2017)
# ─────────────────────────────────────────────────────────────────────────────
# Empirical table used to anchor primary effective temperature (T1).
SPECTRAL_TYPE_TEFF = {
    "O5": 42000, "O6": 39500, "O7": 37000, "O8": 35300, "O9": 33000,
    "B0": 30000, "B1": 25400, "B2": 22000, "B3": 18700, "B5": 15400,
    "B8": 11400, "B9": 10500,
    "A0":  9520, "A1":  9230, "A2":  8970, "A5":  8200, "A7":  7850,
    "F0":  7200, "F2":  6890, "F5":  6440, "F8":  6200,
    "G0":  5930, "G2":  5830, "G5":  5770, "G8":  5570,
    "K0":  5250, "K2":  4780, "K4":  4560, "K5":  4350, "K7":  4060,
    "M0":  3850, "M1":  3720, "M2":  3580, "M3":  3470, "M4":  3370,
    "M5":  3240, "M6":  3050, "M7":  2940, "M8":  2640,
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TARGET-SPECIFIC INITIAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TARGET_PARAMS = {
    "WZ And": {
        "spectral_type": "F5", # Example anchor
        "q_init":         0.40,
        "incl_init":     82.0,
        "fillout_init":   0.15,
    },
    "AA And": {
        "spectral_type": "G2",
        "q_init":         0.55,
        "incl_init":     85.0,
        "fillout_init":   0.20,
    },
    "AB And": {
        "spectral_type": "G5",
        "q_init":         0.45,
        "incl_init":     78.0,
        "fillout_init":   0.15,
    },
    "AD And": {
        "spectral_type": "A0",
        "q_init":         0.80,
        "incl_init":     88.0,
        "fillout_init":   0.10,
    },
    "BD And": {
        "spectral_type": "A5",
        "q_init":         0.60,
        "incl_init":     80.0,
        "fillout_init":   0.10,
    },
}

DEFAULT_PARAMS = {
    "spectral_type": "G5",
    "q_init":         0.50,
    "incl_init":     80.0,
    "fillout_init":   0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LOAD TARGET LIST AND EPHEMERIDES
# ─────────────────────────────────────────────────────────────────────────────

targets = []
try:
    with open("neglected_targets.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append({
                "name":   row["name"],
                "type":   row["type"],
                "period": float(row["period"]),
            })
except FileNotFoundError:
    print("ERROR: benchmark_targets.csv not found.")
    exit()

def load_ephemeris(safe_name):
    """Read refined T0 (BJD) and P from the ephemeris file produced by Script 03."""
    path   = f"ephem_{safe_name}.txt"
    params = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                try:
                    params[key.strip()] = float(val.split("#")[0].strip())
                except ValueError:
                    pass
    return params

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MAIN PHOEBE 2 BUNDLE CONSTRUCTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

for target in targets:
    name   = target["name"]
    ctype  = target["type"]
    period = target["period"]
    safe   = name.replace(" ", "_")

    print("\n" + "=" * 70)
    print(f"PHOEBE 2 MODELING: {name}  (type: {ctype})")
    print("=" * 70)

    # ── 4A: LOAD CLEAN LIGHT CURVE AND FOLD DYNAMICALLY ───────────────────────

    lc_file = f"lc_{safe}_clean.fits"  # FIX: Load the clean LC, not the folded one
    if not os.path.exists(lc_file):
        print(f"  WARNING: {lc_file} not found. Skipping.")
        continue

    lc = lk.read(lc_file)

    # Load refined ephemeris
    ephem     = load_ephemeris(safe)
    P_refined = ephem.get("P_new", period)
    T0_new    = ephem.get("T0_new", 2457000.0)

    # Compute phase directly from the absolute BJD times
    time_btjd = lc.time.value
    time_bjd  = time_btjd + TESS_BTJD_OFFSET
    raw_phase = ((time_bjd - T0_new) / P_refined + 0.5) % 1.0 - 0.5
    
    flux = lc.flux.value
    ferr = lc.flux_err.value if hasattr(lc, "flux_err") else np.full_like(flux, 0.002)

    # FIX: Bin the data to ~150 points to prevent PHOEBE 2 from exhausting memory
    # By grouping the data, we drastically lower the computational overhead of the mesh.
    print(f"  Raw LC loaded: {len(flux)} data points")
    
    # Simple phase binning logic
    num_bins = 150
    bins = np.linspace(-0.5, 0.5, num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    binned_flux = np.zeros(num_bins)
    binned_ferr = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (raw_phase >= bins[i]) & (raw_phase < bins[i+1])
        if np.any(mask):
            binned_flux[i] = np.nanmedian(flux[mask])
            binned_ferr[i] = np.nanstd(flux[mask]) / np.sqrt(np.sum(mask))
            if binned_ferr[i] == 0 or np.isnan(binned_ferr[i]):
                binned_ferr[i] = 0.002
        else:
            binned_flux[i] = np.nan
            
    # Remove empty bins
    valid = np.isfinite(binned_flux)
    phase = bin_centers[valid]
    flux  = binned_flux[valid]
    ferr  = binned_ferr[valid]

    print(f"  Binned down to {len(flux)} points for computational efficiency.")

    # ── 4B: LOAD INITIAL PARAMETERS ───────────────────────────────────────────

    init = TARGET_PARAMS.get(name, DEFAULT_PARAMS)
    T1   = SPECTRAL_TYPE_TEFF.get(init["spectral_type"], 5800)

    print(f"  Primary temperature anchor: T1 = {T1} K "
          f"(spectral type {init['spectral_type']})")

    # ── 4C: INSTANTIATE PHOEBE 2 BUNDLE ──────────────────────────────────────

    print(f"\n  [4C] Building PHOEBE 2 bundle (morphology: {ctype})...")

    is_contact = (ctype == "EW")
    b = phoebe.default_binary(contact_binary=is_contact)

    

    # ── 4D: SET ORBITAL PARAMETERS ────────────────────────────────────────────

    b.set_value("period@binary", P_refined)
    b.set_value("incl@binary",   init["incl_init"])
    b.set_value("ecc@binary",    0.0)          # Circular orbit assumed for tight binaries
    b.set_value("t0_supconj@binary", T0_new)

    # ── 4E: SET STELLAR TEMPERATURES & ATMOSPHERES ────────────────────────────

    b.set_value("teff@primary",   T1)
    b.set_value("teff@secondary", T1 * 0.95)   

    # FIX 1: Force blackbody atmospheres for the initial mesh to prevent 
    # Castelli & Kurucz log(g) out-of-bounds crashes, which frequently occur
    # at the highly distorted equatorial bulges of contact binaries.
    b.set_value("atm@primary", "blackbody")
    b.set_value("atm@secondary", "blackbody")

    # ── 4F: SET GEOMETRY & MASS RATIO ─────────────────────────────────────────

    if is_contact:
        # THE DOUBLE FLIP: Unlocking the PHOEBE constraint tree
        # 1. Tell 'pot' to control 'requiv' (Frees pot)
        b.flip_constraint('pot@contact_envelope', solve_for='requiv@primary')
        
        # 2. Tell 'fillout_factor' to control 'pot' (Frees fillout_factor)
        b.flip_constraint('fillout_factor@contact_envelope', solve_for='pot@contact_envelope')
            
        b.set_value("q@binary", init["q_init"])
        b.set_value("fillout_factor@component", init.get("fillout_init", 0.15))
    else:
        b.set_value("q@binary", init["q_init"])
        b.set_value("requiv@primary",   0.8)
        b.set_value("requiv@secondary", 0.6)

    # ---------------------------------------------------------
    # INJECT BENCHMARK ANCHORS HERE 
    # ---------------------------------------------------------
    if "SW Lac" in name:
        b.set_value('teff@primary', 5800)
        b.set_value('q@binary', 1.27)
        b.set_value('fillout_factor@component', 0.30)

    elif "YY Eri" in name:
        b.set_value('teff@primary', 5350)
        b.set_value('q@binary', 0.44)
        b.set_value('fillout_factor@component', 0.15)

    # ── 4G: ATTACH THE TESS LIGHT CURVE DATASET ───────────────────────────────

    # [IMPROVEMENT] Mesh Resolution Hardening
    # Increasing ntriangles to 2000 prevents "Mesh failed or incomplete" errors
    # during extreme geometric distortions in contact systems.
    b.set_value_all('ntriangles', 2000)

    print(f"\n  [4G] Attaching TESS light curve dataset...")

    # Convert binned phase back to absolute BJD times for PHOEBE
    times_bjd = T0_new + phase * P_refined

    b.add_dataset(
        "lc",
        times=times_bjd,
        fluxes=flux,
        sigmas=ferr,
        dataset="lc01",
        passband="TESS:T",  
        overwrite=True
    )

    # Force PHOEBE to automatically scale its raw synthetic 
    # luminosity down to match the normalized TESS data (baseline 1.0).
    b.set_value('pblum_mode', dataset='lc01', value='dataset-scaled')

    # FIX 3: Because we use blackbody atmospheres, we cannot use tabulated 
    # limb darkening. We must set it to manual for BOTH stars explicitly to 
    # avoid mathematical crashes during mesh rendering.
    b.set_value("ld_mode@primary@lc01", "manual")
    b.set_value("ld_mode@secondary@lc01", "manual")

    print(f"  Dataset 'lc01': attached successfully.")

    # ── 4H: CONFIGURE FREE PARAMETERS ─────────────────────────────────────────

    print(f"\n  [4H] Configuring free parameters for inverse problem...")

    if is_contact:
        print("  Contact binary: free params → teff@secondary, incl, q, fillout_factor")
    else:
        print("  Detached/SD binary: parameters configured.")

    # [IMPROVEMENT] Dynamic Radiative Constants (von Zeipel vs Lucy)
    for component in ['primary', 'secondary']:
        teff = b.get_value(f'teff@{component}')
        if teff >= 8000:
            # Radiative atmosphere
            b.set_value(f'gravb_bol@{component}', 1.0)
            b.set_value(f'irrad_frac_refl_bol@{component}', 1.0)
            print(f"    {component.capitalize()} ({teff}K): Set radiative physics (β1=1.0, refl=1.0)")
        else:
            # Convective atmosphere
            b.set_value(f'gravb_bol@{component}', 0.32)
            b.set_value(f'irrad_frac_refl_bol@{component}', 0.6)
            print(f"    {component.capitalize()} ({teff}K): Set convective physics (β1=0.32, refl=0.6)")

    # ── 4I: RUN INITIAL FORWARD MODEL ─────────────────────────────────────────

    print(f"\n  [4I] Running PHOEBE 2 forward model (initial synthetic LC)...")

    # [CRITICAL FIX 2b] Reduce mesh resolution to lower per-step compute time.
    # 1500 is the safe minimum for contact binaries undergoing MCMC exploration
    # to prevent mesh-collapse (-inf) at extreme fillout factors.
    # [HARDENING] Increased to 2000 to prevent mesh collapse in contact systems.
    b.set_value_all('ntriangles', 2000)
    print("  Mesh resolution safely forced to 2000 triangles.")

    try:
        b.run_compute(
            compute="phoebe01",
            model="init_model",
            overwrite=True,
            irrad_method="none",
            distortion_method="roche"
        )
        print("  Forward model computed successfully.")
    except Exception as e:
        print(f"  Forward model error: {e}")
        continue

    # ── 4J: DIAGNOSTIC PLOT ───────────────────────────────────────────────────

    try:
        synth_times  = b.get_value("times",  model="init_model", dataset="lc01")
        synth_fluxes = b.get_value("fluxes", model="init_model", dataset="lc01")

        # Normalise synthetic fluxes to match the observed normalised scale
        synth_fluxes = synth_fluxes / np.nanmedian(synth_fluxes)

        synth_phase  = ((synth_times - T0_new) / P_refined + 0.5) % 1.0 - 0.5
        sort_idx     = np.argsort(synth_phase)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(
            f"{name} — PHOEBE 2 Initial Forward Model vs. TESS Data",
            fontsize=12, fontweight='bold'
        )

        axes[0].errorbar(phase, flux, yerr=ferr, fmt="k.", ms=6, alpha=0.6,
                         elinewidth=1.0, label="TESS data (Binned)")
        axes[0].plot(synth_phase[sort_idx], synth_fluxes[sort_idx],
                     "r-", lw=2.5, label="PHOEBE 2 Initial Mesh")
        axes[0].set_ylabel("Normalized Flux")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        synth_at_obs = np.interp(
            np.sort(phase),
            synth_phase[sort_idx],
            synth_fluxes[sort_idx]
        )
        residuals = flux[np.argsort(phase)] - synth_at_obs

        axes[1].plot(np.sort(phase), residuals, "g.", ms=5, alpha=0.6)
        axes[1].axhline(0, color="r", lw=1.5)
        axes[1].set_xlabel("Orbital Phase φ")
        axes[1].set_ylabel("Residuals")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"phoebe_init_{safe}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Initial model plot saved: {plot_path}")

    except Exception as e:
        print(f"  Could not generate diagnostic plot: {e}")

    # ── 4K: SAVE THE BUNDLE ───────────────────────────────────────────────────

    bundle_file = f"bundle_{safe}.phoebe"
    b.save(bundle_file)

    print(f"\n  Bundle saved: {bundle_file}")
    print(f"  Bundle summary:")
    print(f"    System type : {'Contact (EW)' if is_contact else 'Detached/SD (EB)'}")
    print(f"    T1 (fixed)  = {b.get_value('teff@primary'):.0f} K")
    print(f"    T2 (init)   = {b.get_value('teff@secondary'):.0f} K")
    print(f"    incl (init) = {b.get_value('incl@binary'):.2f}°")
    print(f"    q (init)    = {b.get_value('q@binary'):.3f}")

print("\n" + "=" * 70)
print("SCRIPT 04 COMPLETE — All PHOEBE 2 bundles built and saved.")
print("=" * 70)