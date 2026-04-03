"""
================================================================================
SCRIPT 05 — STATISTICAL OPTIMIZATION
================================================================================
Thesis: Thermodynamic Analysis and Modeling of Blackbody Radiation in
        Eclipsing Binary Systems Using Space-Based Photometry

Methodology Step 4.5:
    Loads the pre-configured PHOEBE 2 bundle from Script 04 and executes the
    statistical inverse problem solver. Two solver strategies are implemented:

    Strategy A: χ² minimization (Nelder-Mead simplex via scipy)
        → Fast convergence; ideal for well-constrained starting parameters.
        → Risk of local minima; best used as first pass.

    Strategy B: MCMC posterior sampling (emcee)
        → Full posterior probability distribution mapping.
        → Robust uncertainty quantification; computationally expensive.
        → Recommended for publication-quality results.
================================================================================
"""

import csv
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phoebe

warnings.filterwarnings("ignore")
phoebe.logger(clevel="WARNING")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SOLVER_STRATEGY = "nelder_mead"  # Change to "emcee" for final MCMC runs

MCMC_NWALKERS = 16
MCMC_NITER    = 1000
MCMC_BURNIN   = 200

TOLERANCE = 1e-4  # Loosened slightly for faster Nelder-Mead convergence
GELMAN_RUBIN_THRESHOLD = 1.1

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FREE PARAMETER DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# For EW (contact) systems
FREE_PARAMETERS_CONTACT = {
    "teff@secondary":                  (3000,  10000, "Secondary effective temperature (K)"),
    "incl@binary":                     (50.0,   90.0, "Orbital inclination (degrees)"),
    # q@binary is omitted here for Nelder-Mead. Fitting q photometrically for EW 
    # systems causes severe degeneracy and Roche-geometry crashes without RV data.
    "fillout_factor@contact_envelope@envelope@component": ( 0.0,   0.95, "Fillout factor f"), 
}

# For EB (detached/semi-detached) systems
FREE_PARAMETERS_DETACHED = {
    "teff@secondary":    (3000,  10000, "Secondary effective temperature (K)"),
    "incl@binary":       (50.0,   90.0, "Orbital inclination (degrees)"),
    "q@binary":          ( 0.1,    1.0, "Mass ratio m2/m1"),
    "requiv@primary":    ( 0.2,    2.0, "Equivalent radius R1"),
    "requiv@secondary":  ( 0.2,    2.0, "Equivalent radius R2"),
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_chi2(b, free_params, dataset="lc01"):
    try:
        obs_flux  = b.get_value("fluxes", dataset=dataset, context="dataset")
        obs_sigma = b.get_value("sigmas", dataset=dataset, context="dataset")
        syn_flux  = b.get_value("fluxes", dataset=dataset, model="solution_model")

        syn_flux  = syn_flux / np.nanmedian(syn_flux) * np.nanmedian(obs_flux)

        n_obs     = len(obs_flux)
        n_free    = len(free_params)
        dof       = max(1, n_obs - n_free)
        chi2      = np.sum(((obs_flux - syn_flux) / obs_sigma) ** 2)
        chi2_red  = chi2 / dof
        return chi2, chi2_red, n_obs
    except Exception:
        return np.inf, np.inf, 0

def gelman_rubin(chains):
    n_walkers, n_steps, n_params = chains.shape
    R_hat = np.zeros(n_params)
    for p in range(n_params):
        chain_p = chains[:, :, p]
        W       = np.mean(np.var(chain_p, axis=1, ddof=1))
        theta_j = np.mean(chain_p, axis=1)
        theta   = np.mean(theta_j)
        B       = (n_steps / (n_walkers - 1)) * np.sum((theta_j - theta) ** 2)
        V_hat   = ((n_steps - 1) / n_steps) * W + (1 / n_steps) * B
        R_hat[p] = np.sqrt(V_hat / W) if W > 0 else np.inf
    return R_hat

def extract_results(b, free_params):
    results = {}
    T1 = b.get_value("teff@primary")
    T2 = b.get_value("teff@secondary")

    results["T1_K"]       = (T1,      0.0)
    results["T2_K"]       = (T2,      np.nan)
    results["T2_over_T1"] = (T2 / T1, np.nan)
    results["incl_deg"]   = (b.get_value("incl@binary"),  np.nan)
    results["q"]          = (b.get_value("q@binary"),      np.nan)

    if "fillout_factor@contact_envelope" in free_params:
        try:
            results["fillout_factor"] = (b.get_value("fillout_factor@contact_envelope"), np.nan)
        except Exception:
            results["fillout_factor"] = (np.nan, np.nan)
    else:
        try:
            results["requiv_primary"]   = (b.get_value("requiv@primary"),   np.nan)
            results["requiv_secondary"] = (b.get_value("requiv@secondary"), np.nan)
        except Exception:
            results["requiv_primary"]   = (np.nan, np.nan)
            results["requiv_secondary"] = (np.nan, np.nan)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MAIN OPTIMIZATION LOOP
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
    print("ERROR: neglected_targets.csv not found.")
    exit()

for target in targets:
    name  = target["name"]
    ctype = target["type"]
    safe  = name.replace(" ", "_")

    print("\n" + "=" * 70)
    print(f"OPTIMIZATION: {name}  ({ctype})  |  Strategy: {SOLVER_STRATEGY.upper()}")
    print("=" * 70)

    bundle_file = f"bundle_{safe}.phoebe"
    if not os.path.exists(bundle_file):
        print(f"  WARNING: {bundle_file} not found. Skipping.")
        continue

    b = phoebe.load(bundle_file)
    print(f"  Bundle loaded: {bundle_file}")

    # FIX 2: Set bolometric limb darkening to manual so the optimizer
    # doesn't crash when calculating mutual irradiation/reflection.
    b.set_value("ld_mode_bol@primary", "manual")
    b.set_value("ld_mode_bol@secondary", "manual")

    is_contact  = (ctype == "EW")

    # ── NEW FIX: FLIP CONSTRAINTS FOR CONTACT BINARIES ───────────────────────
    if is_contact:
        print("  [FIX] Adjusting contact binary constraints...")
        try:
            # We flip the constraint so we can optimize fillout_factor directly
            b.flip_constraint('fillout_factor', solve_for='pot')
        except Exception as e:
            print(f"    Note: Could not flip constraint (might be flipped): {e}")
    # ─────────────────────────────────────────────────────────────────────────

    free_params = FREE_PARAMETERS_CONTACT if is_contact else FREE_PARAMETERS_DETACHED
    n_free      = len(free_params)

    # ── 4A: APPLY LIMITS / PRIORS ────────────────────────────────────────────

    print(f"\n  [4A] Setting parameter bounds ({n_free} free parameters)...")

    for twig, (lo, hi, desc) in free_params.items():
        try:
            if SOLVER_STRATEGY == "emcee":
                # FIX 1a: Correct phoebe.distributions module for MCMC
                b.add_distribution(twig, dist.uniform(lo, hi), distribution="priors", overwrite=True)
                print(f"    Added Prior   : {twig:<30} [{lo:.2f}, {hi:.2f}]")
            else:
                # FIX 1b: Apply hard limits for Nelder-Mead instead of distributions
                b.get_parameter(twig).set_limits((lo, hi))
                print(f"    Applied Limits: {twig:<30} [{lo:.2f}, {hi:.2f}]")
        except Exception as e:
            print(f"    {twig}: could not set bound — {e}")

    # ── 4B: CONFIGURE THE SOLVER ──────────────────────────────────────────────

    print(f"\n  [4B] Configuring solver: {SOLVER_STRATEGY}...")

    if SOLVER_STRATEGY == "nelder_mead":
        b.add_solver(
            "optimizer.nelder_mead",
            solver="main_optimizer",
            fit_parameters=list(free_params.keys()),
            overwrite=True
        )
        b.set_value("maxiter@main_optimizer", 100) # Capped to prevent freezing
        print("  Optimizer configured.")

    elif SOLVER_STRATEGY == "emcee":
        b.add_solver(
            "sampler.emcee",
            solver="mcmc_sampler",
            fit_parameters=list(free_params.keys()),
            nwalkers=MCMC_NWALKERS,
            niters=MCMC_NITER,
            overwrite=True
        )
        print(f"  MCMC configured: {MCMC_NWALKERS} walkers, {MCMC_NITER} iters.")

    # ── 4C: RUN THE SOLVER ────────────────────────────────────────────────────

    solver_label = "main_optimizer" if SOLVER_STRATEGY != "emcee" else "mcmc_sampler"

    try:
        print(f"\n  [4C] Executing solver (This will take a few minutes)...")
        b.run_solver(solver_label, solution="best_solution", overwrite=True)
        print("  Solver completed successfully.")
    except Exception as e:
        print(f"  Solver error: {e}")
        continue

    # ── 4D: GELMAN-RUBIN CONVERGENCE CHECK (MCMC only) ────────────────────────
    
    if SOLVER_STRATEGY == "emcee":
        print("\n  [4D] Checking MCMC convergence (Gelman-Rubin R̂)...")
        try:
            raw_chains = b.get_value("samples", solution="best_solution")
            
            if raw_chains.ndim == 3:
                chains_post_burnin = raw_chains[:, MCMC_BURNIN:, :]
            else:
                n_total    = raw_chains.shape[0]
                n_steps_   = n_total // MCMC_NWALKERS
                chains_post_burnin = raw_chains.reshape(MCMC_NWALKERS, n_steps_, n_free)[:, MCMC_BURNIN:, :]

            R_hat = gelman_rubin(chains_post_burnin)
            param_names = list(free_params.keys())

            print(f"  {'Parameter':<35} {'R̂':>8}  {'Status'}")
            print(f"  {'─'*55}")
            for i, (pname, rhat) in enumerate(zip(param_names, R_hat)):
                status = "✓ OK" if rhat < GELMAN_RUBIN_THRESHOLD else "⚠ NOT CONVERGED"
                print(f"  {pname:<35} {rhat:>8.4f}  {status}")

        except Exception as e:
            print(f"  Gelman-Rubin check could not be performed: {e}")

    # ── 4E: ADOPT THE SOLUTION ────────────────────────────────────────────────

    print("\n  [4E] Adopting best-fit solution into physical mesh...")
    try:
        if SOLVER_STRATEGY == "emcee":
            b.adopt_solution("best_solution", burnin=MCMC_BURNIN, adopt_values=True, adopt_distributions=True)
        else:
            b.adopt_solution("best_solution", adopt_values=True)
        print("  Solution adopted.")
    except Exception as e:
        print(f"  Solution adoption error: {e}")

    # ── 4F: FINAL FORWARD MODEL ───────────────────────────────────────────────

    print("\n  [4F] Computing final forward model (irrad=none for speed)...")
    try:
        b.run_compute(
            model="solution_model",
            overwrite=True,
            irrad_method="none", # Kept off to prevent Blackbody conflicts
            distortion_method="roche"
        )
        print("  Final model computed.")
    except Exception as e:
        print(f"  Final model error: {e}")
        continue

    # ── 4G: CHI-SQUARED OF FINAL MODEL ───────────────────────────────────────

    chi2, chi2_red, n_obs = compute_chi2(b, free_params, dataset="lc01")
    print(f"\n  Final Reduced χ² = {chi2_red:.4f}  (N = {n_obs}, n_free = {n_free})")

    # ── 4H: EXTRACT AND REPORT FINAL PARAMETERS ───────────────────────────────

    results = extract_results(b, free_params)

    PARAM_LABELS = {
        "T1_K":             ("Primary temperature T₁",   "K"),
        "T2_K":             ("Secondary temperature T₂", "K"),
        "T2_over_T1":       ("Temperature ratio T₂/T₁",  "—"),
        "incl_deg":         ("Orbital inclination i",     "°"),
        "q":                ("Mass ratio q = m₂/m₁",     "—"),
        "requiv_primary":   ("Primary Radius R₁",         "—"),
        "requiv_secondary": ("Secondary Radius R₂",       "—"),
        "fillout_factor":   ("Contact fillout factor f",  "—"),
    }

    print("\n  ─── FINAL PARAMETER TABLE ───────────────────────────────────")
    print(f"  {'Parameter':<30} {'Value':>10}  Unit")
    print(f"  {'─'*48}")
    for key, (val, err) in results.items():
        if not np.isfinite(val): continue
        label, unit = PARAM_LABELS.get(key, (key, ""))
        print(f"  {label:<30} {val:>10.4f}  {unit}")

    # ── 4I: SAVE RESULTS TO CSV ───────────────────────────────────────────────

    result_file = f"results_{safe}.csv"
    with open(result_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Symbol", "Value", "Unit"])
        for key, (val, err) in results.items():
            label, unit = PARAM_LABELS.get(key, (key, ""))
            writer.writerow([label, key, val, unit])
        writer.writerow(["Reduced chi-squared", "chi2_red", chi2_red, "—"])
        writer.writerow(["Solver strategy", "solver", SOLVER_STRATEGY, "—"])

    # ── 4J: FINAL DIAGNOSTIC PLOT ─────────────────────────────────────────────

    try:
        obs_flux  = b.get_value("fluxes", dataset="lc01", context="dataset")
        obs_times = b.get_value("times",  dataset="lc01", context="dataset")
        obs_sigma = b.get_value("sigmas", dataset="lc01", context="dataset")
        syn_flux  = b.get_value("fluxes", dataset="lc01", model="solution_model")
        syn_times = b.get_value("times",  dataset="lc01", model="solution_model")

        syn_flux  = syn_flux / np.nanmedian(syn_flux) * np.nanmedian(obs_flux)

        P_final  = b.get_value("period@binary")
        T0_final = b.get_value("t0_supconj@binary")

        obs_phase = ((obs_times - T0_final) / P_final + 0.5) % 1.0 - 0.5
        syn_phase = ((syn_times - T0_final) / P_final + 0.5) % 1.0 - 0.5
        sort_idx  = np.argsort(syn_phase)

        residuals = obs_flux - np.interp(obs_phase, syn_phase[sort_idx], syn_flux[sort_idx])

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"{name} — PHOEBE 2 Optimized Fit (χ²_red = {chi2_red:.2f})", fontsize=14, fontweight='bold')

        axes[0].errorbar(obs_phase, obs_flux, yerr=obs_sigma, fmt="k.", ms=6, alpha=0.5, label="TESS Observations")
        axes[0].plot(syn_phase[sort_idx], syn_flux[sort_idx], "r-", lw=2.5, label=f"Optimized Model")
        
        param_text = f"T₁ = {results['T1_K'][0]:.0f} K  |  T₂ = {results.get('T2_K', (np.nan,))[0]:.0f} K  |  i = {results['incl_deg'][0]:.1f}°"
        axes[0].text(0.02, 0.04, param_text, transform=axes[0].transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))
                     
        axes[0].set_ylabel("Normalized Flux")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.3)
        axes[0].invert_yaxis()

        axes[1].plot(obs_phase, residuals, ".", ms=5, color="steelblue", alpha=0.7)
        axes[1].axhline(0.0, color="r", lw=1.5, ls="--")
        axes[1].set_xlabel("Orbital Phase φ")
        axes[1].set_ylabel("Residuals")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"phoebe_final_{safe}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  Could not generate final plot: {e}")

    # ── 4L: SAVE FINAL BUNDLE ─────────────────────────────────────────────────

    final_bundle = f"bundle_{safe}_final.phoebe"
    b.save(final_bundle)

print("\n" + "=" * 70)
print("SCRIPT 05 COMPLETE — Statistical optimization finished.")
print("=" * 70)
