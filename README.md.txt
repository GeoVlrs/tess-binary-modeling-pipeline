# TESS Eclipsing Binary Modeling Pipeline

An automated, end-to-end Python pipeline for the acquisition, reduction, time-series analysis, and thermodynamic modeling of neglected eclipsing binary systems using space-based photometry.

This repository contains the computational framework developed for the thesis: **"Thermodynamic Analysis and Modeling of Blackbody Radiation in Eclipsing Binary Systems Using Space-Based Photometry."**

## 🔭 Overview
This pipeline systematically identifies short-period eclipsing binaries (such as W UMa contact systems) that lack recent literature, downloads their high-cadence photometric data from the TESS mission, performs rigorous PCA-based detrending, and extracts precision orbital mechanics. Finally, it constructs full 3D synthetic meshes using **PHOEBE 2** to solve the inverse problem and extract fundamental geometric and thermodynamic stellar parameters.

## ⚙️ Pipeline Architecture

The pipeline is split into five sequential scripts, each handling a distinct step in the data reduction and modeling process:

* **`script_01_target_selection.py`** Queries the Kreiner (2004) ephemeris database via VizieR. Cross-references short-period systems against the AAVSO Variable Star Index (VSX) and General Catalogue of Variable Stars (GCVS) to isolate "neglected" targets with available TESS coverage.
* **`script_02_data_reduction.py`** Interfaces with the MAST archive to download TESS Target Pixel Files (TPFs). Performs custom aperture photometry, PCA-based background detrending (with NaN-imputation for matrix stability), $3\sigma$ iterative clipping, and Savitzky-Golay high-pass continuum normalization.
* **`script_03_oc_analysis.py`** Extracts precise Times of Minimum (ToM) utilizing the Kwee–van Woerden (1956) algorithm. Computes HJD-to-BJD time-scale corrections, generates O–C (Observed minus Calculated) diagrams against historical baselines, and computes refined linear ephemerides.
* **`script_04_phoebe_mesh.py`** Initializes the physical stellar mesh using PHOEBE 2. Dynamically applies Blackbody atmospheric approximations, manages "manual" limb darkening, and flips surface equipotential constraints to properly model W UMa-type contact envelopes.
* **`script_05_optimization.py`** Executes the statistical inverse problem solver. Utilizes a bounded Nelder-Mead simplex algorithm for rapid $\chi^2$ minimization, with secondary support for Markov Chain Monte Carlo (MCMC) posterior sampling via `emcee`. Outputs the final geometric and thermodynamic parameters.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/tess-binary-modeling-pipeline.git](https://github.com/YOUR_USERNAME/tess-binary-modeling-pipeline.git)
   cd tess-binary-modeling-pipeline