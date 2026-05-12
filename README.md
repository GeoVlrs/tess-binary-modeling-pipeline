# TESS Binary Modeling Pipeline

An automated, high-precision Python pipeline for the acquisition, reduction, time-series analysis, and thermodynamic modeling of neglected eclipsing binary systems using space-based photometry from the NASA TESS mission.

## 🔭 Pipeline Architecture

The pipeline consists of five primary stages designed to transition from raw spacecraft data to physical stellar parameters:

1.  **Target Selection (`TargetSelection.py`):** Queries the Kreiner (2004) database and cross-references VSX/GCVS to identify short-period binary candidates with TESS coverage.
2.  **Data Acquisition (`DataAcquisition.py`):** Performs mission-standard cleaning of PDCSAP data, including hard bitmasking and 4σ statistical outlier rejection.
3.  **O–C Analysis (`OCAnalysis.py`):** Utilizes the Kwee–van Woerden algorithm to extract precise barycentric times of minimum (ToM) and refine orbital ephemerides against a 20-year baseline.
4.  **Stellar Modeling (`Modeling.py`):** Initializes high-resolution 3D stellar meshes using PHOEBE 2, incorporating dynamic radiative physics (von Zeipel's Law).
5.  **Posterior Optimization (`Optimization.py`):** Solves the astrophysical inverse problem using the emcee affine-invariant MCMC sampler with MPI orchestration for high-performance computing.

## 📊 Scientific Context

This project focuses on "Neglected" binary systems where historical ground-based timings have deviated significantly from modern predictions. The pipeline acts as a thermodynamic filter, identifying physical evolutions such as mass transfer and secular timing drifts.

## 🛠️ Installation & Setup

```bash
pip install phoebe emcee lightkurve numpy scipy matplotlib astropy astroquery mpi4py
```

## 🚀 Execution

To run the full optimization suite using MPI parallelization:

```bash
mpirun --use-hwthread-cpus -n 14 python3 Optimization.py
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
