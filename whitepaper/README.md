# Inheritance Math Framework (IMF) — Phase 1 Candidate Whitepaper

**Author:** Derek Hone, Remnant Fieldworks Inc.
**Version:** v1.2.1 — Release Integrity Reconciliation
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
**DOI:** Pending (Zenodo archival in progress)
**Status:** Phase 1 candidate framework — open for review. Not peer-reviewed science.

---

## Overview

The Inheritance Math Framework (IMF) is the formal mathematical core of the Coherent Inheritance Framework (CIF). It provides a candidate phenomenological model for coherence dynamics across inheritance boundaries — applicable to quantum information transmission, software system evolution, ecological knowledge transmission, and institutional governance.

**Epistemic status:** This is a candidate phenomenological framework for Phase 1 open review. It does not claim universal physical theorems, peer-reviewed validation, or empirical establishment. All simulations are labeled as illustrations, not validation.

---

## Release bundle contents

| File | Description |
|---|---|
| `IMF_Whitepaper_RF_Inc.pdf` | Formal mathematical whitepaper, v1.2.1 (~29 pages) |
| `IMF_Whitepaper_v121.html` | HTML source for the PDF |
| `imf_simulations.py` | Canonical simulation script — generates all 6 figures |
| `solver_convergence_check.py` | Standalone solver convergence verification (Euler vs RK45) |
| `solver_convergence_results.csv` | Convergence results from the verification run |
| `parameters_manifest.txt` | All simulation parameter values with changelog |
| `requirements.txt` | Python dependencies |
| `imf_fig1_coherence_dynamics.png` | Canonical ODE: three scenarios |
| `imf_fig2_drift_accumulation.png` | Actual-vs-ideal state-at-landmark error E_k (Prop 4.3) |
| `imf_fig3_resonance_correction.png` | Resonance isolation — F held constant at 0.60 |
| `imf_fig4_lane_comparison.png` | Four domain lanes: qualitative illustration |
| `imf_fig5_executionproof_mapping.png` | ExecutionProof alignment matrix (categorical) |
| `imf_fig6_early_warning.png` | Stochastic early-warning AUC (internal model-recovery test) |

---

## Reproducing the figures

```bash
cd whitepaper/
pip install -r requirements.txt
python imf_simulations.py --output . --seed 42
```

Running this command reproduces all six figures pixel-for-pixel from the canonical seed.

## Solver convergence verification

```bash
python solver_convergence_check.py
```

Compares Euler integration (Δt = 0.10, 0.05, 0.01) against scipy RK45 reference on the resonance-matched scenario. Results written to `solver_convergence_results.csv`.

---

## Mathematical structure (v1.2.1)

- 11 Foundational Assumptions
- 8 Core Definitions
- 1 Model Postulate — Coherence Balance Law: dC/dt = −α·D̄ + β·F + γ·R·ū − δ·C
- 5 Propositions (each with explicit stated assumptions)
- 5 Empirical Hypotheses (each requiring experimental validation)
- 0 results labeled "theorem"

All formal results carry epistemic classification tags. No result is labeled as "validated" by simulation — simulations are illustrations only.

---

## Version history summary

| Version | Summary |
|---|---|
| v1.0 | Initial release — 6 results originally labeled "theorems" |
| v1.1 | First review pass — reclassified theorems to propositions/postulate; fixed JS metric, resonance definition, variable normalization |
| v1.1.1 | Second review pass — unified ODE (paper/code mismatch fixed); retired W̄, introduced F; fixed Prop 4.3 actual-vs-ideal; rebuilt Fig 6 landmark-time AUC |
| v1.2 | Third review pass — fixed Prop 4.3 operator order; projected-model language; Def 3.8/Prop 4.3 state-vs-trajectory distinction; corrected figure descriptions; replaced stale convergence table with standalone script |
| v1.2.1 | Final micro-patch — renamed "Trajectory Error" → "State-at-Landmark Error E_k" throughout; Figure 1 clip comment updated to projected-model language |

---

## License

**CC BY 4.0** — Derek Hone / Remnant Fieldworks Inc., 2026.
You are free to share and adapt with attribution. Full text: https://creativecommons.org/licenses/by/4.0/

---

## Citation

See `CITATION.cff` in the repository root, or cite as:

> Hone, D. (2026). *Inheritance Math Framework (IMF): A Phase 1 Candidate Mathematical Framework for Coherence Dynamics Across Inheritance Boundaries* (v1.2.1). Remnant Fieldworks Inc. https://github.com/derekhone/inheritance-math
