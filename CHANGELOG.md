# Changelog

All notable changes to `inheritance-math` are documented here. This project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0a1] — 2026-08-01

Initial alpha release of the Coherent Inheritance Framework (CIF) modeling
library.

### Added
- `InheritanceMap` — single transmission boundary with linear (matrix) or
  nonlinear (callable) operators, Gaussian drift, bounded-safety clipping
  (Axiom 5), and fixed-point / stability analysis.
- `CoherenceLayer` and `LayerStatus` — the five inheritance layers
  (physical, data, method, intent, semantic) with fidelity scores, derived
  status, and recovery (Axiom 3).
- `InheritanceChain` — multi-hop transmission with cumulative coherence-decay
  tracking and bounded-vs-unbounded comparison.
- Metrics: `coherence_score()` (fragile layers weighted higher),
  `boundary_leakage()`, `recovery_index()`, `critical_layers()`.
- `run_sim()` — graph-Laplacian diffusion of a coherence property across a
  network of connected systems (Axiom 4).
- Three worked examples: basic inheritance map, AI fine-tuning inheritance,
  and agent delegation chain.
- Full pytest suite (36 tests) with numeric assertions.

### Notes
- This is a mathematical modeling library, not a physics theory. CIF is a
  candidate framework.
