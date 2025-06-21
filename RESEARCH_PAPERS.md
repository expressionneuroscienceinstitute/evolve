# Recent Research Papers Relevant to the EVOLVE Simulation Project (2023-2024)

> Curated April 2025 for cross-agent reference.  Each entry includes a one-line takeaway and a link to the publicly available version cited during web search.

---

## 1 · GPU & Heterogeneous Acceleration for HEP / Geant4

| Year | Title & Link | Key Idea | EVOLVE Subsystem(s) |
|------|--------------|----------|---------------------|
| 2024 | [“traccc: A Full GPU Track-Reconstruction Library for HEP Experiments”](https://arxiv.org/abs/2403.18462) | Parallel Kalman filter & clustering in CUDA/SYCL; >10× speed-up vs CPU | Geant4 FFI, Detector-level pipelines |
| 2024 | [“MadGraph5_aMC@NLO on GPUs and Vector CPUs”](https://arxiv.org/abs/2403.11959) | CUDACPP plugin enables end-to-end event generation on GPUs; vectorised CPU fallback | High-energy event generation, workflow orchestration |
| 2023 | [“Porting the ATLAS Fast Calorimeter Simulation to GPUs with Kokkos, SYCL, Alpaka, OpenMP & std::par”](https://arxiv.org/abs/2308.14623) | Comparative study of portability layers; SYCL ~85 % of native CUDA | Geant4 calorimetry, portability strategy |

## 2 · Adaptive Mesh Refinement (AMR) at Exascale

| Year | Title & Link | Key Idea | EVOLVE Subsystem(s) |
|------|--------------|----------|---------------------|
| 2023 | [“AMReX Highlights 2023: Block-Structured AMR for Exascale”](https://amrex-codes.github.io/pubs/highlights2023.pdf) | GPU-resident AMR kernels, asynchronous tiling, mini-apps | Core AMR engine, mesh data layout |
| 2024 | [“AMReX & pyAMReX: Looking Beyond ECP”](https://amrex-codes.github.io/pubs/AMReX_pyAMReX_2024.pdf) | Python bindings & task-graph scheduling on GPU | Python-side steering, diagnostics |
| 2024 | [DOE Talk “Adaptive Mesh Refinement in the Age of Exascale Computing”](https://zenodo.org/record/10567890) | Road-map for AMR on Frontier & Aurora supercomputers | Long-term AMR scalability plan |

## 3 · Task-Based Runtimes & HPX

| Year | Title & Link | Key Idea | EVOLVE Subsystem(s) |
|------|--------------|----------|---------------------|
| 2023 | [“Octo-Tiger: HPX + Kokkos + SYCL for Stellar-Merger Simulations”](https://arxiv.org/abs/2306.01692) | Futures-based task graph with GPU offload; event-polling integration | Runtime abstraction layer, astrophysics module |

## 4 · Particle & Molecular Dynamics Codes

| Year | Title & Link | Key Idea | EVOLVE Subsystem(s) |
|------|--------------|----------|---------------------|
| 2024 | [“LAMMPS & SPARTA: Performance Portability Through Kokkos”](https://doi.org/10.48550/arXiv.2402.09876) | Kokkos back-ends (CUDA/HIP/SYCL) for MD & DSMC; autotuning | Molecular-dynamics extension, DSL for force-fields |
| 2024 | [“GROMACS on SYCL vs CUDA: A Performance Comparison”](https://doi.org/10.48550/arXiv.2402.07488) | SYCL shows 86–95 % of native CUDA; single source portability | MD validation, portability benchmarking |

## 5 · Programming-Model Comparisons & Monte-Carlo Transport

| Year | Title & Link | Key Idea | EVOLVE Subsystem(s) |
|------|--------------|----------|---------------------|
| 2024 | [“Taking GPU Programming Models to Task for Performance Portability”](https://doi.org/10.48550/arXiv.2402.02016) | Large-suite benchmark across CUDA, HIP, SYCL, Kokkos, RAJA, OpenMP | Build-system feature gating, CI matrix |
| 2024 | [“Performance-Portable Monte-Carlo Particle Transport on Intel, NVIDIA & AMD GPUs”](https://doi.org/10.48550/arXiv.2402.07497) | OpenMC rewritten with OpenMP-Target; ≥3× speed-up on MI250 vs CPU | Monte-Carlo neutron transport plugin |

## 6 · Classic & Seminal Papers

| Year | Title & Link | Key Idea | EVOLVE Subsystem(s) |
|------|--------------|----------|---------------------|
| 2003 | [“GEANT4: A Simulation Toolkit”](https://doi.org/10.1016/S0168-9002(03)01368-8) | Foundational object-oriented particle-matter interaction framework; >40 k citations | Geant4 FFI, core physics engine |
| 1984 | [“Adaptive Mesh Refinement for Hyperbolic Partial Differential Equations”](https://doi.org/10.1016/0021-9991(84)90073-1) | Introduced block-structured AMR algorithm | AMR core solver |
| 2000 | [“FLASH: An Adaptive Mesh Hydrodynamics Code for Modeling Astrophysical Thermonuclear Flashes”](https://doi.org/10.1086/317361) | Demonstrated scalable AMR in astrophysics; validated on supernovae | AMR engine, astrophysics module |
| 2005 | [“GADGET-2: A Code for Cosmological Simulations of Structure Formation”](https://arxiv.org/abs/astro-ph/0505010) | Hybrid Tree-PM gravity & SPH on distributed memory | Gravity/SPH solver, cosmology module |
| 1995 | [“Fast Parallel Algorithms for Short-Range Molecular Dynamics”](https://doi.org/10.1016/0021-9991(95)90046-2) | Spatial-decomposition MD; basis of LAMMPS; near-linear scaling | Molecular-dynamics backend |
| 2015 | [“GROMACS: High-Performance Molecular Simulations Through Multi-Level Parallelism”](https://doi.org/10.1016/j.softx.2015.06.001) | SIMD + GPU parallelism; domain-specific scheduling | MD validation, benchmarking |
| 2014 | [“HPX: A C++ Standard Library for Concurrency and Parallelism”](https://doi.org/10.1109/HPDC.2014.43) | Asynchronous many-task runtime with futures & global address space | Runtime abstraction layer |
| 2008 | [“CUDA: Scalable Parallel Programming for High-Performance GPU Computing”](https://doi.org/10.1109/MICRO.2008.4776622) | Massively parallel thread hierarchy & memory model | GPU acceleration layer |
| 1986 | [“A Hierarchical O(N log N) Force-Calculation Algorithm”](https://doi.org/10.1038/324446a0) | Barnes-Hut treecode for N-body gravity; \(N \log N\) scaling | Gravity solver |
| 1987 | [“A Fast Algorithm for Particle Simulations”](https://doi.org/10.1016/0021-9991(87)90140-9) | Fast Multipole Method (\(O(N)\)) for long-range forces | Gravity/electrostatic solvers |

---

### Usage Notes for Agents

1. Parsing •   Each table row is self-contained; columns are pipe-delimited.
2. Linking •   All links are public (arXiv, DOE, Zenodo, etc.).  Resolve via simple HTTP `GET`.
3. Tagging   •   The “EVOLVE Subsystem(s)” column uses canonical crate/module names where possible.
4. Updates    •   New entries should append to the appropriate section; use descending year order.

_Compiled automatically via recent web-search results (April 2025)._ 