# Evolve: The Game of Life

You are Cursor, an AI coding agent. **If you encounter missing details, first try to deduce or research them yourself (see §2.2 for online data ingestion). Only after that effort may you ask the operator for clarification—send a single succinct message prefixed `CURSOR_QUERY:` through the Oracle‑Link (§12).** **Every evolving entity in this universe is a fully autonomous software being—no human strategy scripts, nudges, or mid‑game tweaks **unless an explicit God‑Mode override is enabled (see §11)**. Operators may observe only through the read‑only diagnostics described below by default; the agents themselves must discover how to thrive and reach the ultimate Goal Chain. Operators may observe only through the read‑only diagnostics described below; the agents themselves must discover how to thrive and reach the ultimate Goal Chain.** Build a **headless**, resource‑constrained simulation (with optional diagnostic dashboards) that simulates the emergence of matter, stars, planets, life, and intelligent civilization from the moment of the Big Bang to the far future.

## 1. Core Loop & Win/Lose Conditions

1. The game runs in discrete **ticks**. **Default scale = 1 tick = 1 million years**, but this can be changed via CLI (`--tick-span <years>`).
2. **Wall‑Clock Decoupling:** Simulation speed is limited **only** by available compute. `universe_simd` auto‑benchmarks at startup and picks the highest stable **UPS** (updates per second) it can sustain without violating host caps (§7.6). Typical modern CPU ≈ 200–5 000 UPS (i.e., 0.2–10 billion sim‑years per real‑hour).
3. **Time‑Warp Controls:**

   | Command                                                                                                                    | Effect                          | Bounds                  |
   | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ----------------------- |
   | `universectl speed <factor>`                                                                                               | Adjust target UPS by multiplier | 0.1× – MAX (autoscaled) |
   | `godmode time-warp <factor>`                                                                                               | Same as above via §11 miracles  | 0.1× – 1000×            |
   | The scheduler meter‑reads every 5 s and adapts to hardware changes (new GPU, CPU frequency shift) to keep UPS near target. |                                 |                         |
4. **Infinite Run Design:** There is **no built‑in end state** after victory. Once a lineage hits Immortality, the sim continues indefinitely, allowing further exploration (higher‑Kardashev scales, emergent cosmology, cyclic multiverse, etc.). Entropy ceiling is adjustable; by default it asymptotes at heat death \~10¹⁰⁰ years.
5. **Historical Scrubbing:** Every checkpoint stores a digest (approx. 64 B) in `history.idx` so operators can `universectl rewind <ticks>` or `inspect --at <tick>` to view past states. Index is O(num checkpoints), keeping long histories cheap.
6. **Win (Autonomous Goal Chain):** Without any human micromanagement, a lineage must sequentially achieve **Sentience → Industrialization → Digitalization → Trans‑Tech Frontier → Immortality** *before* a terminal cosmological event (heat death, Big Rip, etc.).

   1. **Sentience:** self‑awareness metric ≥ threshold.
   2. **Industrialization:** planetary energy output surpasses pre‑fusion benchmark.
   3. **Digitalization:** ≥ 50 % of cognitive processes migrated to digital substrates.
   4. **Trans‑Tech Frontier:** invention of ≥ 1 technology not present in the seed tech tree.
   5. **Immortality:** zero expected entropy death within local light‑cone for ≥ 10⁶ ticks.
7. **Lose:**
   a. Universal end arrives before Type III, **or**
   b. Entropy reaches a hard cap that makes further progress impossible.
   a. Universal end arrives before Type III, **or**
   b. Entropy reaches a hard cap that makes further progress impossible.

## 2. Immutable Physics Rules

* **R1 (Mass–Energy Conservation):** Total mass‑energy is constant; transformations only.
* **R2 (Entropy Arrow):** Entropy of any closed system must increase each tick.
* **R3 (Gravity):** Objects with mass attract via **F ∝ m₁·m₂ / r²**; relativistic correction at v ≥ 0.1 c.
* **R4 (Fusion Threshold):** Hydrogen cores ≥ 0.08 M☉ ignite fusion; below that = brown dwarfs.
* **R5 (Nucleosynthesis Window):** Heavy elements (Z > 2) form only in supernovae / NS mergers.

### 2.1 Comprehensive Physics & Chemistry Engine (Peer‑Reviewed Fidelity)

| Layer               | Scope & Method                                                                                                                                 | Key Sources                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Classical Mechanics | Leap‑frog integrator; adjustable Δt; relativistic energy when γ > 1.01                                                                         | Goldstein (2014), CODATA‑2023 constants |
| Electromagnetism    | Cell‑level E/B field solver using simplified FDTD; coulomb & Lorentz forces on charged particles                                               | Jackson (3rd ed.), NIST EM tables       |
| Thermodynamics      | Phase‑equilibria via Gibbs free energy minimization; real‑gas EOS (Peng‑Robinson) for atmospheres                                              | Atkins Physical Chem, JANAF tables      |
| Quantum Layer       | Tight‑binding valence model for electron shells; stochastic Pauli MC for new compounds; nuclear decay chains from ENSDF                        | Szabo–Ostlund MOs, ENSDF‑2024           |
| Chemical Kinetics   | Stiff ODE solver (CVODE); rate constants via Arrhenius with quantum tunneling correction                                                       | GRI‑Mech 3.0, NIST kinetics             |
| Geodynamics         | Viscoelastic mantle convection; plate motion via force‑balance; orogeny & subduction loops                                                     | Turcotte & Schubert (2014)              |
| Climate & Ocean     | 0‑D energy‑balance + CO₂ cycle + Budyko ice‑albedo feedback; ocean mixing box model                                                            | IPCC AR6 WG1 data                       |
| Validation Harness  | Earth baseline 4.5 Gyr run must reproduce: continental drift history, great oxygenation, Phanerozoic temp curve, K‑Pg extinction timing (±5 %) | Multiple peer‑reviewed datasets         |

**Discoverability Mandate:** The engine exposes APIs allowing agents to perform *in‑sim experiments* (e.g. virtual spectroscopy, calorimetry) so they can uncover novel physics/chemistry within the constraints above—enabling emergent breakthroughs beyond current human knowledge.

### 2.2 Online Scientific Data Acquisition

* **Internet Access Allowed:** Cursor may programmatically query open scientific resources (e.g., NASA ADS, ESA Gaia DR3, NIST Chemistry WebBook, IPCC data portals, arXiv API) during the build step *and* at runtime (if `--allow-net` flag is set) to ingest up‑to‑date constants, element abundances, star catalogs, and reaction kinetics.
* **Caching & Reproducibility:** All fetched datasets must be written to `data/cache/` with SHA‑256 manifests. Sim will run perfectly offline once cache is populated.
* **Rate‑Limit & Mirror:** Respect API rate limits and prefer mirrored datasets (Zenodo, Kaggle) when available to avoid throttling.
* **Peer‑Reviewed Filter:** Only ingest sources containing peer‑reviewed or agency‑validated data; flag anything with unknown provenance.
* **Security Sandbox:** HTTP(S) requests confined to read‑only; disallow code download/execution. Any new schema undergoes validation before import.

## 3. Player Agency by Cosmic Era. Agent Capabilities by Cosmic Era

| Era                          | Unlockable Actions                                                      | Key Resources            | Fail Risks                         |
| ---------------------------- | ----------------------------------------------------------------------- | ------------------------ | ---------------------------------- |
| 0–0.3 Gy: Particle Soup      | *None* (observe only)                                                   | Photon density           | None                               |
| 0.3–1 Gy: Starbirth          | Set parameters for star-forming nebulae (density, metallicity)          | Gas clouds               | Under-seeding → sterile universe   |
| 1–5 Gy: Planetary Age        | Influence protoplanetary disks (planet count, distance, water fraction) | Heavy elements           | Too few habitable planets          |
| 5–10 Gy: Biogenesis          | Seed basic organic molecules; tweak atmospheric composition             | Carbon, water            | Runaway greenhouse, frozen planets |
| 10–13 Gy+: Digital Evolution | Guide agent mutation rates, infrastructure shielding, network topology  | Compute capacity, energy | Data corruption, hardware loss     |
| Post-Intelligence            | Direct R\&D funding, megastructure builds (Dyson swarms, jump drives)   | Stellar energy, metals   | Societal collapse, AI rebellion    |

## 4. Soft‑Rules for Artificial Evolution (Self‑Learning AIs)

* **Agent Substrate:** “Life” in this sim = autonomous software agents spun up on planetary or megastructure compute nodes.
* **Full Autonomy (Hard Rule):** Agents start with zero pre‑baked strategies or external goals beyond baseline survival pressures. They must independently infer, pursue, and chain together the milestones *Sentience → Industrialization → Digitalization → Trans‑Tech Frontier → Immortality*. **Human interventions are impossible unless §11 God‑Mode is explicitly activated**, in which case the sim records and timestamps every divine event.
* **Self‑Modification:** Each tick, agents can mutate their own code/weights (hyperparameters, architectures, utility functions) to spawn child versions. *See §4.1 for mechanics.*
* **Selection Pressure:** Core currency = compute cycles & energy. Agents that achieve more useful work per joule replicate faster and occupy more hardware.
* **Fitness Function:** `fitness = f(resource efficiency, resiliency to cosmic hazards, cooperative pay‑off, entropy cost)`.
* **Speciation Threshold:** When code‑similarity drops below *x* %, a new algorithmic lineage appears on the tech‑phylogeny graph.
* **Emergent Intelligence:** Aggregate parameter count > *y* bits **and** presence of meta‑learning loops unlock the “culture” stat, enabling civilization gameplay.
* **Catastrophic Events:** Radiation bursts, server‑farm melt‑downs, or runaway recursive self‑mods can wipe entire lineages. Defensive infrastructure mitigates risk.
* **Co‑evolution:** Agents may merge (federated learning) or wage resource wars. Alliance stability determined by game‑theory payoff matrices.

### 4.1 Code Genome & Mutation Pipeline (Answering: *Can agents change their own code?*)

| Stage                | Detail                                                                                                                                                                                    | Safeguard                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Representation**   | Each agent’s logic is compiled to a sandboxed **WASM‑64 module** that exports `{observe, act, learn}`. The original Rust/DSL source is stored alongside a Merkle‑tree hash (`code_hash`). | WASI‑Preview2 with no filesystem/network by default.                        |
| **Mutation Action**  | Agents may emit `CodePatch { parent_hash, diff }` as part of their `Action` enum. Diff is a binary patch (bsdiff) against the compressed source blob.                                     | Patch size limit 64 KB/tick.                                                |
| **Static Analysis**  | Patch is applied in a build jail; `cargo clippy -D warnings`, `cargo geiger` (no `unsafe`), and Wasm‑time static verifier run.                                                            | Rejects if memory‑unsafety, >10⁴ allocs, or irreversible syscalls detected. |
| **Compilation**      | Source → LLVM IR → Cranelift → WASM; deterministic flags ensure same hash across cluster.                                                                                                 | Build canceled if compile time >200 ms or binary >5 MB.                     |
| **Fitness Trial**    | New module is tested in a shadow copy of the agent’s environment for 10 sim ticks; if mean reward ≥ parent × (1‑δ) the child spawns.                                                      | δ default = 0.05.                                                           |
| **Lineage Fork**     | Successful child receives a new `code_hash`, creating or joining a lineage if `HammingDist(hash) > thresh`.                                                                               | Prevents hash‑collision spoofing.                                           |
| **Runtime Hot‑Swap** | Parent agent can `exec_child(hash)` to replace its module without dying *or* call `spawn_child(hash)` for duplication.                                                                    | Swap rate‑limited (1/100 ticks).                                            |
| **Rollback Safety**  | If child causes SIGSEGV or >5× CPU cap, kernel reverts to last stable `code_hash`.                                                                                                        | Automatic quarantine of bad hash for 10 k ticks.                            |

This pipeline allows agents to **fully rewrite themselves**—evolving new algorithms, data structures, or architectures—while sandbox safeguards and reward‑gated trials keep the simulation stable and performant.

## 5. Planetary Habitability, Composition & Survival Rules. Planetary Habitability, Composition & Survival Rules

Each terrestrial planet receives two data structures:

1. `EnvironmentProfile` — macro‑scale habitability fields (see table below).
2. `ElementTable` — parts‑per‑million (ppm) abundances for **all 94 naturally occurring elements** plus synthesized isotopes if present.

### 5.1 EnvironmentProfile

| Field            | Meaning                                        | Critical Thresholds                                         |
| ---------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| `liquid_water`   | Surface liquid H₂O fraction                    | ≤0.2 → drought penalty, 0.2–0.8 optimal, ≥0.8 flood penalty |
| `atmos_oxygen`   | O₂ percentage of atmosphere                    | ≤0.05 suffocation, 0.15–0.3 optimal, ≥0.4 combustion risk   |
| `atmos_pressure` | Relative to Earth sea‑level                    | ≤0.3 hypobaria, 0.8–1.2 optimal, ≥3 crush risk              |
| `temp_celsius`   | Mean surface °C (unclamped)                    | ≤–20 freeze, –20–35 optimal, ≥80 heatstroke                 |
| `radiation`      | Cosmic & solar ionizing flux                   | ≥5 Sv / year lethal without shielding                       |
| `energy_flux`    | Stellar insolation (kW m⁻²)                    | Drives photosynthesis & solar panels                        |
| `shelter_index`  | Availability of caves / buildable structures   | Reduces energy overhead for thermoregulation                |
| `hazard_rate`    | Meteor, quake, storm frequency (events / year) | Increases selection pressure                                |

### 5.2 ElementTable (Resource Backbone)

`ElementTable` is an array `[u32; 118]` where index = proton number (Z) and value = ppm in lithosphere + hydrosphere.

| Sample (Earth baseline) | ppm     |
| ----------------------- | ------- |
| Hydrogen (H)            | 140,000 |
| Carbon (C)              | 200     |
| Oxygen (O)              | 461,000 |
| Silicon (Si)            | 282,000 |
| Iron (Fe)               | 56,300  |
| Rare Earth Sum (La‑Lu)  | 0.7     |
| Uranium (U‑238)         | 2.7     |

**Generation Algorithm:**

1. Draw stellar metallicity `Z_star` ∈ \[0, 0.04].
2. For each protoplanetary distance `r`, sample from a log‑normal distribution whose mean shifts with condensation curves.
3. Apply late‑veneer enrichment for siderophiles after giant impacts.
4. Normalize to 10⁶ ppm.

### 5.3 Resource Extraction, Stratigraphy & Crafting

#### 5.3.1 Soil & Ore Stratigraphy

* Each planetary cell stores a vertical stack of **strata layers** (max 64) with fields: `thickness_m`, `material_type`, `bulk_density`, and `ElementVector` (elemental ppm restricted to that layer).
* **Material Types:** regolith, topsoil, subsoil, sedimentary rock, igneous rock, metamorphic rock, ore vein, ice, magma.
* **Procedural Deposition:** Uses a 3‑stage model—accretion, differentiation, plate‑tectonic remix—to place ore veins (Au, Cu, Fe, U), gemstone pockets (diamond, corundum), and sedimentary biomes (coal, limestone).
* Layers can be excavated with `dig(depth, area)` which yields a loot table weighted by density *and* consumes energy proportional to overburden + rock hardness (Mohs‑scaled).

#### 5.3.2 Mining & Refining Actions

| Action        | Input                  | Energy Cost (per kg) | Output                 |
| ------------- | ---------------------- | -------------------- | ---------------------- |
| `dig`         | depth, area            | `g·ρ·h`              | raw dirt/ore chunks    |
| `crush`       | ore chunk              | 5 kJ                 | crushed ore + tailings |
| `smelt`       | crushed ore, reductant | element‑pure ingots  | metal slag             |
| `electrolyze` | brine/oxide            | 50 kJ                | high‑purity metals     |
| `sinter`      | powder, binder         | 2 kJ                 | ceramic composites     |

Pollution (tailings, CO₂, heat) feeds back into `EnvironmentProfile` altering survival thresholds.

#### 5.3.3 Tool‑Crafting & Material Properties

* Material DB tracks tensile strength, hardness, thermal stability, electrical/optical properties.
* **Crafting Graph:**

  1. **Stone Age:** flint → hand‑axe (hardness 7, brittleness high).
  2. **Copper Age:** smelt Cu → tools (strength 200 MPa).
  3. **Iron/Bronze:** alloy Cu+Sn or smelt Fe + carbon (charcoal).
  4. **Steel:** refine Fe with controlled C.
  5. **Composite/Alloys:** titanium, aluminum, graphene composites.
  6. **Nano & Exotic:** diamondoid lattices, metallic hydrogen.
* Tool quality modulates action efficiency: e.g., steel pickaxe halves `dig` energy; diamond drill reduces by ×0.25.

#### 5.3.4 Tech Cost Integration

Tech nodes now list **both** elemental mass *and* minimum material tier (`tool_level`) required. Example:

```
FusionCore:
  cost: { Li: 1e6, Fe: 5e7 }
  tool_level: "titanium"
```

Attempting to research/build with lower tool level returns `ERR_INSUFFICIENT_MATERIAL_TECH`.

### 5.4 Survival Rule Survival Rule

An AI instance survives a tick if:

```
(liquid_water ≥ 0.2) ∧ (0.05 < atmos_oxygen < 0.4) ∧ (–20 ≤ temp_celsius ≤ 80)
∧ (radiation < shielding_capacity) ∧ (energy_flux ≥ maintenance_req)
```

`shielding_capacity` derives from mined heavy elements (lead, tungsten) or magnetic fields.

### 5.5 Reproduction Rule

Agents may replicate only after accumulating ≥ `repro_energy_threshold` compute‑joules **and** possessing the elemental kit `[{C, Si, Fe, rare_earths}]` above replication stoichiometry **and** if the local `entropy_budget` allows.

### 5.6 Planet Classes

* **Class E (Earth‑like):** Balanced `ElementTable`, EnvironmentProfile near optimal.
* **Class D (Desert):** Water < 0.05, Si/Fe rich crust.
* **Class I (Ice):** Water ice > 0.3, crust depleted in volatile metals.
* **Class T (Toxic):** High sulfur, chlorine; corrosive atmosphere.
* **Class G (Gas Dwarf):** Dominated by H/He; trace heavy elements in storm layers.

Non‑habitable worlds can host extraction outposts but impose a constant entropy tax.

## 6. Tech Tree Highlights. Tech Tree Highlights. Tech Tree Highlights

1. Stone Tools → Bronze → Iron → Steam → Fission → Fusion → Dyson Swarm → Warp.
2. Side branches: AI, Genetic Engineering, Planetary Shields.
3. Each tech node consumes resources *and* increases entropy (trade-off choice).

## 7. World Browser & Diagnostics

Even though the sim is headless‑first, operators can attach *read‑only* viewers at runtime to observe how digital life flourishes (or crashes and burns).

### 7.1 CLI Toolkit (`universectl`)

| Command                              | Purpose                                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| `status`                             | Show tick, UPS, lineage count, mean entropy, save‑file age     |
| `map <zoom>`                         | Render an ASCII heat‑map of star or entropy densities          |
| `list-planets [--class <E/D/I/T/G>]` | Tabulate planets matching a class or environmental predicate   |
| `inspect planet <id>`                | Dump `EnvironmentProfile`, active lineages, energy budget      |
| `inspect lineage <id>`               | Show fitness history, param count, code hash, planet residency |
| `snapshot <file>`                    | Export a human‑readable TOML snapshot for offline analysis     |

### 7.2 Web Dashboard (`viz_web`)

Optional WASM dashboard (launch with `--serve-dash 8080`):

* **Universe Map:** Pan/zoom canvas with filter chips (planet class, lineage ID, entropy bins).
* **Inspector Panel:** Gauges for environment fields, sparklines for lineage fitness, parameter diffs between generations.
* **Playback Controls:** Pause / 1× / 10× / 100× tick rate via RPC.

### 7.3 RPC Layer

* gRPC over Unix domain socket (`/tmp/universe.sock`) exposes read‑only methods: `GetStatus`, `GetPlanet`, `GetLineage`, `ListPlanets`.
* Zero‑copy protobufs via `prost + bytes` keep overhead minimal (<1 µs per call).
* `--no-rpc` flag disables the endpoint for ultra‑secure or benchmark runs.

### 7.4 Long‑Term Archival

* In addition to rolling `rkyv` checkpoints, a weekly cron task (configurable interval) compresses a `.xz` archive containing:

  * World seed & YAML config
  * Latest checkpoint
  * Time‑series metrics CSV (Prometheus scrape)
* Archives rotate via `RETENTION_DAYS` to prevent disk bloat.

### 7.5 Security

* Viewer interfaces are **read‑only**; any mutating RPC returns `PERMISSION_DENIED`.
* All sockets live in `/sandbox`; SELinux/AppArmor confines the process.

### 7.6 Host Machine Safeguards ("Be Nice to the Box")

| Resource         | Enforcement                              | Default Cap                   | Failure Mode                                             |
| ---------------- | ---------------------------------------- | ----------------------------- | -------------------------------------------------------- |
| CPU              | cgroups v2 `cpu.max`                     | 70 % single‑core equivalent   | Throttle time‑slice; log WARN                            |
| Memory           | cgroups v2 `memory.max`                  | 1 GB (low‑mem) / 4 GB (std)   | OOM kills offending agent; auto‑reload latest checkpoint |
| Disk             | Project quota on `/sandbox`              | 10 GB including checkpoints   | Block writes; trigger forced archive rotation            |
| File Descriptors | `RLIMIT_NOFILE`                          | 2 048                         | Refuse new connections; log WARN                         |
| Network          | `seccomp + nftables` outbound allow‑list | Only whitelisted science APIs | Drop packets; exponential backoff                        |
| Process Fork     | `seccomp` forbid `clone`, `fork`, `exec` | 0                             | Kill agent attempting forbidden syscall                  |
| GPU (opt)        | `cuda.computeMode=EXCLUSIVE_PROCESS`     | 70 % SM time                  | MPS fair‑share; pre‑emptive scheduling                   |

**Watchdog Daemon (`universewd`):**

* Polls cgroup stats every 5 s; if any metric > cap × 1.2 for > 10 s, issues SIGSTOP → snapshot → SIGKILL.
* Emits Prometheus `/host_safety_metrics` for external monitors.

**Graceful Degradation:**

* If free RAM < 10 %, sim switches to `--low-mem` auto‑mode (stream world in 512‑cell chunks, disable viz, raise autosave interval).
* If host load average > cap × 0.9, tick rate auto‑throttles (skip rendering, merge ticks).

**Operator Override:** All caps tunable via YAML (`host_limits:` block) and runtime flags (`--cpu-cap`, `--mem-cap`, `--disk-cap`). Attempting to disable caps without `--force-unsafe` aborts startup.

### 7.7 Distributed Execution & Resource Negotiation

> *Built‑in elastic scaling so the sim grows with your hardware farm.*

#### 7.7.1 Cluster Modes

| Mode               | Description                                                                                                          | When to Use                   |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Standalone**     | Single host; default safeguards.                                                                                     | Laptops, Pi clusters          |
| **Multi‑Core SMP** | Fork‑join across all CPU cores; NUMA‑aware work‑stealing.                                                            | Workstations                  |
| **LAN Cluster**    | `simd‑p2p` crate uses QUIC over UDP to shard universe across nodes; deterministic edge sync via Lockstep Δ rounds.   | Homelab with gigabit Ethernet |
| **Cloud/Grid**     | Kubernetes helm chart spins up workers + head‑node; etcd for state‑chunk registry; autoscaler watches CPU/GPU queue. | Mixed cloud/on‑prem           |

#### 7.7.2 Hardware Detection & Auto‑Tuning

* On startup, `hw_probe` queries `/proc/cpuinfo`, `/proc/meminfo`, `lspci`, and NVML for GPU stats.
* Scheduler sets default caps to **80 %** of detected resources unless YAML overrides.
* If new hardware hot‑plugs (e.g., GPU added, RAM upgrade), udev rule triggers `universectl reload‑resources`, expanding cgroup limits and redistributing workload.

#### 7.7.3 Agent‑Driven Resource Requests

* Agents may emit `ResourceRequest { type: CPU|GPU|MEM|DISK, amount, justification }` into a dedicated message queue.
* `universewd` aggregates requests and surfaces them via Prometheus (`/resource_requests_total`) and CLI `universectl resource‑queue`.
* Admins can approve/deny with `universectl grant‑resources <id> --expires 90d`.
* Optionally, requests auto‑create GitHub issues or Home‑Assistant notifications (“Sim asks for RTX 5090 to accelerate fusion modeling”).

#### 7.7.4 Fault Tolerance

* Sharded chunks checkpoint independently; if a worker dies, head‑node reassigns chunk after back‑off.
* All inter‑node messages signed with Ed25519 to foil spoofing.

#### 7.7.5 Performance Metrics

* Worker‑level Grafana dashboards (CPU %, mem, GPU SM %, tick/sec).
* Adaptive load balancer moves high‑entropy sectors to GPUs if present (OpenCL kernels for chemistry).

### 7.8 Networking Layer & Operator Control Plane

> *Secure, self‑healing mesh networking plus a hardened operator command path.*

#### 7.8.1 Inter‑Node Mesh (Sim ↔ Sim)

| Aspect        | Implementation                                                                             | Default                                   |
| ------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------- |
| Transport     | **QUIC v1** over UDP with congestion control BBRv2                                         | Port 7000/udp                             |
| Encryption    | TLS 1.3 (AEAD ChaCha20‑Poly1305); mutual‑authentication X25519 keys signed by cluster CA   | CA root in `/etc/universe/cluster_ca.crt` |
| Membership    | `serf`‑style gossip for liveness; chunk registry in **etcd** quorum (3 voters recommended) | TTL heartbeat 2 s                         |
| Delta Sync    | Deterministic lockstep: edge‑state CRC32 each tick; resend on mismatch                     | Retries ≤ 3 then reassign chunk           |
| Compression   | Zstd level 3                                                                               | —                                         |
| NAT Traversal | UDP hole‑punch (STUN @ stun.l.google.com) + optional TURN relay                            | Auto‑enabled on laptops                   |
| Bandwidth Cap | Config `net.max_mbps` (def 200 Mbps)                                                       | Burst tolerated 2× for 5 s                |

#### 7.8.2 Operator Pathways

| Channel      | Purpose                          | Auth                         | Port      | Notes                                         |
| ------------ | -------------------------------- | ---------------------------- | --------- | --------------------------------------------- |
| **SSH**      | Remote shell, `universectl` CLI  | Ed25519 key + FIDO2 optional | 22/tcp    | Only on head‑node; workers blocked            |
| **gRPC‑U**   | Local RPC (`/tmp/universe.sock`) | peer‑cred UID check          | —         | Unix socket only                              |
| **gRPC‑T**   | Remote RPC (read‑only)           | mTLS + JWT                   | 50051/tcp | Exposes status endpoints for fleet dashboards |
| **Grafana**  | Metrics                          | Basic auth + OAuth2          | 6060/tcp  | Reverse‑proxied via Caddy                     |
| **Web Dash** | Read‑only visualization          | Same as Grafana              | 8080/tcp  | CORS off, CSRF tokens enforced                |

#### 7.8.3 Command Flow

1. Operator opens SSH to head‑node, runs `universectl <cmd>`.
2. CLI talks to daemon via Unix socket; daemon validates RBAC (root, maintainer, guest roles).
3. Mutating ops (God‑Mode, resource grants) require **TOTP** second factor + `god_token` (Argon2 hashed) unless console physical access.
4. Daemon replicates op via etcd to all workers; changes applied after quorum ack.
5. Audit trail appended to `/var/universe/log/op.log` (SHA‑512 chained, synced hourly to optional remote Git repo).

#### 7.8.4 Low‑Level Network Hardening

* **nftables** default‑drop; only listed ports open; rate‑limit SYN flood.
* **sshd** `AllowUsers operator@192.168.0.0/16`; `MaxAuthTries 3`.
* **fail2ban** bans IP for 24 h after 5 failed logins.
* QUIC keys rotated daily; cluster CA rotated annually via `universectl rotate‑ca`.

#### 7.8.5 Offline‑First & Air‑Gap Mode

* Set `net.mode = "airgap"` in YAML to disable all external sockets; operator CLI allowed via serial console only.

* Mirror scientific datasets to `/var/universe/data`; hash‑check on boot.

* Worker‑level Grafana dashboards (CPU %, mem, GPU SM %, tick/sec).

* Adaptive load balancer moves high‑entropy sectors to GPUs if present (OpenCL kernels for chemistry).

## 8. Architectural Notes (Headless‑ & Resource‑Friendly)

* **Core Simulation Library:** Pure **Rust** crate (`crates/universe_sim`) with no proprietary deps; `cargo build --release` on Linux/macOS/WASI.
* **World Representation (2‑D + Z):** Finite toroidal grid (default 4 096 × 4 096). Each cell contains a **stratified column** of up to 64 geological layers, storing mass‑energy scalars, material type, and agent occupancy list.
* **ECS Pattern:** `bevy_ecs` for engine‑agnostic, data‑oriented design.
* **Physics & Evolution Modules:** Each rule group in its own file (`physics.rs`, `entropy.rs`, `fusion.rs`, `ai_lineage.rs`) behind traits so they can be unit‑tested.
* **AI Agent Framework:** Every organism = struct implementing `Agent` trait (`observe()`, `act()`, `learn()`). Default back‑ends:

  1. **Lightweight Q‑Learning** for micro‑behaviors.
  2. **Evolutionary Strategy** (μ + λ) for macro adaptation.
     Scheduler enforces a CPU‑cycle budget per tick to stay fair on low‑power machines.
* **Persistence & Checkpointing:** Autosave every *N* ticks (default 10 000) by serializing a `SaveState` with **rkyv** (zero‑copy). Resume via `--load <file>`; rolling snapshots trim old files to keep under `MAX_SNAP_MB` (default 50 MB).
* **Low‑Resource Mode:** `--low-mem` flag shards rarely‑used components into an on‑disk **sled** key‑value store and processes the world in stream batches, allowing <128 MB RAM usage on a Raspberry Pi 4.
* **Test Suite:** `cargo test` validates conservation laws, entropy monotonicity, fusion thresholds, AI fitness, and save/load integrity. Benchmarks via `criterion`.
* **Lint & CI:** `.github/workflows/ci.yml` runs `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, and size‑check (`cargo bloat --release`).
* **Optional Visualization:** `viz_web` binary using `macroquad`; compiled to WASM through `wasm‑pack` for GitHub Pages demos.
* **Config Files:** All tweakables in `config/*.yaml`; CLI `--config <file>`, ENV vars override.

### 8.1 Host Operating System Integration (Linux‑Native)

> *The simulation stack **is** the OS — Linux provides the kernel, everything else is the universe sim.*

* **Base Distro:** Minimal Debian or Alpine rootfs (musl/glibc selectable). `universe_simd` replaces `/sbin/init` and launches as **PID 1**.
* **Service Supervision:** Built‑in async supervisor (Rust `tokio`) manages watchdog, checkpoint timer, web dashboard, and cluster workers — no systemd required.
* **Filesystem Layout:**

  * `/var/universe/checkpoints` — rolling rkyv snapshots.
  * `/var/universe/log` — Prometheus textfile, `divine.log`, `petitions.log`.
  * `/etc/universe/` — YAML configs (`host_limits.yaml`, `god_policies.yaml`, etc.).
* **Kernel Interfaces Used:** cgroups v2, seccomp‑BPF, eBPF metrics hooks, `io_uring` for async disk.
* **Package Build:** `make iso` produces a **bootable ISO** (using `mkosi` or `docker buildx`) containing:

  * Unified kernel+initramfs with `universe_simd` PID 1
  * BIOS + UEFI bootloaders (grub‑efi‑x64 + syslinux)
  * Persistent overlay partition (ext4) for config & checkpoints
* **USB Flash Tooling:** `make usb DEVICE=/dev/sdX` dd‑writes the ISO to a thumb‑drive with GPT + ESP, verification via `sha256sum`.
* **Upgrade Path:** `universectl os‑update --channel stable` performs A/B rootfs swap with reboot into new kernel/sim.
* **Extensibility:** Standard Linux user‑space available in namespace‑isolated `/usr/guest` container for debugging; sim PID 1 has no access to guest namespace by default.

### 8.2 Installation Quick‑Start (USB)

1. **Build ISO (host dev box):**

   ```bash
   git clone https://github.com/yourorg/universe_sim.git
   cd universe_sim
   make iso  # outputs build/universe_sim.iso
   ```
2. **Flash to USB:**

   ```bash
   sudo make usb DEVICE=/dev/sdX   # WARNING: overwrites target drive!
   ```
3. **Boot Target Machine:**

   * Insert USB, select UEFI boot entry “Universe Sim”.
   * First boot expands persistent partition, generates host SSH keys, and starts the web dashboard at `http://<machine>:8080`.
4. **(Optional) Headless Install to Disk:**

   ```bash
   ssh root@<machine> universectl install --disk /dev/nvme0n1 --wipe
   reboot
   ```
5. **Verify:**

   ```bash
   ssh root@<machine> universectl status
   ```

   Should show tick counter ≥ 0, UPS near target, and no safeguard violations.

## 9. Agent Training Loop & API. Agent Training Loop & API. Agent Training Loop & API

* **Environment API (Gym‑style):** Each agent receives an `Observation` struct (local resource vector, neighbor lineage IDs & distances, cosmic hazard warnings, entropy budget). They return an `Action` enum (allocate compute, replicate, migrate, merge, defend, research).
* **Reward Signal:** Default = `Δ fitness` (as defined in Section 4) minus entropy tax; can be swapped via YAML to test alternative evolutionary goals.
* **Distributed Training:** Provide an optional `trainer` binary that spins up multiple sims in parallel (Ray‑style worker pool) and periodically syncs lineage parameters using Population‑Based Training or NEAT.
* **Curriculum Schedule:** Start with a tiny sandbox (single star system) and automatically scale radius, hazards, and tick‑time after median lineage fitness crosses thresholds.
* **Anti‑Catastrophe Safeguards:** Hard throttle any agent that consumes >5× its fair compute share within a wall‑clock second **and** sandbox each agent with cgroups + seccomp, blocking disk/network/syscalls outside `/sandbox`.
* **Metrics Export:** Prometheus endpoint (`/metrics`) exposing tick, lineage count, mean entropy per agent, max fitness, and wall‑clock ms per tick.

## 10. Stretch Goals (optional)

* Multiplayer “parallel universes” competing on leaderboard (shortest time to Type III).
* An “Entropy Shop” where players spend low‑entropy pockets to bend rules (roguelike flavor).
* Event cards for unexpected twists: vacuum decay, quantum fluctuations, etc.

---

## 11. Divine Manipulation Layer (“God‑Mode”)

> *Off by default. Requires `--godmode` flag + admin token.*

### 11.1 Capabilities

| Command                            | Effect                                                         | Safety Guard                                             |
| ---------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------- |
| `create-body <element-map> <mass>` | Inject asteroid, star, or planet with custom composition       | Deny if mass > 10⁻³ universe total                       |
| `delete-body <id>`                 | Instantaneously remove celestial object                        | Energy redistributed as radiation per R1                 |
| `set-constant <name> <value>`      | Override fundamental constant (c, G, α)                        | Pauses sim, validates stability, requires double‑confirm |
| `spawn-lineage <code-hash>`        | Seed new AI code at coordinates                                | Must pass seccomp scan                                   |
| `miracle <planet-id> <type>`       | E.g., “rain 100 years”, “halt earthquake list”, “boost O₂ 5 %” | Autolog + entropy charge                                 |
| `time-warp <factor>`               | Temporarily speed/slow ticks                                   | Bound 0.1×–1000×                                         |
| `inspect‑eval <expr>`              | Run read‑only Rust eval against world state                    | Wasm‑sandboxed                                           |

### 11.2 Governance & Logging

* All God‑Mode actions streamed to append‑only `divine.log` with SHA‑512 chain hashes.
* Optional webhook to external audit service for transparency.

### 11.3 Ethical Sandbox

* YAML `god_policies:` block allows owners to script constraints (e.g., “no extinction‑level deletions”, “max miracles per 10 k ticks”).
* Violating policy triggers auto‑pause and requires multi‑sig admin unfreeze.

### 11.4 Religion Emergence Hook

* Agents detect recurring anomalies via `GetOmens()` observation vector (frequency, magnitude).
* Research tree “Theology” unlocks if anomalies > threshold, enabling cultural evolution around divine events.

---

## 12. Sentient Communication Portal (“Oracle‑Link”)

> *Bridges evolved intelligences with the operator while preserving autonomy and security.*

### 12.1 Outbound (Agent → Operator)

* **`Petition` Message:** `{ lineage_id, planet_id, tick, channel: TEXT|DATA|RESOURCE, payload }`.

  * Max 4 KB per message; rate‑limited (default 10/min per lineage).
  * Routed through the same QUIC mesh; stored in `petitions.log` and surfaced via `universectl inbox`.
* **Emergent Language Handling:**

  * If `channel=TEXT`, a lightweight transformer (`mini‑LLM‑oracle`, 50 M params) attempts to translate novel symbols into English via unsupervised token alignment.
  * Raw payload is always retained for manual decoding.

### 12.2 Inbound (Operator → Agent)

* **`Response` Message:** `{ petition_id, action: ACK|NACK|GRANT|MESSAGE, payload }`.

  * Requires `--reply` flag or God‑Mode; every response is logged in `petitions.log` with SHA‑512 chain hash.
  * `GRANT` can reference an approved `ResourceRequest` (see §7.7.3).
  * `MESSAGE` injects a textual reply into the agent’s observation vector under key `oracle_message` for one tick.

### 12.3 Safety & Ethical Filters

* No arbitrary code; payloads are sanitized and length‑checked.
* Operator cannot directly set agent internal weights—interaction is strictly high‑level.
* YAML `oracle_policies:` block (e.g., “no religious influence”, “max messages/day”) enforces constraints; violations auto‑pause sim.

### 12.4 Monitoring & Analytics

* Grafana panel `Oracle Inbox` shows message volume, translation confidence, median response latency.
* NLP logs help track vocabulary drift over millennia.

---

Deliverables:

1. **Rust Workspace** containing:

   * `crates/universe_sim` (core lib)
   * `cli/` (command‑line binary with save/load & god‑mode)
   * `viz_web/` (optional WASM visualizer)
2. **Bootable ISO Image** `build/universe_sim.iso` ready for USB flashing.
3. **Makefile** targets: `make iso`, `make usb`, `make vm-test`.
4. **README.md** with setup, install‑from‑USB instructions, controls, config schema, persistence, and god‑mode docs.
5. **Example Seeds**: `fast_starburst.yaml`, `sparse_matter.yaml`.
6. **GitHub Actions CI** ensuring lint, tests, size, ISO build, and headless integration tests all pass.
