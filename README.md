# E2EVRP – Electric Two-Echelon Vehicle Routing Problem with RL

End-to-end pipeline for solving and *learning to solve* the **E2EVRP** — from instance parsing through an ALNS metaheuristic to a GNN-backed MDP environment ready for offline reinforcement learning.

> Breunig, U., Baldacci, R., Hartl, R. F., & Vidal, T. (2019).
> *The Electric Two-echelon Vehicle Routing Problem.*
> Computers & Operations Research, 103, 198–210.

---

## Repository Structure

```
E2EVRP_Instances/
│
│  ── Core Pipeline ──────────────────────────────────────
├── e2e_vrp_loader.py       # Instance parser & data structures
├── lns_solver.py            # ALNS solver + hetero-graph state extraction
├── gnn_encoder.py           # Heterogeneous GNN encoder (PyG / fallback)
├── lns_env.py               # Gym-style MDP environment for operator selection
│
│  ── Tooling ────────────────────────────────────────────
├── benchmark_runner.py      # Multi-run benchmark suite + offline data generator
├── visualizer.py            # matplotlib solution plotter
├── test_pipeline.py         # 59-test end-to-end validation suite
│
│  ── Data ───────────────────────────────────────────────
├── Set2/ … Set8/            # Instance .dat files (Breunig et al.)
├── E2EVRP_Instances.txt     # Instance file format specification
├── benchmark_results/       # Plots, CSV summary, offline trajectories
└── README.md
```

---

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  .dat files  │────▶│ e2e_vrp_     │────▶│ lns_solver   │────▶│ lns_env      │
│  (Set2–8)    │     │ loader.py    │     │ .py          │     │ .py          │
│              │     │              │     │              │     │              │
│ Instance data│     │ Parse into   │     │ ALNS solver  │     │ Gym-style    │
│              │     │ typed objects│     │ + graph state│     │ MDP wrapper  │
└──────────────┘     └──────────────┘     └──────┬───────┘     └──────┬───────┘
                                                 │                    │
                                       get_hetero_state()      (S, A, R, S')
                                                 │                    │
                                                 ▼                    ▼
                                          ┌──────────────┐   ┌──────────────┐
                                          │ gnn_encoder  │   │ benchmark_   │
                                          │ .py          │   │ runner.py    │
                                          │              │   │              │
                                          │ Fixed-size   │   │ Offline RL   │
                                          │ state vector │   │ data (.pkl)  │
                                          └──────────────┘   └──────────────┘
```

---

## Module Details

### 1. `e2e_vrp_loader.py` — Instance Parser

Parses `.dat` instance files into typed Python dataclasses.

| Data Structure         | Fields                                         |
|------------------------|-------------------------------------------------|
| `TruckConfig`          | count, capacity, cost_per_distance, fixed_cost  |
| `CityFreighterConfig`  | max_per_satellite, total_count, capacity, cost_per_distance, fixed_cost, max_charge, energy_consumption |
| `Depot / Satellite / Customer / RechargingStation` | id, x, y + type-specific fields |
| `E2EVRPInstance`        | Aggregates all of the above                     |

Node IDs are sequential: `0` = Depot, `1..S` = Satellites, `S+1..S+C` = Customers, remainder = Recharging Stations.

---

### 2. `lns_solver.py` — ALNS Solver + State Extraction

Decomposes the E2EVRP into two echelons:

**Level 2 (Satellites → Customers):** Solved per-satellite with Adaptive LNS.

| Destroy Operators | Repair Operators |
|---|---|
| Related Nodes Removal (seed + k-nearest) | Greedy Cheapest Insertion |
| Random Routes Removal (up to 37% of routes) | Regret-2 Insertion |
| Close Satellite (removes half the routes) | |

**Level 1 (Depot → Satellites):** Aggregated satellite demands routed via nearest-neighbour CVRP.

**Charging Feasibility:** A dynamic-programming SPPRC algorithm inserts recharging station visits optimally into each route to prevent battery depletion.

**`get_hetero_state()`** extracts a heterogeneous graph dictionary:
- **4 node types**: Depot, Satellite, Customer, RechargingStation — each with 10 features (normalised coords, one-hot type, constraint, status, battery remaining, load)
- **5+ edge relations**: Customer↔Customer (kNN + route-sequence), Customer→Satellite, Satellite→Customer, *→RechargingStation
- All features and edge attributes are normalised to `[0, 1]`

---

### 3. `gnn_encoder.py` — Heterogeneous GNN Encoder

Converts the variable-size graph dict into a **fixed-size state vector** for RL.

| Backend | Architecture | Output Dim |
|---|---|---|
| **PyG** (`torch_geometric`) | 2-layer HeteroConv (SAGEConv per relation) → attention-weighted Customer pooling → scalar fusion → MLP | `output_dim` (default 128) |
| **Fallback** (numpy only) | Mean-pool per node type + scalar concat | 45 |

**Search-progress scalars** (5 features appended to graph embedding):
1. `current_cost / initial_cost`
2. `best_cost / initial_cost`
3. `(current - best) / initial` (gap)
4. `iteration / max_iterations` (progress)
5. `(current - best) / best` (stagnation)

The `encode_state()` convenience function handles the full pipeline: raw graph → encoder → numpy vector.

---

### 4. `lns_env.py` — RL Environment

OpenAI Gym-style MDP for **operator selection** within the L2 ALNS loop.

| Component | Detail |
|---|---|
| **State** | Fixed-size numpy vector from `encode_state()` |
| **Actions** | `0` = Related Nodes, `1` = Random Routes, `2` = Close Satellite |
| **Repair** | Deterministic Regret-2 insertion + SPPRC charging |
| **Reward** | Normalised `Δcost / initial_cost` + terminal bonus (see below) |
| **Acceptance** | Simulated annealing — can accept worse solutions early, cools over episode |
| **Done** | After `max_steps` iterations (default 200) |
| **Physics** | Strict capacity + battery feasibility check; `−1000` penalty if violated |

**Reward Design:**
- **Per-step**: `(cost_old − cost_new) / initial_cost` — normalised so reward scale is consistent across instances of different sizes.
- **Terminal bonus**: On the final step, an additional `(initial_cost − best_cost) / initial_cost` is added, rewarding cumulative episode improvement.
- **Infeasible penalty**: Physics violations (capacity or battery exceeded) receive `−1000` and the routes are reverted.

**Simulated Annealing Acceptance:**
The environment uses an SA-style Metropolis criterion to decide whether to *accept* a repaired solution. Temperature `T` cools linearly from 1.0 to 0.01 over the episode:
```
T = max(0.01, 1.0 − step / max_steps)
Accept if  Δ > 0  or  random() < exp(Δ / (T × current_cost))
```
This allows the agent to escape local optima early in the episode while converging to greedy acceptance near the end.

```python
env = LNSMDPEnv(instance, satellite, customer_ids, max_steps=200)
state = env.reset()
next_state, reward, done, info = env.step(action=0)
```

---

### 5. `benchmark_runner.py` — Benchmarking & Offline Data

**Benchmark mode** (`python benchmark_runner.py`):
Runs the ALNS solver multiple times per instance, reports min/avg/max cost, saves best-solution plots and a summary CSV.

**Offline RL data generation** (`python benchmark_runner.py --generate_data`):
Rolls out random-policy episodes through `LNSMDPEnv` for every satellite subproblem, saving `(S, A, R, S', done)` tuples to `offline_trajectories.pkl` for CQL/DQN training.

---

### 6. `test_pipeline.py` — Validation Suite

**59 tests** across 7 sections, runnable without pytest:

```
1. Instance loading           — file parsing, field validation
2. Solver basics              — distance, assignment, construction, destroy/repair
3. get_hetero_state()         — shape, dtype, normalisation, one-hot, no NaN/Inf
4. GNN encoder                — build, encode, fixed-size output, cross-instance consistency
5. LNSMDPEnv reset + step     — state validity, all actions, reward finiteness
6. Full episode               — 200-step random policy, cost tracking
7. State consistency          — dim/dtype stability across full episode
```

---

## Quick Start

```bash
# 1. Parse & inspect an instance
python e2e_vrp_loader.py Set2/E-Set2a_E-n22-k4-s10-14_int.dat

# 2. Solve a single instance
python lns_solver.py Set2/E-Set2a_E-n22-k4-s10-14_int.dat

# 3. Solve and visualise
python visualizer.py Set2/E-Set2a_E-n22-k4-s10-14_int.dat

# 4. Run the full benchmark suite
python benchmark_runner.py --runs 5 --iters 200

# 5. Generate offline RL trajectories
python benchmark_runner.py --generate_data --iters 200

# 6. Run the full test suite (59 tests)
python test_pipeline.py
```

---

## Instance Datasets

Six benchmark sets from Breunig et al. (2019):

| Set | Instances | Scale | Notes |
|-----|-----------|-------|-------|
| Set2 | 30 | 22 customers | Small, varied satellite configs |
| Set3 | 12 | 22 customers | Different satellite layouts |
| Set5 | 18 | ~100 customers | Medium |
| Set6a | 27 | ~101 customers | Large, A-series |
| Set7 | 200 | Various | Extended set |
| Set8 | 200 | Up to 800 customers | Large-scale |

All coordinates are scaled ×10 from the classical 2E-VRP benchmarks. Distances are Euclidean, rounded to nearest integer.

---

## Dependencies

**Minimum (solver + env with fallback encoder):**
```bash
pip install numpy matplotlib pandas
```

**Full GNN pipeline (recommended):**
```bash
pip install numpy matplotlib pandas torch torch_geometric
```

---

## License

Academic use only. Instance data from Breunig et al. (2019).
