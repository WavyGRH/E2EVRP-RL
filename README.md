# E2EVRP – Electric Two-Echelon Vehicle Routing Problem

Implementation of the **E2EVRP** based on:

> Breunig, U., Baldacci, R., Hartl, R. F., & Vidal, T. (2019).  
> *The Electric Two-echelon Vehicle Routing Problem.*  
> Computers & Operations Research, 103, 198–210.

## Repository Structure

```
E2EVRP_Instances/
├── e2e_vrp_loader.py      # Instance parser & data structures
├── lns_solver.py           # ALNS solver (destroy / repair operators)
├── visualizer.py           # matplotlib solution plotter
├── benchmark_runner.py     # Multi-run benchmark suite
├── README.md               # This file
├── Set2/ … Set8/           # Instance .dat files
└── benchmark_results/      # Generated results (plots + CSV)
```

## Quick Start

```bash
# 1. Parse & inspect an instance
python e2e_vrp_loader.py Set2/E-Set2a_E-n22-k4-s10-14_int.dat

# 2. Solve a single instance and visualise
python visualizer.py Set2/E-Set2a_E-n22-k4-s10-14_int.dat

# 3. Run the full benchmark suite
python benchmark_runner.py --runs 5 --iters 200
```

## Solver Details

### Algorithm: Adaptive Large Neighbourhood Search (ALNS)

The solver decomposes the E2EVRP into two echelons:

1. **Level 2** (Satellites → Customers): Each satellite's customer set is
   solved as a capacity-constrained VRP using ALNS with:
   - **Destroy operators**: Random Removal (20%), Worst Removal (15%)
   - **Repair operators**: Greedy Cheapest Insertion, Regret-2 Insertion

2. **Level 1** (Depot → Satellites): Trucks route from the depot to
   satellites based on aggregated demand, solved via a greedy nearest-
   neighbour CVRP heuristic.

### Benchmarking

`benchmark_runner.py` runs each instance multiple times with different
random seeds and reports:

| Metric | Description |
|--------|-------------|
| Best Cost | Minimum total cost across all runs |
| Avg Cost  | Mean total cost |
| Worst Cost | Maximum total cost |
| Best L1/L2 | Level-1 and Level-2 cost breakdown for the best solution |
| Trucks / CFs | Number of vehicles used in the best solution |
| Avg Time | Mean wall-clock time per run |

## Dependencies

- Python 3.8+
- `matplotlib`
- `pandas`

```bash
pip install matplotlib pandas
```

## License

Academic use only. Instance data from Breunig et al. (2019).
