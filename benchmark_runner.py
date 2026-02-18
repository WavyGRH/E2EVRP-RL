"""
Benchmark Runner for the E2EVRP ALNS Solver
Based on Breunig, Baldacci, Hartl & Vidal (2019)

Runs the ALNS solver multiple times per instance, collects
Min / Avg / Max cost and timing, saves best-solution plots,
and writes a summary CSV.
"""

import os
import time
import pandas as pd
from typing import List

from e2e_vrp_loader import load_instance
from lns_solver import LNSSolver
from visualizer import plot_solution


# ==========================================
# Instance catalogue (representative set)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCE_FILES = [
    # Set2 – small (22 customers)
    os.path.join(BASE_DIR, "Set2", "E-Set2a_E-n22-k4-s10-14_int.dat"),
    # Set3 – small (22 customers, different satellites)
    os.path.join(BASE_DIR, "Set3", "E-Set3a_E-n22-k4-s13-14_int.dat"),
    # Set5 – medium (100 customers)
    os.path.join(BASE_DIR, "Set5", "E-Set5_100-10-1_int.dat"),
    # Set6a – large (101 customers)
    os.path.join(BASE_DIR, "Set6a", "E-Set6a_A-n101-4_int.dat"),
    # Set7
    os.path.join(BASE_DIR, "Set7", "E-Set7_01_02.dat"),
    # Set8 – large (800 customers implied by name)
    os.path.join(BASE_DIR, "Set8", "E-Set8_01_0800.dat"),
]


# ==========================================
# Benchmark logic
# ==========================================

def run_benchmark(
    instance_files: List[str] = None,
    num_runs: int = 5,
    max_iterations: int = 200,
    output_dir: str = None,
):
    """
    Benchmark the ALNS solver on a list of instances.

    For each instance, run `num_runs` independent solves (different seeds),
    record costs and times, save best-solution plots, and produce a summary
    CSV matching the style of Breunig et al. (2019) Tables 2-5.
    """
    if instance_files is None:
        instance_files = INSTANCE_FILES
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"E2EVRP ALNS Benchmark  |  Runs/instance: {num_runs}  |  Iters: {max_iterations}")
    print(f"{'='*60}")

    rows = []

    for fpath in instance_files:
        if not os.path.exists(fpath):
            print(f"  ⚠  Skipping (not found): {fpath}")
            continue

        inst = load_instance(fpath)
        iname = inst.name
        print(f"\n▶ {iname}  ({len(inst.customers)} customers, "
              f"{len(inst.satellites)} satellites)")

        costs, times = [], []
        best_cost = float('inf')
        best_sol = None

        for run_id in range(num_runs):
            solver = LNSSolver(inst, seed=42 + run_id)
            t0 = time.time()
            l1, l2 = solver.solve(max_iterations=max_iterations)
            dur = time.time() - t0

            l1_cost = solver.calculate_total_cost(l1)
            l2_cost = solver.calculate_total_cost(l2)
            total = l1_cost + l2_cost
            costs.append(total)
            times.append(dur)

            tag = ""
            if total < best_cost:
                best_cost = total
                best_sol = (l1, l2, l1_cost, l2_cost)
                tag = " ★"
            print(f"    Run {run_id+1}/{num_runs}  cost={total:,.2f}  "
                  f"(L1={l1_cost:,.2f}  L2={l2_cost:,.2f})  "
                  f"t={dur:.2f}s{tag}")

        avg_cost = sum(costs) / len(costs)
        max_cost = max(costs)
        min_cost = min(costs)
        avg_time = sum(times) / len(times)

        rows.append({
            "Instance": iname,
            "Customers": len(inst.customers),
            "Satellites": len(inst.satellites),
            "Best Cost": round(min_cost, 2),
            "Avg Cost": round(avg_cost, 2),
            "Worst Cost": round(max_cost, 2),
            "Best L1": round(best_sol[2], 2),
            "Best L2": round(best_sol[3], 2),
            "Trucks": len(best_sol[0]),
            "CFs": len(best_sol[1]),
            "Avg Time (s)": round(avg_time, 2),
        })

        # Save best-solution plot
        plot_path = os.path.join(output_dir, f"{iname}_best.png")
        plot_solution(inst, best_sol[0], best_sol[1], output_path=plot_path)

    # --- Summary table ---
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved → {csv_path}")
    print(f"✓ Plots saved → {output_dir}")

    return df


# ==========================================
# CLI entry point
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2EVRP ALNS Benchmark")
    parser.add_argument("--runs", type=int, default=5,
                        help="Independent runs per instance (default 5)")
    parser.add_argument("--iters", type=int, default=200,
                        help="LNS iterations per solve (default 200)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    run_benchmark(
        num_runs=args.runs,
        max_iterations=args.iters,
        output_dir=args.outdir,
    )
