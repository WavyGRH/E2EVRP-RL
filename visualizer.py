"""
E2E VRP Solution Visualizer
Plots depot, satellites, customers, recharging stations, and routes.
"""

import matplotlib.pyplot as plt
from typing import List
from e2e_vrp_loader import E2EVRPInstance


def plot_solution(
    instance: E2EVRPInstance,
    l1_routes: List[List[int]],
    l2_routes: List[List[int]],
    output_path: str = None,
):
    """
    Visualise an E2EVRP solution.

    Parameters
    ----------
    instance   : parsed instance
    l1_routes  : truck routes   (depot ↔ satellites)
    l2_routes  : CF routes      (satellite ↔ customers)
    output_path: if given, save plot to this path; otherwise plt.show()
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # --- Nodes ---
    ax.scatter(
        [instance.depot.x], [instance.depot.y],
        c='black', marker='s', s=200, label='Depot', zorder=10,
    )
    ax.scatter(
        [s.x for s in instance.satellites],
        [s.y for s in instance.satellites],
        c='red', marker='^', s=150, label='Satellites', zorder=9,
    )
    ax.scatter(
        [c.x for c in instance.customers],
        [c.y for c in instance.customers],
        c='blue', marker='o', s=50, alpha=0.6, label='Customers', zorder=5,
    )
    ax.scatter(
        [r.x for r in instance.recharging_stations],
        [r.y for r in instance.recharging_stations],
        c='green', marker='P', s=100, label='Recharging Stations', zorder=6,
    )

    # --- ID → Node map ---
    node_map = {0: instance.depot}
    for s in instance.satellites:
        node_map[s.id] = s
    for c in instance.customers:
        node_map[c.id] = c
    for r in instance.recharging_stations:
        node_map[r.id] = r

    # --- Level 1 routes (trucks) ---
    for route in l1_routes:
        xs = [node_map[n].x for n in route if n in node_map]
        ys = [node_map[n].y for n in route if n in node_map]
        ax.plot(xs, ys, c='gray', ls='--', lw=2, alpha=0.5)

    # --- Level 2 routes (city freighters) ---
    cmap = plt.cm.get_cmap('tab10')
    for i, route in enumerate(l2_routes):
        xs = [node_map[n].x for n in route if n in node_map]
        ys = [node_map[n].y for n in route if n in node_map]
        ax.plot(xs, ys, c=cmap(i % 10), ls='-', lw=1.5, alpha=0.8)

    ax.set_title(f"E2E VRP Solution: {instance.name}")
    ax.legend(loc='upper right')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, ls=':', alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved → {output_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import os, sys
    from e2e_vrp_loader import load_instance
    from lns_solver import LNSSolver

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(
            os.path.dirname(__file__),
            "Set2", "E-Set2a_E-n22-k4-s10-14_int.dat",
        )

    inst = load_instance(path)
    solver = LNSSolver(inst)
    l1, l2 = solver.solve(max_iterations=100)
    plot_solution(inst, l1, l2, output_path="solution_test.png")
