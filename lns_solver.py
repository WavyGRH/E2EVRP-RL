"""
ALNS Solver for the Electric Two-Echelon VRP (E2EVRP)
Based on Breunig, Baldacci, Hartl & Vidal (2019)

Features
--------
- Two-echelon decomposition (Trucks → Satellites → Customers)
- Nearest-Neighbour constructive heuristic
- Adaptive LNS with two destroy operators (Random / Worst Removal)
  and two repair operators (Greedy / Regret-2 Insertion)
- Level-1 CVRP solver for truck routing
"""

import math
import copy
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

from e2e_vrp_loader import (
    E2EVRPInstance, Node, Customer, Satellite, Depot, RechargingStation
)


# ==========================================
# Helper dataclass (not currently tracked per-route, kept for future use)
# ==========================================

@dataclass
class Route:
    path: List[int]
    load: int
    distance: float
    battery_used: float
    is_feasible: bool = True


# ==========================================
# Solver
# ==========================================

class LNSSolver:
    """Adaptive Large Neighbourhood Search solver for E2EVRP."""

    def __init__(self, instance: E2EVRPInstance, seed: int = 42):
        self.instance = instance
        random.seed(seed)

        # Fast node lookup
        self.nodes: Dict[int, Node] = {0: instance.depot}
        for s in instance.satellites:
            self.nodes[s.id] = s
        for c in instance.customers:
            self.nodes[c.id] = c
        for r in instance.recharging_stations:
            self.nodes[r.id] = r

        # Customer → satellite mapping (filled during solve)
        self.sat_assignments: Dict[int, List[int]] = {
            s.id: [] for s in instance.satellites
        }

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def dist(self, u_id: int, v_id: int) -> float:
        u, v = self.nodes[u_id], self.nodes[v_id]
        return math.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2)

    # ------------------------------------------------------------------
    # Customer → Satellite assignment (nearest)
    # ------------------------------------------------------------------

    def assign_customers_to_satellites(self):
        """Assign every customer to its nearest satellite."""
        for cust in self.instance.customers:
            best_sat, min_d = None, float('inf')
            for sat in self.instance.satellites:
                d = self.dist(cust.id, sat.id)
                if d < min_d:
                    min_d = d
                    best_sat = sat
            self.sat_assignments[best_sat.id].append(cust.id)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        max_iterations: int = 200,
        time_limit: float = 60.0,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Solve the E2EVRP.

        Returns
        -------
        l1_routes : list of truck routes  (depot ↔ satellites)
        l2_routes : list of CF routes     (satellite ↔ customers)
        """
        t0 = time.time()
        print("  Assigning customers to satellites...")
        self.assign_customers_to_satellites()

        # --- Level 2: Satellite sub-problems ---
        l2_routes_all: List[List[int]] = []
        for sat in self.instance.satellites:
            customers = self.sat_assignments[sat.id]
            if not customers:
                continue
            print(f"  Solving L2 for Satellite {sat.id} ({len(customers)} cust)...")
            routes = self._solve_subproblem_lns(sat, customers, max_iterations)
            l2_routes_all.extend(routes)

        # --- Aggregate satellite demands ---
        sat_demands: Dict[int, int] = {s.id: 0 for s in self.instance.satellites}
        for route in l2_routes_all:
            route_demand = sum(
                self.nodes[nid].demand
                for nid in route
                if isinstance(self.nodes[nid], Customer)
            )
            start_node = route[0]
            if start_node in sat_demands:
                sat_demands[start_node] += route_demand

        # --- Level 1: Truck routing ---
        print("  Solving L1 (Trucks)...")
        l1_routes = self._solve_level1_cvrp(sat_demands)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f}s")
        return l1_routes, l2_routes_all

    # ------------------------------------------------------------------
    # Level-2 LNS loop for one satellite
    # ------------------------------------------------------------------

    def _solve_subproblem_lns(
        self,
        sat: Satellite,
        customer_ids: List[int],
        iterations: int,
    ) -> List[List[int]]:
        current_routes = self._construct_initial(sat, customer_ids)
        current_cost = self.calculate_total_cost(current_routes)
        best_routes = copy.deepcopy(current_routes)
        best_cost = current_cost

        destroy_ops = [self._destroy_random, self._destroy_worst]
        repair_ops = [self._repair_greedy, self._repair_regret]

        for _ in range(iterations):
            destroy_op = random.choice(destroy_ops)
            repair_op = random.choice(repair_ops)

            destroyed, removed = destroy_op(current_routes, sat)
            repaired = repair_op(destroyed, removed, sat)
            new_cost = self.calculate_total_cost(repaired)

            if new_cost < current_cost:
                current_routes = repaired
                current_cost = new_cost
                if new_cost < best_cost:
                    best_routes = copy.deepcopy(current_routes)
                    best_cost = new_cost

        return self._ensure_charging_feasibility(best_routes)

    # ------------------------------------------------------------------
    # Constructive heuristic (Nearest Neighbour)
    # ------------------------------------------------------------------

    def _construct_initial(
        self, sat: Satellite, customer_ids: List[int]
    ) -> List[List[int]]:
        unvisited = set(customer_ids)
        routes: List[List[int]] = []
        cap = self.instance.city_freighter_config.capacity

        while unvisited:
            route = [sat.id]
            load = 0
            cur = sat.id
            while True:
                best, best_d = None, float('inf')
                for cand in unvisited:
                    if load + self.nodes[cand].demand <= cap:
                        d = self.dist(cur, cand)
                        if d < best_d:
                            best_d = d
                            best = cand
                if best is not None:
                    route.append(best)
                    load += self.nodes[best].demand
                    cur = best
                    unvisited.remove(best)
                else:
                    break
            route.append(sat.id)
            routes.append(route)
        return routes

    # ------------------------------------------------------------------
    # Destroy operators
    # ------------------------------------------------------------------

    def _destroy_random(
        self, routes: List[List[int]], sat: Satellite
    ) -> Tuple[List[List[int]], List[int]]:
        """Remove ~20 % of customers at random."""
        rate = 0.20
        n_remove = max(1, int(len(self.sat_assignments[sat.id]) * rate))
        new_routes = copy.deepcopy(routes)

        custs = [
            (ri, ni, node)
            for ri, r in enumerate(new_routes)
            for ni, node in enumerate(r)
            if isinstance(self.nodes[node], Customer)
        ]
        if not custs:
            return new_routes, []

        chosen = random.sample(custs, min(n_remove, len(custs)))
        ids = set(x[2] for x in chosen)
        return self._strip(new_routes, ids), list(ids)

    def _destroy_worst(
        self, routes: List[List[int]], sat: Satellite
    ) -> Tuple[List[List[int]], List[int]]:
        """Remove ~15 % of customers that cause the largest detour."""
        n_remove = max(1, int(len(self.sat_assignments[sat.id]) * 0.15))
        new_routes = copy.deepcopy(routes)

        savings = []
        for r in new_routes:
            if len(r) <= 2:
                continue
            for i in range(1, len(r) - 1):
                node = r[i]
                if not isinstance(self.nodes[node], Customer):
                    continue
                s = (
                    self.dist(r[i - 1], node)
                    + self.dist(node, r[i + 1])
                    - self.dist(r[i - 1], r[i + 1])
                )
                savings.append((s, node))

        savings.sort(key=lambda x: x[0], reverse=True)
        ids = set(x[1] for x in savings[:n_remove])
        return self._strip(new_routes, ids), list(ids)

    @staticmethod
    def _strip(routes: List[List[int]], ids: set) -> List[List[int]]:
        """Remove nodes from routes and drop empty routes."""
        return [
            [n for n in r if n not in ids]
            for r in routes
            if sum(1 for n in r if n not in ids) > 2
        ]

    # ------------------------------------------------------------------
    # Repair operators
    # ------------------------------------------------------------------

    def _repair_greedy(
        self,
        routes: List[List[int]],
        removed: List[int],
        sat: Satellite,
    ) -> List[List[int]]:
        """Greedy cheapest-insertion."""
        random.shuffle(removed)
        return self._insert_greedy(routes, removed, sat)

    def _repair_regret(
        self,
        routes: List[List[int]],
        removed: List[int],
        sat: Satellite,
    ) -> List[List[int]]:
        """
        Regret-2 insertion: insert the customer whose cost difference
        between best and second-best position is largest.
        """
        remaining = list(removed)
        cap = self.instance.city_freighter_config.capacity

        while remaining:
            best_regret = -1
            best_cust = None
            best_r_idx = -1
            best_pos = -1

            for cid in remaining:
                c_demand = self.nodes[cid].demand
                insertion_costs = []  # (cost_increase, route_idx, pos)

                for r_idx, r in enumerate(routes):
                    load = sum(
                        self.nodes[n].demand
                        for n in r
                        if isinstance(self.nodes[n], Customer)
                    )
                    if load + c_demand > cap:
                        continue
                    for pos in range(1, len(r)):
                        u, v = r[pos - 1], r[pos]
                        delta = (
                            self.dist(u, cid)
                            + self.dist(cid, v)
                            - self.dist(u, v)
                        )
                        insertion_costs.append((delta, r_idx, pos))

                # New route option
                new_cost = self.dist(sat.id, cid) * 2
                insertion_costs.append((new_cost, -1, -1))

                insertion_costs.sort(key=lambda x: x[0])

                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                else:
                    regret = 0.0

                if regret > best_regret or best_cust is None:
                    best_regret = regret
                    best_cust = cid
                    best_r_idx = insertion_costs[0][1]
                    best_pos = insertion_costs[0][2]

            # Perform the insertion
            remaining.remove(best_cust)
            if best_r_idx == -1:
                routes.append([sat.id, best_cust, sat.id])
            else:
                routes[best_r_idx].insert(best_pos, best_cust)

        return routes

    def _insert_greedy(
        self,
        routes: List[List[int]],
        customers: List[int],
        sat: Satellite,
    ) -> List[List[int]]:
        cap = self.instance.city_freighter_config.capacity

        for cid in customers:
            best_delta = float('inf')
            best_ri = -1
            best_pos = -1
            c_demand = self.nodes[cid].demand

            for ri, r in enumerate(routes):
                load = sum(
                    self.nodes[n].demand
                    for n in r
                    if isinstance(self.nodes[n], Customer)
                )
                if load + c_demand > cap:
                    continue
                for pos in range(1, len(r)):
                    u, v = r[pos - 1], r[pos]
                    delta = (
                        self.dist(u, cid) + self.dist(cid, v) - self.dist(u, v)
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_ri = ri
                        best_pos = pos

            new_route_cost = self.dist(sat.id, cid) * 2
            if new_route_cost < best_delta:
                routes.append([sat.id, cid, sat.id])
            elif best_ri != -1:
                routes[best_ri].insert(best_pos, cid)
            else:
                routes.append([sat.id, cid, sat.id])

        return routes

    # ------------------------------------------------------------------
    # Charging feasibility (stub – extend with RS insertion logic)
    # ------------------------------------------------------------------

    def _ensure_charging_feasibility(
        self, routes: List[List[int]]
    ) -> List[List[int]]:
        """Placeholder for battery-feasibility enforcement."""
        return routes

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    def calculate_total_cost(self, routes: List[List[int]]) -> float:
        total = 0.0
        for r in routes:
            for i in range(len(r) - 1):
                total += self.dist(r[i], r[i + 1])
        return total

    # ------------------------------------------------------------------
    # Level-1 CVRP for trucks
    # ------------------------------------------------------------------

    def _solve_level1_cvrp(self, sat_demands: Dict[int, int]) -> List[List[int]]:
        depot_id = self.instance.depot.id
        truck_cap = self.instance.truck_config.capacity
        active = [sid for sid, d in sat_demands.items() if d > 0]

        # Split demands that exceed truck capacity
        tasks = []
        for sid in active:
            dem = sat_demands[sid]
            while dem > truck_cap:
                tasks.append({'id': sid, 'demand': truck_cap})
                dem -= truck_cap
            if dem > 0:
                tasks.append({'id': sid, 'demand': dem})

        routes: List[List[int]] = []
        route = [depot_id]
        load = 0
        cur = depot_id

        while tasks:
            best_idx, best_d = -1, float('inf')
            for i, t in enumerate(tasks):
                if load + t['demand'] <= truck_cap:
                    d = self.dist(cur, t['id'])
                    if d < best_d:
                        best_d = d
                        best_idx = i

            if best_idx != -1:
                t = tasks.pop(best_idx)
                route.append(t['id'])
                load += t['demand']
                cur = t['id']
            else:
                route.append(depot_id)
                routes.append(route)
                route = [depot_id]
                load = 0
                cur = depot_id

        if len(route) > 1:
            route.append(depot_id)
            routes.append(route)

        return routes


# ==========================================
# Quick self-test
# ==========================================

if __name__ == "__main__":
    import os, sys
    from e2e_vrp_loader import load_instance

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(
            os.path.dirname(__file__),
            "Set2", "E-Set2a_E-n22-k4-s10-14_int.dat",
        )

    inst = load_instance(path)
    solver = LNSSolver(inst, seed=42)
    l1, l2 = solver.solve(max_iterations=200)
    print(f"\nLevel-1 routes ({len(l1)}): {l1}")
    print(f"Level-2 routes ({len(l2)}): {l2}")
    print(f"L1 cost: {solver.calculate_total_cost(l1):.2f}")
    print(f"L2 cost: {solver.calculate_total_cost(l2):.2f}")
    print(f"Total  : {solver.calculate_total_cost(l1) + solver.calculate_total_cost(l2):.2f}")
