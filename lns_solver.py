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
import numpy as np
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

        destroy_ops = [
            self._destroy_related_nodes,
            self._destroy_random_routes,
            self._destroy_close_satellite
        ]
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

    def _destroy_related_nodes(
        self, routes: List[List[int]], sat: Satellite
    ) -> Tuple[List[List[int]], List[int]]:
        """A) Related nodes removal: Pick a seed customer, remove it and its closest neighbors."""
        rate = 0.20
        n_remove = max(1, int(len(self.sat_assignments.get(sat.id, [])) * rate))
        new_routes = copy.deepcopy(routes)
        
        custs = [n for r in new_routes for n in r if isinstance(self.nodes[n], Customer)]
        if not custs:
            return new_routes, []
            
        seed = random.choice(custs)
        
        # Sort by distance to seed
        dists = [(self.dist(seed, c), c) for c in custs]
        dists.sort(key=lambda x: x[0])
        
        ids = set(c for _, c in dists[:n_remove])
        return self._strip(new_routes, ids), list(ids)

    def _destroy_random_routes(
        self, routes: List[List[int]], sat: Satellite
    ) -> Tuple[List[List[int]], List[int]]:
        """B) Random routes removal: Removes randomly selected routes entirely."""
        if not routes:
            return routes, []
            
        new_routes = []
        removed_custs = []
        
        # Determine number of routes to remove. Breunig removes e.g. up to p_2 proportion.
        max_remove = max(1, int(len(routes) * 0.37))
        num_remove = random.randint(1, max_remove)
        
        routes_to_remove = random.sample(range(len(routes)), num_remove)
        
        for i, r in enumerate(routes):
            if i in routes_to_remove:
                removed_custs.extend([n for n in r if isinstance(self.nodes[n], Customer)])
            else:
                new_routes.append(list(r))
                
        return new_routes, removed_custs

    def _destroy_close_satellite(
        self, routes: List[List[int]], sat: Satellite
    ) -> Tuple[List[List[int]], List[int]]:
        """
        C) Close satellite: In a single-satellite L2 subproblem, this acts equivalently to 
        removing the majority of the worst routes to simulate major satellite failure/transfer.
        """
        if not routes:
            return routes, []
            
        new_routes = []
        removed_custs = []
        
        # Simulate closing by removing half the routes
        num_remove = max(1, len(routes) // 2)
        routes_to_remove = random.sample(range(len(routes)), num_remove)
        
        for i, r in enumerate(routes):
            if i in routes_to_remove:
                removed_custs.extend([n for n in r if isinstance(self.nodes[n], Customer)])
            else:
                new_routes.append(list(r))
                
        return new_routes, removed_custs

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
        """
        Optimal insertion of charging station visits using a DP algorithm (SPPRC).
        Finds the shortest path across the sequence of customers, splicing in RS visits
        to prevent battery depletion.
        """
        cf_config = self.instance.city_freighter_config
        max_batt = float(cf_config.max_charge)
        energy_rate = cf_config.energy_consumption
        rss = self.instance.recharging_stations
        
        final_routes = []
        
        for route in routes:
            # Drop existing RS nodes to re-optimize
            clean_route = [n for n in route if not isinstance(self.nodes[n], RechargingStation)]
            if len(clean_route) <= 2:
                final_routes.append(clean_route)
                continue
                
            K = len(clean_route) - 1
            
            # dp[i] = list of (cost, battery_remaining, path_of_nodes) arriving at clean_route[i]
            dp = [[] for _ in range(K + 1)]
            dp[0].append((0.0, max_batt, [clean_route[0]]))
            
            for i in range(K):
                if not dp[i]:
                    break
                    
                u = clean_route[i]
                v = clean_route[i + 1]
                
                # Keep only Pareto-optimal labels at node i to prevent explosion
                # A label (cost1, batt1) dominates (cost2, batt2) if cost1 <= cost2 and batt1 >= batt2
                dp[i].sort(key=lambda x: x[0])
                pareto = []
                best_batt = -1.0
                for c, b, p in dp[i]:
                    if b > best_batt:
                        pareto.append((c, b, p))
                        best_batt = b
                
                for cost, current_batt, path in pareto:
                    # 1. Direct travel u -> v
                    e_direct = self.dist(u, v) * energy_rate
                    if current_batt >= e_direct:
                        dp[i+1].append((cost + self.dist(u, v), current_batt - e_direct, path + [v]))
                        
                    # 2. Travel via a Recharging Station u -> rs -> v
                    for rs in rss:
                        e_to_rs = self.dist(u, rs.id) * energy_rate
                        e_from_rs = self.dist(rs.id, v) * energy_rate
                        
                        if current_batt >= e_to_rs and max_batt >= e_from_rs:
                            # Valid detour
                            new_cost = cost + self.dist(u, rs.id) + self.dist(rs.id, v)
                            new_batt = max_batt - e_from_rs
                            dp[i+1].append((new_cost, new_batt, path + [rs.id, v]))
                            
            # Pick best label at the end
            if not dp[K]:
                # Infeasible to fix, just return the original (the physics checker will reject it later)
                final_routes.append(route)
            else:
                best_label = min(dp[K], key=lambda x: x[0])
                final_routes.append(best_label[2])
                
        return final_routes

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    def calculate_total_cost(self, routes: List[List[int]]) -> float:
        cost = 0.0
        for r in routes:
            for i in range(len(r) - 1):
                cost += self.dist(r[i], r[i + 1])
        return cost

    # ------------------------------------------------------------------
    # Environment State Extraction (Hetero-GNN)
    # ------------------------------------------------------------------

    def get_hetero_state(self, current_routes: List[List[int]]) -> dict:

        cf_config = self.instance.city_freighter_config
        max_batt   = float(cf_config.max_charge)
        energy_rate = cf_config.energy_consumption
        max_cap    = float(cf_config.capacity)

        # Coordinate normalisation
        all_coords = [(n.x, n.y) for n in self.nodes.values()]
        max_x = max(c[0] for c in all_coords) or 1.0
        max_y = max(c[1] for c in all_coords) or 1.0
        max_dist = (max_x**2 + max_y**2) ** 0.5 + 1e-8

        depot = self.instance.depot
        sats  = self.instance.satellites
        custs = self.instance.customers
        rss   = self.instance.recharging_stations

        # --- Dynamic state: remaining battery and served status ---
        # Key fix: start at max_charge, subtract, reset to max at RS (not 0)
        node_batt_remaining = {n: max_batt for n in self.nodes}
        node_load_remaining = {n: 0.0 for n in self.nodes}
        cust_served   = {c.id: False for c in custs}
        # Track unserved demand per satellite
        sat_unserved  = {s.id: sum(
            self.nodes[cid].demand for cid in self.sat_assignments.get(s.id, [])
        ) for s in sats}

        for route in current_routes:
            batt = max_batt
            load = 0.0
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                batt -= self.dist(u, v) * energy_rate
                node_v = self.nodes[v]
                if isinstance(node_v, RechargingStation):
                    batt = max_batt          # FIX: reset to max, not 0
                if isinstance(node_v, Customer):
                    load += node_v.demand
                    cust_served[v] = True
                    sat_id = route[0]        # satellite is always first node
                    if sat_id in sat_unserved:
                        sat_unserved[sat_id] = max(0.0,
                            sat_unserved[sat_id] - node_v.demand)
                node_batt_remaining[v] = max(batt, 0.0)
                node_load_remaining[v] = load

        # Normalise satellite handling cost and max capacity
        max_handling = max((s.handling_cost for s in sats), default=1.0) or 1.0
        max_sat_cap  = max((s.max_capacity  for s in sats), default=1.0) or 1.0

        # --- Feature builder ---
        # 10 features: [x, y, is_depot, is_sat, is_cust, is_rs,
        #               constraint_norm, status, batt_remaining_norm, load_norm]
        def feats(node_list, type_idx, constraint_fn, status_fn):
            rows = []
            for n in node_list:
                oh = [float(type_idx == i) for i in range(4)]
                rows.append([
                    n.x / max_x,
                    n.y / max_y,
                    *oh,
                    constraint_fn(n),             # FIX: normalised per type
                    status_fn(n),                 # FIX: meaningful for all types
                    node_batt_remaining[n.id] / max_batt,  # FIX: remaining, not consumed
                    node_load_remaining[n.id] / max_cap,   # FIX: normalised by capacity
                ])
            return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 10), dtype=np.float32)

        depot_x = feats([depot], 0,
                        lambda n: 0.0,
                        lambda n: 0.0)
        sats_x  = feats(sats, 1,
                        lambda n: n.handling_cost / max_handling,
                        lambda n: sat_unserved[n.id] / (n.max_capacity or 1))  # FIX: unserved ratio
        custs_x = feats(custs, 2,
                        lambda n: n.demand / max_cap,
                        lambda n: 0.0 if cust_served[n.id] else 1.0)
        rss_x   = feats(rss, 3,
                        lambda n: 0.0,
                        lambda n: 0.0)

        # --- Edge builder: kNN WITHOUT self-loops ---
        def knn_edges(src_nodes, dst_nodes, k, exclude_self=False):
            if not src_nodes or not dst_nodes or k <= 0:
                return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
            sc = np.array([[n.x, n.y] for n in src_nodes])
            dc = np.array([[n.x, n.y] for n in dst_nodes])
            diff = sc[:, None, :] - dc[None, :, :]
            dists = np.sqrt((diff**2).sum(-1))
            eu, ev, ea = [], [], []
            for ui in range(len(src_nodes)):
                order = np.argsort(dists[ui])
                count = 0
                for vi in order:
                    if count >= k:
                        break
                    # FIX: skip self-loops when src and dst are the same list
                    if exclude_self and src_nodes[ui].id == dst_nodes[vi].id:
                        continue
                    eu.append(ui); ev.append(vi)
                    ea.append([dists[ui, vi] / max_dist])
                    count += 1
            return (np.vstack([eu, ev]).astype(np.int64) if eu
                    else np.zeros((2, 0), dtype=np.int64),
                    np.array(ea, dtype=np.float32) if ea
                    else np.zeros((0, 1), dtype=np.float32))

        c2c_idx, c2c_attr = knn_edges(custs, custs, k=5, exclude_self=True)  # FIX: no self-loops
        c2s_idx, c2s_attr = knn_edges(custs, sats,  k=2)
        s2c_idx, s2c_attr = knn_edges(sats,  custs, k=min(10, len(custs)))   # FIX: sat→cust

        # --- Route-sequence edges: the critical missing piece ---
        # FIX: add (Customer, follows, Customer) edges from current solution
        seq_u, seq_v, seq_a = [], [], []
        cust_id_to_local = {c.id: i for i, c in enumerate(custs)}
        for route in current_routes:
            cust_nodes_in_route = [n for n in route
                                    if n in cust_id_to_local]
            for i in range(len(cust_nodes_in_route) - 1):
                u_id = cust_nodes_in_route[i]
                v_id = cust_nodes_in_route[i + 1]
                seq_u.append(cust_id_to_local[u_id])
                seq_v.append(cust_id_to_local[v_id])
                d = self.dist(u_id, v_id)
                seq_a.append([d / max_dist])

        seq_idx  = (np.vstack([seq_u, seq_v]).astype(np.int64)
                    if seq_u else np.zeros((2, 0), dtype=np.int64))
        seq_attr = (np.array(seq_a, dtype=np.float32)
                    if seq_a else np.zeros((0, 1), dtype=np.float32))

        edge_dict = {
            ("Customer",  "near",     "Customer"):  {"edge_index": c2c_idx, "edge_attr": c2c_attr},
            ("Customer",  "near",     "Satellite"): {"edge_index": c2s_idx, "edge_attr": c2s_attr},
            ("Satellite", "serves",   "Customer"):  {"edge_index": s2c_idx, "edge_attr": s2c_attr},
            ("Customer",  "follows",  "Customer"):  {"edge_index": seq_idx,  "edge_attr": seq_attr},
        }
        for src_type, src_nodes in [("Depot",[depot]),("Satellite",sats),
                                      ("Customer",custs),("RechargingStation",rss)]:
            idx, attr = knn_edges(src_nodes, rss, k=min(2, len(rss)))
            edge_dict[(src_type, "near", "RechargingStation")] = {
                "edge_index": idx, "edge_attr": attr
            }

        return {
            "Depot":              {"x": depot_x},
            "Satellite":          {"x": sats_x},
            "Customer":           {"x": custs_x},
            "RechargingStation":  {"x": rss_x},
            **edge_dict,
        }
    
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
