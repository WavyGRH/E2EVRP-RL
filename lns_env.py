"""
RL-ready Phase 1 Environment for ALNS L2 Subproblems.

Wraps the LNSSolver's single-satellite routing logic into a Gym-style MDP.
Generates transition tuples (S, A, R, S') with PyTorch Hetero-GNN states.
Enforces strict EV routing constraints.
"""
import copy
from typing import List, Tuple
from gnn_encoder import build_encoder, encode_state
from e2e_vrp_loader import E2EVRPInstance, Satellite, Customer
from lns_solver import LNSSolver
import random
import math 

class LNSMDPEnv:
    """
    OpenAI Gym-style RL Environment for operator selection in LNS.
    State: Heterogeneous graph dictionary (Depot, Sats, Custs, RS).
    Action: 0 (Related Nodes), 1 (Random Routes), 2 (Close Satellite).
    Reward: Improvement in route cost (Cost_old - Cost_new), or -1000 for invalid.
    """

    def __init__(
        self,
        instance: E2EVRPInstance,
        sat: Satellite,
        customer_ids: List[int],
        max_steps: int = 200,
        seed: int = 42
    ):
        self.instance = instance
        self.sat = sat
        self.customer_ids = customer_ids
        self.max_steps = max_steps
        self.encoder = build_encoder()
        self.initial_cost = 0.0
        self.best_cost = 0.0

        # Core ALNS solver wrapper
        self.solver = LNSSolver(instance, seed=seed)
        # Ensure solver knows the assignments for this subproblem
        self.solver.sat_assignments[sat.id] = customer_ids
        
        self.step_count = 0
        self.routes = []
        self.current_cost = 0.0

    def reset(self) -> dict:
        """Runs the constructive heuristic to build initial safe routes."""
        self.step_count = 0
        self.routes = self.solver._construct_initial(self.sat, self.customer_ids)
        
        # Enforce charging feasibility on initial routes to ensure a valid starting point
        self.routes = self.solver._ensure_charging_feasibility(self.routes)
        self.current_cost = self.solver.calculate_total_cost(self.routes)
        self.initial_cost = self.current_cost
        self.best_cost = self.current_cost
        raw = self.solver.get_hetero_state(self.routes)
        return encode_state(
            self.encoder, raw, self.current_cost, self.current_cost, self.current_cost,
            0, self.max_steps
        )

    def _check_strict_physics(self, routes: List[List[int]]) -> bool:
        """
        Acts as a strict referee.
        Rejects moves that violate capacity or max_charge limits.
        """
        cf_config = self.instance.city_freighter_config
        max_cap = cf_config.capacity
        max_charge = float(cf_config.max_charge)
        energy_rate = cf_config.energy_consumption

        for route in routes:
            load = 0
            battery = max_charge
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                node_v = self.solver.nodes[v]
                
                # Check Capacity
                if isinstance(node_v, Customer):
                    load += node_v.demand
                    if load > max_cap:
                        return False
                        
                # Check Battery
                energy_needed = self.solver.dist(u, v) * energy_rate
                battery -= energy_needed
                if battery < 0:
                    return False
                
                # Assume recharging stations recharge to full
                if getattr(node_v, "id", -1) in [r.id for r in self.instance.recharging_stations]:
                    battery = max_charge

        return True

    def step(self, action_index: int) -> Tuple[dict, float, bool, dict]:
        """
        Takes an operator action.
        0: Related Nodes Removal
        1: Random Routes Removal
        2: Close Satellite Removal
        Always follows with Deterministic _repair_regret and SPPRC charging station insertion.
        """
        if action_index == 0:
            destroy_op = self.solver._destroy_related_nodes
        elif action_index == 1:
            destroy_op = self.solver._destroy_random_routes
        elif action_index == 2:
            destroy_op = self.solver._destroy_close_satellite
        else:
            raise ValueError(f"Unknown action {action_index}")

        # 1. Destroy
        destroyed_routes, removed_customers = destroy_op(self.routes, self.sat)
        
        # 2. Repair (Deterministic Regret-2)
        repaired_routes = self.solver._repair_regret(destroyed_routes, removed_customers, self.sat)
        
        # Optional: ensure charging feasibility locally
        repaired_routes = self.solver._ensure_charging_feasibility(repaired_routes)

        # 3. Strict Physics Review
        if not self._check_strict_physics(repaired_routes):
            # Massive negative penalty, revert to old routes
            reward = -1000.0
            # State remains the same (reverted)
        else:
            new_cost = self.solver.calculate_total_cost(repaired_routes)
            delta = self.current_cost - new_cost
            # Normalised reward
            reward = delta / (self.initial_cost + 1e-8)
            T = max(0.01, 1.0 - self.step_count / self.max_steps) 
            exponent = delta / (T * self.current_cost + 1e-8)
            if delta > 0 or random.random() < math.exp(min(exponent, 0)):
                self.routes = repaired_routes
                self.current_cost = new_cost

        self.step_count += 1
        done = self.step_count >= self.max_steps
        self.best_cost = min(self.best_cost, self.current_cost)
        if done:
            reward += (self.initial_cost - self.best_cost) / (self.initial_cost + 1e-8)
        raw = self.solver.get_hetero_state(self.routes)
        # 4. Extract new Hetero-GNN state
        next_state = encode_state(
            self.encoder, raw, self.current_cost, self.best_cost, self.initial_cost,
            self.step_count, self.max_steps
        )
        
        info = {
            "current_cost": self.current_cost,
            "reward": reward,
            "action": action_index
        }
        
        return next_state, reward, done, info
