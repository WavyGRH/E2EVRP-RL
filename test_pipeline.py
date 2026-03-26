"""
test_pipeline.py
================
Drop-in CMD test for the full E2EVRP → GNN → MDP pipeline.
No pytest required.  Run with:

    python test_pipeline.py

Tests run in order.  Each prints PASS / FAIL with a reason.
If a test fails the rest of the section still runs so you see
every broken thing, not just the first one.

Sections
--------
  1. Instance loading
  2. Solver basics (distance, assignment, construction)
  3. get_hetero_state() — shape, dtype, value sanity
  4. GNN encoder (fallback or PyG)
  5. encode_state() fixed-size output
  6. LNSMDPEnv — reset, step, episode, reward
  7. Full episode with random policy
"""

import os
import sys
import math
import copy
import random
import traceback
import numpy as np

# ── Colour helpers (work on any terminal) ─────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS_TOTAL = 0
FAIL_TOTAL = 0


def ok(msg):
    global PASS_TOTAL
    PASS_TOTAL += 1
    print(f"  {GREEN}PASS{RESET}  {msg}")


def fail(msg, detail=""):
    global FAIL_TOTAL
    FAIL_TOTAL += 1
    line = f"  {RED}FAIL{RESET}  {msg}"
    if detail:
        line += f"\n        {YELLOW}{detail}{RESET}"
    print(line)


def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def run(label, fn):
    """Run fn(), catch any exception, report pass/fail."""
    try:
        result = fn()
        if result is False:
            fail(label, "returned False")
        else:
            ok(label)
        return result
    except Exception as exc:
        fail(label, f"{type(exc).__name__}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Locate instance file
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INSTANCE_PATH = os.path.join(SCRIPT_DIR, "Set2",
                              "E-Set2a_E-n22-k4-s10-14_int.dat")

if not os.path.exists(INSTANCE_PATH):
    # Fallback: search common locations
    for candidate in [
        os.path.join(SCRIPT_DIR, "instances", "Set2",
                     "E-Set2a_E-n22-k4-s10-14_int.dat"),
        os.path.join(SCRIPT_DIR, "..", "Set2",
                     "E-Set2a_E-n22-k4-s10-14_int.dat"),
    ]:
        if os.path.exists(candidate):
            INSTANCE_PATH = candidate
            break
    else:
        print(f"\n{RED}ERROR: Instance file not found.{RESET}")
        print(f"  Expected: {INSTANCE_PATH}")
        print(f"  Ensure Set2/ folder is in the same directory as this script.")
        sys.exit(1)

print(f"\n{BOLD}Instance: {INSTANCE_PATH}{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Instance loading
# ─────────────────────────────────────────────────────────────────────────────
section("1 · Instance loading  (e2e_vrp_loader.py)")

try:
    from e2e_vrp_loader import load_instance, E2EVRPInstance
    inst = load_instance(INSTANCE_PATH)
except Exception as exc:
    print(f"  {RED}FATAL{RESET}  Cannot import e2e_vrp_loader or load instance: {exc}")
    sys.exit(1)

run("Instance is E2EVRPInstance",       lambda: isinstance(inst, E2EVRPInstance))
run("Depot exists",                     lambda: inst.depot is not None)
run("Has satellites (>0)",              lambda: len(inst.satellites) > 0)
run("Has customers (>0)",               lambda: len(inst.customers)  > 0)
run("Has recharging stations (>0)",     lambda: len(inst.recharging_stations) > 0)
run("TruckConfig capacity > 0",         lambda: inst.truck_config.capacity > 0)
run("CityFreighter capacity > 0",       lambda: inst.city_freighter_config.capacity > 0)
run("CityFreighter max_charge > 0",     lambda: inst.city_freighter_config.max_charge > 0)
run("CityFreighter energy_consumption > 0",
    lambda: inst.city_freighter_config.energy_consumption > 0)
run("Customer IDs are unique",
    lambda: len({c.id for c in inst.customers}) == len(inst.customers))

print(f"\n  Summary: {len(inst.customers)} customers, "
      f"{len(inst.satellites)} satellites, "
      f"{len(inst.recharging_stations)} charging stations")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Solver basics
# ─────────────────────────────────────────────────────────────────────────────
section("2 · Solver basics  (lns_solver.py)")

try:
    from lns_solver import LNSSolver
    solver = LNSSolver(inst, seed=42)
except Exception as exc:
    print(f"  {RED}FATAL{RESET}  Cannot import LNSSolver: {exc}")
    sys.exit(1)

run("Distance depot→sat[0] > 0",
    lambda: solver.dist(inst.depot.id, inst.satellites[0].id) > 0)
run("Distance is symmetric",
    lambda: abs(solver.dist(0, inst.satellites[0].id) -
                solver.dist(inst.satellites[0].id, 0)) < 1e-9)

solver.assign_customers_to_satellites()
run("All customers assigned",
    lambda: sum(len(v) for v in solver.sat_assignments.values()) == len(inst.customers))
run("No customer assigned twice", lambda: (
    len({c for ids in solver.sat_assignments.values() for c in ids}) ==
    len(inst.customers)
))

sat   = inst.satellites[0]
cids  = solver.sat_assignments[sat.id]

def _test_construction():
    assert len(cids) > 0, "no customers assigned to satellite 0"
    routes = solver._construct_initial(sat, cids)
    assert len(routes) > 0
    # Every customer appears exactly once
    served = [n for r in routes for n in r
              if n not in (sat.id,) and
              any(n == c for c in cids)]
    assert len(served) == len(cids), \
        f"served={len(served)} vs assigned={len(cids)}"
    return True

run("Constructive heuristic covers all assigned customers", _test_construction)

routes = solver._construct_initial(sat, cids)
routes = solver._ensure_charging_feasibility(routes)
cost   = solver.calculate_total_cost(routes)

run("Total cost after construction > 0",  lambda: cost > 0)
run("Routes start and end at satellite",
    lambda: all(r[0] == sat.id and r[-1] == sat.id for r in routes))
run("No empty routes",
    lambda: all(len(r) >= 3 for r in routes))

def _test_destroy_repair():
    r2, removed = solver._destroy_related_nodes(routes, sat)
    assert isinstance(removed, list)
    repaired = solver._repair_regret(r2, removed, sat)
    new_cost  = solver.calculate_total_cost(repaired)
    assert new_cost > 0
    return True

run("Destroy→repair cycle completes without crash", _test_destroy_repair)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — get_hetero_state() output
# ─────────────────────────────────────────────────────────────────────────────
section("3 · Hetero-graph feature extraction  (get_hetero_state)")

graph = solver.get_hetero_state(routes)

# Shape checks
NODE_TYPES = ["Depot", "Satellite", "Customer", "RechargingStation"]
FEAT_DIM   = 10

for ntype in NODE_TYPES:
    run(f"{ntype} key present in dict",
        lambda nt=ntype: nt in graph)
    run(f"{ntype} has 'x' sub-key",
        lambda nt=ntype: "x" in graph.get(nt, {}))
    run(f"{ntype} feature dim == {FEAT_DIM}",
        lambda nt=ntype: graph[nt]["x"].shape[1] == FEAT_DIM
        if graph[nt]["x"].shape[0] > 0 else True)

# Value sanity
def _check_coords():
    for nt in NODE_TYPES:
        x = graph[nt]["x"]
        if x.shape[0] == 0:
            continue
        # x_norm and y_norm should be in (0, 1]
        assert x[:, 0].min() > 0 and x[:, 0].max() <= 1.0 + 1e-6, \
            f"{nt} x_norm out of (0,1]: {x[:,0].min():.4f}..{x[:,0].max():.4f}"
        assert x[:, 1].min() > 0 and x[:, 1].max() <= 1.0 + 1e-6, \
            f"{nt} y_norm out of (0,1]: {x[:,1].min():.4f}..{x[:,1].max():.4f}"
    return True

run("Coordinate normalisation in (0, 1]", _check_coords)

def _check_onehot():
    expected = {"Depot": 0, "Satellite": 1, "Customer": 2, "RechargingStation": 3}
    for nt, idx in expected.items():
        x = graph[nt]["x"]
        if x.shape[0] == 0:
            continue
        oh = x[:, 2:6]
        # Each row should be a valid one-hot
        assert (oh.sum(axis=1) == 1).all(), f"{nt} one-hot rows don't sum to 1"
        assert (oh[:, idx] == 1).all(),     f"{nt} wrong one-hot position"
    return True

run("One-hot node type encoding is correct", _check_onehot)

def _check_battery():
    # Battery remaining should be in [0, 1] (normalised by max_charge)
    for nt in NODE_TYPES:
        x = graph[nt]["x"]
        if x.shape[0] == 0:
            continue
        batt = x[:, 8]
        assert batt.min() >= -1e-6, \
            f"{nt} battery remaining is negative: {batt.min():.4f}"
        assert batt.max() <= 1.0 + 1e-6, \
            f"{nt} battery remaining > 1: {batt.max():.4f}"
    return True

run("Battery feature (remaining) is normalised in [0, 1]", _check_battery)

def _check_no_nan():
    for nt in NODE_TYPES:
        x = graph[nt]["x"]
        assert not np.isnan(x).any(), f"{nt} features contain NaN"
        assert not np.isinf(x).any(), f"{nt} features contain Inf"
    return True

run("No NaN or Inf in any node feature", _check_no_nan)

# Edge checks
EXPECTED_EDGE_TYPES = [
    ("Customer",  "near",    "Customer"),
    ("Customer",  "near",    "Satellite"),
    ("Satellite", "serves",  "Customer"),
    ("Customer",  "follows", "Customer"),
]

for et in EXPECTED_EDGE_TYPES:
    run(f"Edge type {et[0]}-{et[1]}-{et[2]} present",
        lambda e=et: e in graph)

def _check_no_selfloops():
    key = ("Customer", "near", "Customer")
    if key not in graph:
        return True
    ei = graph[key]["edge_index"]
    self_loops = (ei[0] == ei[1]).sum()
    assert self_loops == 0, f"Found {self_loops} self-loops in C→C edges"
    return True

run("No self-loops in Customer→Customer edges", _check_no_selfloops)

def _check_sequence_edges():
    key = ("Customer", "follows", "Customer")
    if key not in graph:
        return True
    ei = graph[key]["edge_index"]
    ea = graph[key]["edge_attr"]
    # If routes exist, there should be sequence edges
    n_cust_in_routes = sum(
        sum(1 for n in r if any(n == c.id for c in inst.customers))
        for r in routes
    )
    if n_cust_in_routes >= 2:
        assert ei.shape[1] > 0, "No sequence edges despite multi-customer routes"
        assert ea.shape[0] == ei.shape[1], "edge_attr row count mismatch"
    return True

run("Route-sequence edges populated correctly", _check_sequence_edges)

def _check_edge_attr_normalised():
    for key, val in graph.items():
        if not isinstance(key, tuple):
            continue
        ea = val["edge_attr"]
        if ea.shape[0] == 0:
            continue
        assert ea.min() >= -1e-6, f"{key} edge_attr negative: {ea.min():.4f}"
        assert ea.max() <= 1.0 + 1e-6, f"{key} edge_attr > 1: {ea.max():.4f}"
    return True

run("All edge attributes normalised in [0, 1]", _check_edge_attr_normalised)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — GNN encoder
# ─────────────────────────────────────────────────────────────────────────────
section("4 · GNN encoder  (gnn_encoder.py)")

try:
    from gnn_encoder import build_encoder, encode_state, FallbackMeanPoolEncoder
    encoder = build_encoder(hidden_dim=64, output_dim=128)
except ImportError as exc:
    print(f"  {YELLOW}SKIP{RESET}  gnn_encoder.py not found — "
          f"place it in the same directory. ({exc})")
    encoder = None

if encoder is not None:
    run("build_encoder() returns an object",
        lambda: encoder is not None)

    encoder_type = type(encoder).__name__
    print(f"\n  Using encoder: {CYAN}{encoder_type}{RESET}")

    if encoder_type == "FallbackMeanPoolEncoder":
        run("Fallback encoder output dim == 45",
            lambda: FallbackMeanPoolEncoder.OUTPUT_DIM == 45)

    scalars = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _test_encode():
        state = encode_state(
            encoder, graph,
            current_cost=cost, best_cost=cost, initial_cost=cost,
            iteration=0, max_iterations=200,
        )
        assert state is not None
        assert isinstance(state, np.ndarray), f"type={type(state)}"
        assert state.ndim == 1, f"ndim={state.ndim}, expected 1"
        assert state.dtype == np.float32, f"dtype={state.dtype}"
        assert not np.isnan(state).any(),  "NaN in encoded state"
        assert not np.isinf(state).any(),  "Inf in encoded state"
        return True

    run("encode_state() returns 1-D float32 array with no NaN/Inf", _test_encode)

    def _test_state_changes():
        state1 = encode_state(encoder, graph, cost, cost, cost, 0,   200)
        state2 = encode_state(encoder, graph, cost, cost, cost, 100, 200)
        # Iteration progress scalar should make states differ
        assert not np.allclose(state1, state2, atol=1e-6), \
            "State identical at iteration 0 and 100 — scalars not changing"
        return True

    run("State changes when search-progress scalars change", _test_state_changes)

    def _test_fixed_size_across_instances():
        # Build a minimal second solver with different instance size
        from e2e_vrp_loader import (TruckConfig, CityFreighterConfig,
                                     Depot, Satellite, Customer,
                                     RechargingStation, E2EVRPInstance)
        t2  = TruckConfig(3, 100, 1.0, 10.0)
        cf2 = CityFreighterConfig(3, 6, 30, 1.0, 5.0, 100, 0.1)
        d2  = Depot(0, 500, 500)
        s2  = [Satellite(1, 600, 600, 1.0, 200, 5.0)]
        c2  = [Customer(2+i, 400+i*10, 400+i*10, 5) for i in range(4)]
        r2  = [RechargingStation(10, 700, 700)]
        inst2  = E2EVRPInstance("tiny", t2, cf2, d2, s2, c2, r2)
        slv2   = LNSSolver(inst2, seed=0)
        slv2.assign_customers_to_satellites()
        sat2   = inst2.satellites[0]
        cids2  = slv2.sat_assignments[sat2.id]
        rts2   = slv2._construct_initial(sat2, cids2)
        rts2   = slv2._ensure_charging_feasibility(rts2)
        cost2  = slv2.calculate_total_cost(rts2)
        graph2 = slv2.get_hetero_state(rts2)

        s1 = encode_state(encoder, graph,  cost,  cost,  cost,  0, 200)
        s2_ = encode_state(encoder, graph2, cost2, cost2, cost2, 0, 200)
        assert s1.shape == s2_.shape, \
            f"State shapes differ: {s1.shape} vs {s2_.shape}"
        return True

    run("State dim identical across instances of different sizes",
        _test_fixed_size_across_instances)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — LNSMDPEnv reset and single step
# ─────────────────────────────────────────────────────────────────────────────
section("5 · LNSMDPEnv — reset + single step  (lns_env.py)")

try:
    from lns_env import LNSMDPEnv
except ImportError as exc:
    print(f"  {RED}FATAL{RESET}  Cannot import LNSMDPEnv: {exc}")
    sys.exit(1)

env = LNSMDPEnv(inst, sat, cids, max_steps=50, seed=42)

def _test_reset():
    s = env.reset()
    assert s is not None
    assert isinstance(s, (np.ndarray, dict)), f"Unexpected state type: {type(s)}"
    if isinstance(s, np.ndarray):
        assert s.ndim == 1,                "State not 1-D"
        assert s.dtype == np.float32,      "State not float32"
        assert not np.isnan(s).any(),      "NaN in initial state"
        assert not np.isinf(s).any(),      "Inf in initial state"
    return True

run("reset() returns a valid state",                _test_reset)
run("env.routes not empty after reset",            lambda: len(env.routes) > 0)
run("env.current_cost > 0 after reset",            lambda: env.current_cost > 0)
run("env.step_count == 0 after reset",             lambda: env.step_count == 0)

# Check initial_cost and best_cost exist (your fix)
run("env.initial_cost attribute exists",
    lambda: hasattr(env, "initial_cost") and env.initial_cost > 0)
run("env.best_cost attribute exists",
    lambda: hasattr(env, "best_cost") and env.best_cost > 0)

def _test_single_step():
    env.reset()
    s, r, done, info = env.step(0)   # action 0 = related nodes removal
    assert s is not None
    if isinstance(s, np.ndarray):
        assert s.ndim == 1,           "next_state not 1-D"
        assert not np.isnan(s).any(), "NaN in next_state"
    assert isinstance(r, float),      f"reward type={type(r)}"
    assert isinstance(done, bool),    f"done type={type(done)}"
    assert isinstance(info, dict),    f"info type={type(info)}"
    assert env.step_count == 1,       "step_count not incremented"
    return True

run("step(0) executes without crash", _test_single_step)

def _test_all_actions():
    for action in [0, 1, 2]:
        env.reset()
        s, r, done, info = env.step(action)
        assert s is not None, f"action {action} returned None state"
        assert isinstance(r, float), f"action {action} reward not float"
    return True

run("All 3 actions execute without crash", _test_all_actions)

def _test_reward_finite():
    env.reset()
    rewards = []
    for action in [0, 1, 2]:
        env.reset()
        _, r, _, _ = env.step(action)
        rewards.append(r)
    # Rewards should be finite (not ±inf, not NaN)
    assert all(math.isfinite(r) for r in rewards), \
        f"Non-finite rewards: {rewards}"
    return True

run("All action rewards are finite", _test_reward_finite)

def _test_physics_penalty():
    # The penalty for an infeasible move is -1000 in the original code
    # We just check that step never silently returns a None reward
    env.reset()
    _, r, _, _ = env.step(1)
    assert r is not None
    assert isinstance(r, (int, float))
    return True

run("Physics-invalid moves return a numeric penalty (not None)", _test_physics_penalty)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Full episode with random policy
# ─────────────────────────────────────────────────────────────────────────────
section("6 · Full episode — random policy for 200 steps")

def _run_full_episode():
    env2 = LNSMDPEnv(inst, sat, cids, max_steps=200, seed=99)
    state = env2.reset()
    assert state is not None

    rewards    = []
    costs      = []
    step_count = 0
    done       = False

    while not done:
        action = random.choice([0, 1, 2])
        state, reward, done, info = env2.step(action)
        rewards.append(reward)
        costs.append(info.get("current_cost", 0))
        step_count += 1

        # Safety: never infinite loop
        if step_count > 500:
            raise RuntimeError("Episode exceeded 500 steps — infinite loop?")

    assert step_count > 0,                   "Episode completed 0 steps"
    assert all(math.isfinite(r) for r in rewards), "Non-finite reward in episode"
    assert all(c > 0 for c in costs),         "Cost became 0 or negative"

    # State must still be valid at episode end
    if isinstance(state, np.ndarray):
        assert not np.isnan(state).any(), "NaN in terminal state"
        assert not np.isinf(state).any(), "Inf in terminal state"

    final_cost   = costs[-1]
    initial_cost = costs[0]
    improvement  = (initial_cost - final_cost) / initial_cost * 100

    print(f"\n  Steps      : {step_count}")
    print(f"  Init cost  : {initial_cost:.2f}")
    print(f"  Final cost : {final_cost:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Total reward: {sum(rewards):.4f}")

    return True

run("200-step random episode completes cleanly", _run_full_episode)

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — State consistency across episode
# ─────────────────────────────────────────────────────────────────────────────
section("7 · State vector consistency across full episode")

def _check_state_consistency():
    env3  = LNSMDPEnv(inst, sat, cids, max_steps=100, seed=7)
    state = env3.reset()
    if not isinstance(state, np.ndarray):
        print(f"  {YELLOW}SKIP{RESET}  State is dict "
              f"(encode_state not wired in yet) — shape test skipped")
        return True

    expected_dim = state.shape[0]
    issues       = []
    step         = 0

    done = False
    while not done and step < 100:
        action       = random.choice([0, 1, 2])
        state, r, done, _ = env3.step(action)
        step        += 1

        if state.shape[0] != expected_dim:
            issues.append(f"step {step}: dim={state.shape[0]}, expected={expected_dim}")
        if np.isnan(state).any():
            issues.append(f"step {step}: NaN detected")
        if np.isinf(state).any():
            issues.append(f"step {step}: Inf detected")
        if state.dtype != np.float32:
            issues.append(f"step {step}: dtype={state.dtype}")

    if issues:
        raise AssertionError(f"{len(issues)} issues — first: {issues[0]}")

    print(f"\n  State dim : {expected_dim}  (consistent over {step} steps)")
    return True

run("State dim, dtype, no NaN/Inf consistent across episode",
    _check_state_consistency)

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  RESULTS{RESET}")
print(f"{BOLD}{'='*60}{RESET}")
print(f"  {GREEN}PASSED{RESET} : {PASS_TOTAL}")
print(f"  {RED}FAILED{RESET} : {FAIL_TOTAL}")

if FAIL_TOTAL == 0:
    print(f"\n  {GREEN}{BOLD}All tests passed. Pipeline is coherent.{RESET}")
    sys.exit(0)
else:
    print(f"\n  {RED}{BOLD}{FAIL_TOTAL} test(s) failed. "
          f"Fix the FAIL lines above before training.{RESET}")
    sys.exit(1)