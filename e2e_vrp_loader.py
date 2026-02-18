"""
E2E VRP Instance Loader & Data Structures
Based on Breunig, Baldacci, Hartl & Vidal (2019)
"The Electric Two-echelon Vehicle Routing Problem"

Parses .dat instance files into structured Python objects.
"""

import os
from dataclasses import dataclass
from typing import List


# ==========================================
# Data Structures
# ==========================================

@dataclass
class TruckConfig:
    count: int
    capacity: int
    cost_per_distance: float
    fixed_cost: float


@dataclass
class CityFreighterConfig:
    max_per_satellite: int
    total_count: int
    capacity: int
    cost_per_distance: float
    fixed_cost: float
    max_charge: int
    energy_consumption: float


@dataclass
class Node:
    id: int
    x: int
    y: int


@dataclass
class Depot(Node):
    pass


@dataclass
class Satellite(Node):
    handling_cost: float
    max_capacity: int
    fixed_cost: float


@dataclass
class Customer(Node):
    demand: int


@dataclass
class RechargingStation(Node):
    pass


@dataclass
class E2EVRPInstance:
    name: str
    truck_config: TruckConfig
    city_freighter_config: CityFreighterConfig
    depot: Depot
    satellites: List[Satellite]
    customers: List[Customer]
    recharging_stations: List[RechargingStation]


# ==========================================
# Parsing Utilities
# ==========================================

def parse_line_values(line: str) -> List[str]:
    """Split a line by whitespace or commas."""
    return line.replace(',', ' ').split()


def load_instance(file_path: str) -> E2EVRPInstance:
    """
    Load an E2E VRP instance from a .dat file.

    Node IDs are assigned sequentially:
      0           -> Depot
      1..S        -> Satellites
      S+1..S+C    -> Customers
      S+C+1..end  -> Recharging Stations
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Filter out comment lines
    data_lines = [line for line in lines if not line.startswith('!')]

    # --- Line 1: Truck configuration ---
    truck_vals = parse_line_values(data_lines[0])
    truck_config = TruckConfig(
        count=int(truck_vals[0]),
        capacity=int(truck_vals[1]),
        cost_per_distance=float(truck_vals[2]),
        fixed_cost=float(truck_vals[3])
    )

    # --- Line 2: City Freighter configuration ---
    cf_vals = parse_line_values(data_lines[1])
    cf_config = CityFreighterConfig(
        max_per_satellite=int(cf_vals[0]),
        total_count=int(cf_vals[1]),
        capacity=int(cf_vals[2]),
        cost_per_distance=float(cf_vals[3]),
        fixed_cost=float(cf_vals[4]),
        max_charge=int(cf_vals[5]),
        energy_consumption=float(cf_vals[6])
    )

    # --- Line 3: Depot + Satellites ---
    stores_vals = parse_line_values(data_lines[2])
    depot = Depot(id=0, x=int(stores_vals[0]), y=int(stores_vals[1]))

    satellites = []
    sat_data = stores_vals[2:]
    num_sat_params = 5
    for i in range(0, len(sat_data), num_sat_params):
        if i + num_sat_params <= len(sat_data):
            sat_id = len(satellites) + 1
            satellites.append(Satellite(
                id=sat_id,
                x=int(sat_data[i]),
                y=int(sat_data[i + 1]),
                handling_cost=float(sat_data[i + 2]),
                max_capacity=int(sat_data[i + 3]),
                fixed_cost=float(sat_data[i + 4])
            ))

    # --- Line 4: Customers ---
    cust_vals = parse_line_values(data_lines[3])
    customers = []
    num_cust_params = 3
    current_id_offset = 1 + len(satellites)
    for i in range(0, len(cust_vals), num_cust_params):
        if i + num_cust_params <= len(cust_vals):
            customers.append(Customer(
                id=current_id_offset + (i // num_cust_params),
                x=int(cust_vals[i]),
                y=int(cust_vals[i + 1]),
                demand=int(cust_vals[i + 2])
            ))

    # --- Line 5: Recharging Stations ---
    rs_vals = parse_line_values(data_lines[4])
    recharging_stations = []
    num_rs_params = 2
    rs_id_offset = current_id_offset + len(customers)
    for i in range(0, len(rs_vals), num_rs_params):
        if i + num_rs_params <= len(rs_vals):
            recharging_stations.append(RechargingStation(
                id=rs_id_offset + (i // num_rs_params),
                x=int(rs_vals[i]),
                y=int(rs_vals[i + 1])
            ))

    return E2EVRPInstance(
        name=os.path.basename(file_path),
        truck_config=truck_config,
        city_freighter_config=cf_config,
        depot=depot,
        satellites=satellites,
        customers=customers,
        recharging_stations=recharging_stations
    )


# ==========================================
# Quick self-test
# ==========================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(os.path.dirname(__file__),
                            "Set2", "E-Set2a_E-n22-k4-s10-14_int.dat")

    inst = load_instance(path)
    print(f"Instance : {inst.name}")
    print(f"Trucks   : {inst.truck_config}")
    print(f"CFs      : {inst.city_freighter_config}")
    print(f"Depot    : ({inst.depot.x}, {inst.depot.y})")
    print(f"Satellites ({len(inst.satellites)}): {[(s.id, s.x, s.y) for s in inst.satellites]}")
    print(f"Customers  ({len(inst.customers)}): first 5 = {[(c.id, c.x, c.y, c.demand) for c in inst.customers[:5]]}")
    print(f"Recharging ({len(inst.recharging_stations)}): {[(r.id, r.x, r.y) for r in inst.recharging_stations[:5]]}")
