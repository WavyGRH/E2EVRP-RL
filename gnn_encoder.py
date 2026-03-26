"""
gnn_encoder.py
==============
Heterogeneous GNN encoder for E2EVRP operator selection.

Consumes the raw graph dict produced by LNSSolver.get_hetero_state()
and outputs a fixed-size state vector for the CQL Q-network.

Architecture
------------
  Input : hetero graph dict  (4 node types, 5 edge relation types)
  Layer 1-2 : HeteroConv(HGTConv per relation) — message passing
  Pooling   : Attention-weighted mean pool over Customer nodes
  Output    : state_dim-dimensional vector  (graph_feat ++ scalar_feat)

Dependencies
------------
  pip install torch torch_geometric

Node types   : Depot, Satellite, Customer, RechargingStation
Edge types   : Customer-near-Customer, Customer-near-Satellite,
               Satellite-serves-Customer, Customer-follows-Customer,
               *-near-RechargingStation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

# ── PyG imports ──────────────────────────────────────────────────────────────
try:
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[gnn_encoder] WARNING: torch_geometric not installed. "
          "Using fallback mean-pool encoder (no learned message passing).")


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match get_hetero_state() output
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPES    = ["Depot", "Satellite", "Customer", "RechargingStation"]
NODE_FEAT_DIM = 10          # dimension of each node's raw feature vector
SCALAR_DIM    = 5           # number of search-progress scalars
N_SCALAR      = SCALAR_DIM  # alias used in forward()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: convert raw numpy graph dict → PyG HeteroData
# ─────────────────────────────────────────────────────────────────────────────

def dict_to_heterodata(graph_dict: dict) -> "HeteroData":
    """
    Convert the dict returned by LNSSolver.get_hetero_state()
    into a PyG HeteroData object ready for GNN forward pass.

    Parameters
    ----------
    graph_dict : dict
        Output of get_hetero_state(). Keys are either node type strings
        (mapping to {"x": np.ndarray}) or (src, rel, dst) tuples
        (mapping to {"edge_index": np.ndarray, "edge_attr": np.ndarray}).

    Returns
    -------
    HeteroData
    """
    data = HeteroData()

    # Node features
    for ntype in NODE_TYPES:
        if ntype in graph_dict:
            x = graph_dict[ntype]["x"]              # (N_type, 10)
            if x.shape[0] == 0:
                # PyG needs at least one node; insert a dummy zero row
                x = np.zeros((1, NODE_FEAT_DIM), dtype=np.float32)
            data[ntype].x = torch.from_numpy(x)

    # Edge indices and attributes
    for key, val in graph_dict.items():
        if not isinstance(key, tuple):
            continue
        src_type, rel, dst_type = key
        ei   = val["edge_index"]                    # (2, E)
        attr = val["edge_attr"]                     # (E, 1)
        if ei.shape[1] == 0:
            continue                                # skip empty edge sets
        data[src_type, rel, dst_type].edge_index = torch.from_numpy(ei)
        data[src_type, rel, dst_type].edge_attr  = torch.from_numpy(attr)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Core GNN model
# ─────────────────────────────────────────────────────────────────────────────

class E2EVRPHeteroGNN(nn.Module):
    """
    Two-layer heterogeneous GNN encoder.

    Each HeteroConv layer applies a separate SAGEConv per edge relation type,
    then sums the messages arriving at each node.  After two rounds of
    message passing, an attention-based pooling over Customer nodes produces
    the graph-level embedding.  Scalar search-progress features are fused in
    via a separate projection and concatenated before the final output layer.

    Parameters
    ----------
    hidden_dim  : dimension of intermediate node embeddings
    output_dim  : dimension of the final state vector fed to CQL
    n_layers    : number of HeteroConv layers (default 2)
    dropout     : dropout rate applied after each conv layer
    """

    def __init__(
        self,
        hidden_dim: int  = 64,
        output_dim: int  = 128,
        n_layers:   int  = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert PYG_AVAILABLE, (
            "torch_geometric is required for E2EVRPHeteroGNN. "
            "Install with: pip install torch_geometric"
        )

        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        self.n_layers    = n_layers
        self.dropout_p   = dropout

        # ── Input projections: one per node type ────────────────────────────
        # Each node type has 10 raw features → hidden_dim
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(NODE_FEAT_DIM, hidden_dim)
            for ntype in NODE_TYPES
        })

        # ── HeteroConv layers ────────────────────────────────────────────────
        # We define the same set of edge relations at every layer.
        # SAGEConv(in, out) handles variable neighbourhood sizes cleanly.
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv(
                {
                    ("Customer",  "near",    "Customer"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                    ("Customer",  "near",    "Satellite"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                    ("Satellite", "serves",  "Customer"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                    ("Customer",  "follows", "Customer"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                    ("Depot",     "near",    "RechargingStation"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                    ("Satellite", "near",    "RechargingStation"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                    ("Customer",  "near",    "RechargingStation"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                    ("RechargingStation", "near", "RechargingStation"):
                        SAGEConv(hidden_dim, hidden_dim, normalize=True),
                },
                aggr="sum",     # how to combine messages from different relations
            )
            self.convs.append(conv)

        # Layer norms — one per node type per layer
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in NODE_TYPES})
            for _ in range(n_layers)
        ])

        # ── Attention pooling over Customer nodes ────────────────────────────
        # Produces a single graph-level vector from variable-length Customer set
        self.attn_pool = nn.Linear(hidden_dim, 1)

        # ── Scalar feature projection ────────────────────────────────────────
        self.scalar_proj = nn.Sequential(
            nn.Linear(SCALAR_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Output projection ────────────────────────────────────────────────
        # Concatenate pooled graph embedding + scalar projection → output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        data:    "HeteroData",
        scalars: torch.Tensor,          # (batch, SCALAR_DIM) or (SCALAR_DIM,)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        data    : HeteroData produced by dict_to_heterodata()
        scalars : 1-D or 2-D tensor of search-progress features
                  [cost_ratio, best_ratio, gap, progress, stagnation]

        Returns
        -------
        state : (output_dim,) tensor — fixed-size state for CQL
        """
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)          # (1, SCALAR_DIM)

        # ── 1. Input projection ──────────────────────────────────────────────
        x_dict = {}
        for ntype in NODE_TYPES:
            if hasattr(data[ntype], "x") and data[ntype].x is not None:
                x_dict[ntype] = F.relu(self.input_proj[ntype](data[ntype].x))
            else:
                # Node type absent in this instance — use a zero placeholder
                x_dict[ntype] = torch.zeros(
                    1, self.hidden_dim, device=scalars.device
                )

        # ── 2. Message passing layers ────────────────────────────────────────
        for layer_idx, conv in enumerate(self.convs):
            # Build edge_index_dict for only the edges present in data
            edge_index_dict = {}
            for key in data.edge_types:
                src, rel, dst = key
                if data[src, rel, dst].edge_index.shape[1] > 0:
                    edge_index_dict[key] = data[src, rel, dst].edge_index

            new_x = conv(x_dict, edge_index_dict)

            # Residual + LayerNorm + Dropout
            norms = self.layer_norms[layer_idx]
            for ntype in NODE_TYPES:
                if ntype in new_x and ntype in x_dict:
                    # Residual connection (shapes match because hidden_dim is fixed)
                    out = new_x[ntype] + x_dict[ntype]
                    out = norms[ntype](out)
                    out = F.dropout(out, p=self.dropout_p, training=self.training)
                    x_dict[ntype] = out

        # ── 3. Attention pooling over Customer nodes ─────────────────────────
        cust_emb = x_dict["Customer"]               # (N_cust, hidden_dim)

        if cust_emb.shape[0] > 1:
            attn_logits  = self.attn_pool(cust_emb).squeeze(-1)   # (N_cust,)
            attn_weights = torch.softmax(attn_logits, dim=0)       # (N_cust,)
            graph_vec    = (attn_weights.unsqueeze(-1) * cust_emb).sum(dim=0)  # (hidden_dim,)
        else:
            graph_vec = cust_emb.squeeze(0)          # single customer edge case

        # ── 4. Scalar projection ─────────────────────────────────────────────
        scalar_vec = self.scalar_proj(scalars).squeeze(0)   # (hidden_dim,)

        # ── 5. Fuse and project ──────────────────────────────────────────────
        combined = torch.cat([graph_vec, scalar_vec], dim=-1)   # (hidden_dim*2,)
        state    = self.output_proj(combined)                   # (output_dim,)

        return state                                            # (output_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: no-PyG mean-pool encoder (numpy only, no learned weights)
# ─────────────────────────────────────────────────────────────────────────────

class FallbackMeanPoolEncoder:
    """
    Drop-in replacement for E2EVRPHeteroGNN when torch_geometric is not
    installed.  Does not learn — useful for validating the pipeline before
    installing PyG.

    Output dim = 4 * NODE_FEAT_DIM + SCALAR_DIM = 45
    """

    OUTPUT_DIM = 4 * NODE_FEAT_DIM + SCALAR_DIM

    def __call__(
        self,
        graph_dict: dict,
        scalars:    np.ndarray,
    ) -> np.ndarray:
        parts = []
        for ntype in NODE_TYPES:
            x = graph_dict.get(ntype, {}).get("x", np.zeros((1, NODE_FEAT_DIM)))
            parts.append(x.mean(axis=0) if x.shape[0] > 0
                         else np.zeros(NODE_FEAT_DIM, dtype=np.float32))
        graph_vec = np.concatenate(parts)               # (40,)
        return np.concatenate([graph_vec, scalars]).astype(np.float32)  # (45,)


# ─────────────────────────────────────────────────────────────────────────────
# Public factory — returns the right encoder based on environment
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder(
    hidden_dim: int = 64,
    output_dim: int = 128,
    n_layers:   int = 2,
    dropout:    float = 0.1,
):
    """
    Returns E2EVRPHeteroGNN if PyG is available, otherwise FallbackMeanPoolEncoder.
    Both expose the same interface:
        encoder(graph_dict_or_heterodata, scalars) → fixed-size vector
    """
    if PYG_AVAILABLE:
        return E2EVRPHeteroGNN(hidden_dim, output_dim, n_layers, dropout)
    return FallbackMeanPoolEncoder()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: encode a single transition from lns_env
# ─────────────────────────────────────────────────────────────────────────────

def encode_state(
    encoder:        "E2EVRPHeteroGNN | FallbackMeanPoolEncoder",
    graph_dict:     dict,
    current_cost:   float,
    best_cost:      float,
    initial_cost:   float,
    iteration:      int,
    max_iterations: int,
) -> np.ndarray:
    """
    Full pipeline: raw graph dict → fixed-size numpy state vector.
    Used inside lns_env.py at every step.

    Parameters
    ----------
    encoder       : returned by build_encoder()
    graph_dict    : output of LNSSolver.get_hetero_state()
    current_cost  : cost of the current solution
    best_cost     : best cost seen so far in this episode
    initial_cost  : cost of the initial greedy solution (episode baseline)
    iteration     : current LNS iteration number
    max_iterations: total budget for this episode

    Returns
    -------
    state : (output_dim,) numpy float32 array
    """
    eps = 1e-8
    scalars = np.array([
        current_cost  / (initial_cost + eps),
        best_cost     / (initial_cost + eps),
        (current_cost - best_cost) / (initial_cost + eps),
        iteration     / max(max_iterations, 1),
        (current_cost - best_cost) / (best_cost + eps),
    ], dtype=np.float32)

    if isinstance(encoder, FallbackMeanPoolEncoder):
        return encoder(graph_dict, scalars)

    # PyG path
    data    = dict_to_heterodata(graph_dict)
    scalars_t = torch.from_numpy(scalars)
    with torch.no_grad():
        state = encoder(data, scalars_t)
    return state.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))

    from e2e_vrp_loader import load_instance
    from lns_solver import LNSSolver

    path = os.path.join(os.path.dirname(__file__),
                        "Set2", "E-Set2a_E-n22-k4-s10-14_int.dat")
    inst   = load_instance(path)
    solver = LNSSolver(inst, seed=42)
    solver.assign_customers_to_satellites()

    sat    = inst.satellites[0]
    cids   = solver.sat_assignments[sat.id]
    routes = solver._construct_initial(sat, cids)
    routes = solver._ensure_charging_feasibility(routes)
    cost   = solver.calculate_total_cost(routes)

    graph_dict = solver.get_hetero_state(routes)
    encoder    = build_encoder(hidden_dim=64, output_dim=128)

    state = encode_state(
        encoder, graph_dict,
        current_cost=cost, best_cost=cost, initial_cost=cost,
        iteration=0, max_iterations=200,
    )

    print(f"Encoder type  : {type(encoder).__name__}")
    print(f"State shape   : {state.shape}")
    print(f"State dtype   : {state.dtype}")
    print(f"State range   : [{state.min():.4f}, {state.max():.4f}]")
    print(f"Any NaN/Inf   : {np.isnan(state).any() or np.isinf(state).any()}")
    print()

    if PYG_AVAILABLE:
        # Check that gradients flow through the full encoder
        enc    = E2EVRPHeteroGNN(hidden_dim=64, output_dim=128)
        data   = dict_to_heterodata(graph_dict)
        scl    = torch.tensor([[cost/cost, 1.0, 0.0, 0.0, 0.0]])
        out    = enc(data, scl)
        loss   = out.sum()
        loss.backward()
        print(f"Gradient check: output={out.shape}, loss backward OK")
        n_params = sum(p.numel() for p in enc.parameters())
        print(f"Total params  : {n_params:,}")