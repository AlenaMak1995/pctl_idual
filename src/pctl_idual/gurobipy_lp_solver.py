from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sp
import time

import gurobipy as gp
from gurobipy import GRB

from .gridworld import GridWorld, State, Action


def solve_shortest_path_lp_gurobi(mdp: GridWorld, verbose: bool = False,env=None):
    """
    Shortest-path LP using gurobipy directly:

        minimize   c^T x
        subject to A x = b
                   x >= 0

    Variables x correspond to (s,a) for non-goal states s.
    Returns: (obj_value, x_opt_dict, solve_time)
    """

    # 1) Enumerate variables over (s,a) for non-goal states
    sa_list: list[Tuple[State, Action]] = []
    for s in mdp.states:
        if mdp.is_goal(s):
            continue
        for a in mdp.actions_from(s):
            sa_list.append((s, a))
    n_vars = len(sa_list)

    sa_index: Dict[Tuple[State, Action], int] = {(s, a): j for j, (s, a) in enumerate(sa_list)}

    # 2) State indexing
    state_index: Dict[State, int] = {s: i for i, s in enumerate(mdp.states)}
    n_states = len(mdp.states)

    # 3) Build sparse A and b for flow constraints: out(s) - in(s) = b_s
    A = sp.lil_matrix((n_states, n_vars), dtype=float)
    b = np.zeros(n_states, dtype=float)

    for s in mdp.states:
        if mdp.is_goal(s):
            continue
        i = state_index[s]
        b[i] = 1.0 if s == mdp.start else 0.0

        for a in mdp.actions_from(s):
            j = sa_index[(s, a)]
            A[i, j] += 1.0

            s2 = mdp.move(s, a)
            if not mdp.is_goal(s2):
                i2 = state_index[s2]
                A[i2, j] -= 1.0

    A = A.tocsr()

    # 4) Cost vector c
    c = np.array([mdp.cost(s, a) for (s, a) in sa_list], dtype=float)

    # 5) Build and solve Gurobi model
    t0 = time.perf_counter()

    m = gp.Model("shortest_path_lp", env=env) if env is not None else gp.Model("shortest_path_lp")
    m.Params.OutputFlag = 1 if verbose else 0

    # Optional knobs 
    # m.Params.Presolve = 2
    # m.Params.Method = 2       # 2=barrier, 1=dual simplex, 0=primal simplex, -1=auto
    # m.Params.Crossover = 0    
    # m.Params.NumericFocus = 1

    x = m.addMVar(shape=n_vars, lb=0.0, name="x")
    m.setObjective(c @ x, GRB.MINIMIZE)

    # Add A x = b row-by-row from CSR (memory-safe)
    for i in range(n_states):
        start, end = A.indptr[i], A.indptr[i + 1]
        cols = A.indices[start:end]
        vals = A.data[start:end]

        if cols.size == 0:
            # 0 == b[i]
            m.addConstr(0.0 == float(b[i]))
        else:
            # vals @ x[cols] is a linear expr
            m.addConstr(vals @ x[cols] == float(b[i]))

    m.optimize()
    t1 = time.perf_counter()
    solve_time = t1 - t0

    if m.Status != GRB.OPTIMAL:
        print("Gurobi status:", m.Status)
        return None, None, solve_time

    x_val = x.X  # numpy array length n_vars

    # Build dict x_opt[(s,a)] for compatibility with the code
    x_opt: Dict[Tuple[State, Action], float] = {
        (s, a): float(x_val[j]) for j, (s, a) in enumerate(sa_list)
    }

    return float(m.ObjVal), x_opt, solve_time


def recover_policy_from_x_gurobi(mdp: GridWorld, x_opt, tol=1e-8):
    policy = {}
    for s in mdp.states:
        if mdp.is_goal(s):
            continue
        flows = [(a, x_opt.get((s, a), 0.0)) for a in mdp.actions_from(s)]
        total = sum(v for _, v in flows)
        if total > tol:
            policy[s] = {a: v / total for a, v in flows}
        else:
            policy[s] = {a: 0.0 for a in mdp.actions_from(s)}
    return policy


def print_policy_grid_gurobi(mdp: GridWorld, policy, G2=None, G3=None):
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for r in range(mdp.N):
        row = ""
        for c in range(mdp.N):
            s = (r, c)
            if s == mdp.start:
                row += " S  "
            elif s in mdp.goal:
                row += " G  "
            elif s not in policy or not policy[s]:
                row += " ·  "
            else:
                best_a = max(policy[s], key=lambda a: policy[s][a])
                if policy[s][best_a] < 1e-6:
                    row += " ·  "
                else:
                    row += f" {arrow[best_a]}  "
        print(row)
