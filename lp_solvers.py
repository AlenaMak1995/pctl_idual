# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Tuple
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import time
from gridworld import GridWorld, State, Action

# =========================
# LP for shortest path (no PCTL)
# =========================

def solve_shortest_path_lp_no_pctl(mdp: GridWorld):
    # Occupational measures x_{s,a}
    x = {
        (s, a): cp.Variable(nonneg=True)
        for s in mdp.states
        if not mdp.is_goal(s)
        for a in mdp.actions_from(s)
    }

    def in_expr(s: State):
        return sum(var for (sp, a), var in x.items() if mdp.move(sp, a) == s)

    def out_expr(s: State):
        return sum(var for (sp, a), var in x.items() if sp == s)

    # Flow constraints
    constraints = []
    for s in mdp.states:
        if mdp.is_goal(s):
            continue
        b = 1.0 if s == mdp.start else 0.0
        constraints.append(out_expr(s) - in_expr(s) == b)

    # Vectorized objective: minimize sum c(s,a) x(s,a)
    vars_list = []
    costs_list = []
    for (s, a), var in x.items():
        vars_list.append(var)
        costs_list.append(mdp.cost(s, a))
    # shape (n_vars,)
    vars_vec = cp.hstack(vars_list) 
    # shape (n_vars,)
    costs_vec = np.array(costs_list)           

    obj = cp.Minimize(costs_vec @ vars_vec)
    prob = cp.Problem(obj, constraints)

    t0 = time.perf_counter()
    prob.solve(solver=cp.HIGHS)
    t1 = time.perf_counter()
    solve_time = t1 - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("LP status:", prob.status)
        return None, None, solve_time

    x_opt = {(s, a): var.value for (s, a), var in x.items()}
    return obj.value, x_opt, solve_time

def solve_shortest_path_lp(mdp: GridWorld, solver: str = "MOSEK"):
    """
    LP for shortest path using a vectorized formulation:

        minimize   c^T x
        subject to A x = b,  x >= 0

    where x is over (state,action) pairs for non-goal states.
    """

    # 1) Enumerate all (s, a) that get a variable
    sa_list = []
    for s in mdp.states:
        if mdp.is_goal(s):
            continue
        for a in mdp.actions_from(s):
            sa_list.append((s, a))

    n_vars = len(sa_list)

    # map (s,a) -> index in x
    sa_index: Dict[Tuple[State, Action], int] = {
        (s, a): i for i, (s, a) in enumerate(sa_list)
    }

    # 2) State indexing
    state_index: Dict[State, int] = {s: i for i, s in enumerate(mdp.states)}
    n_states = len(mdp.states)

    # 3) Build A and b for flow constraints: out(s) - in(s) = b_s
    A = sp.lil_matrix((n_states, n_vars))
    b = np.zeros(n_states)

    for s in mdp.states:
        if mdp.is_goal(s):
            continue
        i = state_index[s]
        b[i] = 1.0 if s == mdp.start else 0.0

        for a in mdp.actions_from(s):
            j = sa_index[(s, a)]
            # out-flow from s
            A[i, j] += 1.0

            # in-flow to successor s2, but only if s2 is non-goal
            s2 = mdp.move(s, a)
            if not mdp.is_goal(s2):
                i2 = state_index[s2]
                A[i2, j] -= 1.0

    A = A.tocsr()

    # 4) Cost vector c
    costs = np.zeros(n_vars)
    for j, (s, a) in enumerate(sa_list):
        costs[j] = mdp.cost(s, a)

    # 5) CVXPY variable and problem
    x_vec = cp.Variable(n_vars, nonneg=True)

    obj = cp.Minimize(costs @ x_vec)
    constraints = [A @ x_vec == b]

    prob = cp.Problem(obj, constraints)

    t0 = time.perf_counter()
    if solver == "MOSEK":
        prob.solve(solver=cp.MOSEK)
    elif solver == "HIGHS":
        prob.solve(solver=cp.HIGHS)
    else:
        prob.solve()
    t1 = time.perf_counter()
    solve_time = t1 - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("LP status:", prob.status)
        return None, None, solve_time

    x_val = x_vec.value
    # Rebuild dict x_opt[(s,a)] for compatibility 
    x_opt: Dict[Tuple[State, Action], float] = {}
    for j, (s, a) in enumerate(sa_list):
        x_opt[(s, a)] = x_val[j]

    return float(prob.value), x_opt, solve_time

def recover_policy_from_x(mdp: GridWorld, x_opt, tol=1e-8):
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


def print_policy_grid(mdp: GridWorld, policy, G2=None, G3=None):
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for r in range(mdp.N):
        row = ""
        for c in range(mdp.N):
            s = (r, c)
            if s == mdp.start:
                row += " S  "
            elif s in mdp.goal:
                row += " G  "
            # elif G2 is not None and s in G2:
            #     row += " 2  "
            # elif G3 is not None and s in G3:
            #     row += " 3  "    
            elif s not in policy or not policy[s]:
                row += " ·  "
            else:
                best_a = max(policy[s], key=lambda a: policy[s][a])
                if policy[s][best_a] < 1e-6:
                    row += " ·  "
                else:
                    row += f" {arrow[best_a]}  "
        print(row)

               

