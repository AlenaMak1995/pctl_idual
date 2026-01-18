# -*- coding: utf-8 -*-
from __future__ import annotations
import random
from typing import Dict, Any
from pctl_idual.gridworld import GridWorld

# =========================
# DP helpers (ground truth)
# =========================

def value_iteration_shortest_path(mdp: GridWorld, gamma: float = 1.0,
                                  tol: float = 1e-6, max_iter: int = 10_000):
    # 0 at goal, large elsewhere
    V = {s: 0.0 if mdp.is_goal(s) else 1e6 for s in mdp.states}

    for _ in range(max_iter):
        delta = 0.0
        V_new = V.copy()

        for s in mdp.states:
            if mdp.is_goal(s):
                continue

            best = float("inf")
            for a in mdp.actions_from(s):
                # Expected Bellman backup:
                # Q(s,a) = sum_{s'} P(s'|s,a) * [ c(s,a,s') + gamma V(s') ]
                q = 0.0
                for s2, p in mdp.transitions(s, a).items():
                    r, c = s2
                    c_sa_s2 = mdp.cost_cell(r, c)   # cost of the realized next cell
                    q += p * (c_sa_s2 + gamma * V[s2])
                best = min(best, q)

            V_new[s] = best
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < tol:
            break

    return V


def greedy_policy_from_V(mdp: GridWorld, V, gamma: float = 1.0):
    pi = {}
    for s in mdp.states:
        if mdp.is_goal(s):
            pi[s] = None
            continue

        best_a, best_q = None, float("inf")
        for a in mdp.actions_from(s):
            q = 0.0
            for s2, p in mdp.transitions(s, a).items():
                r, c = s2
                c_sa_s2 = mdp.cost_cell(r, c)
                q += p * (c_sa_s2 + gamma * V[s2])

            if q < best_q:
                best_q, best_a = q, a

        pi[s] = best_a
    return pi


def simulate_policy(mdp: GridWorld, pi, max_steps=500, rng=None):
    rng = rng or random.Random(0)

    s = mdp.start
    traj = [s]

    for _ in range(max_steps):
        a = pi.get(s, None)
        if a is None:
            break

        trans = mdp.transitions(s, a)
        # sample s2 from trans
        u = rng.random()
        cum = 0.0
        s2 = None
        for sp, p in trans.items():
            cum += p
            if u <= cum:
                s2 = sp
                break
        if s2 is None:
            s2 = next(iter(trans))  # numerical fallback

        s = s2
        traj.append(s)

        if mdp.is_goal(s):
            break

    return traj


