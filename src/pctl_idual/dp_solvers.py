# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from pctl_idual.gridworld import GridWorld

# =========================
# DP helpers (ground truth)
# =========================

def value_iteration_shortest_path(
    mdp: GridWorld, gamma: float = 1.0,
    tol: float = 1e-6, max_iter: int = 1000
):
    V = {s: 0.0 if mdp.is_goal(s) else 1e6 for s in mdp.states}

    for _ in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in mdp.states:
            if mdp.is_goal(s):
                continue
            qs = []
            for a in mdp.actions_from(s):
                s2 = mdp.move(s, a)
                qs.append(mdp.cost(s, a) + gamma * V[s2])
            if qs:
                V_new[s] = min(qs)
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
            s2 = mdp.move(s, a)
            q = mdp.cost(s, a) + gamma * V[s2]
            if q < best_q:
                best_q, best_a = q, a
        pi[s] = best_a
    return pi


def simulate_policy(mdp: GridWorld, pi, max_steps=500):
    s = mdp.start
    traj = [s]
    for _ in range(max_steps):
        a = pi[s]
        if a is None:
            break
        s = mdp.move(s, a)
        traj.append(s)
        if mdp.is_goal(s):
            break
    return traj

