# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Tuple, Set, List, Optional
import cvxpy as cp
import numpy as np
import time

from pctl_idual.gridworld import GridWorld, State, Action  

AugState = Tuple  
Region = Set[State]

# =========================
# Flag / PCTL specs
# =========================

@dataclass
class RegionFlagSpec:
    """Flag that becomes 1 if we ever visit `region`."""
    name: str
    region: Region


@dataclass
class PCTLRegionConstraint:
    """
    Reachability-style constraints on simple flags:
      - P(ever visit region_i) <= bound  -> kind = 'visit_region_max'
      - P(ever visit region_i) >= bound  -> kind = 'visit_region_min'
    """
    kind: str  
    region_name: str
    bound: float


@dataclass
class UntilSpec:
    """
    Formula:  name :   A U B

    A_region = states where A holds
    B_region = states where B holds

    Semantics: along a run, the formula succeeds if agent hits B
    while A has held at all previous steps; fails if agent ever
    leaves A before hitting B.
    """
    name: str
    A_region: Region
    B_region: Region


@dataclass
class UntilConstraint:
    """
    Constraints on an UntilSpec:

      kind = "until_min":  P( A U B ) >= bound
      kind = "until_max":  P( A U B ) <= bound
    """
    kind: str  
    spec_name: str
    bound: float


# =========================
# Augmented MDP
# =========================

@dataclass
class AugmentedMDP:
    """
    Augmented MDP = base GridWorld × bits:

      - first len(flags) bits: "have we visited region i?"
      - for each UntilSpec name: 2 bits (success, fail)
          success = 1  if A U B has been satisfied
          fail    = 1  if A U B has been violated
    """
    base: GridWorld
    flags: List[RegionFlagSpec]
    until_specs: List[UntilSpec] = None

    def __post_init__(self):
        if self.until_specs is None:
            self.until_specs = []

        # indices for simple region flags
        self.flag_indices = {f.name: idx for idx, f in enumerate(self.flags)}

        # indices for until success/fail bits
        self.until_success_idx: Dict[str, int] = {}
        self.until_fail_idx: Dict[str, int] = {}

        offset = len(self.flags)
        for i, uspec in enumerate(self.until_specs):
            self.until_success_idx[uspec.name] = offset + 2 * i
            self.until_fail_idx[uspec.name] = offset + 2 * i + 1

        total_bits = len(self.flags) + 2 * len(self.until_specs)
        self.states_aug: List[AugState] = []

        # all 0/1 combinations for all bits
        if total_bits > 0:
            Z_space = list(np.ndindex(*(2 for _ in range(total_bits))))
        else:
            Z_space = [()]

        for s in self.base.states:
            for z in Z_space:
                self.states_aug.append((s,) + tuple(z))

    def is_absorbing_aug(self, st: AugState) -> bool:
        return self.base.is_goal(st[0])

    def actions_from_aug(self, st: AugState) -> List[Action]:
        if self.is_absorbing_aug(st):
            return []
        return self.base.actions_from(st[0])

    def move_aug(self, st: AugState, a: Action) -> AugState:
        s = st[0]
        bits = list(st[1:])
        s2 = self.base.move(s, a)

        # --- simple "visited region" flags ---
        for i, spec in enumerate(self.flags):
            if s2 in spec.region:
                bits[i] = 1

        # --- A U B formula bits ---
        for uspec in self.until_specs:
            i_succ = self.until_success_idx[uspec.name]
            i_fail = self.until_fail_idx[uspec.name]

            # already decided for this formula?
            if bits[i_succ] == 1 or bits[i_fail] == 1:
                continue

            if s2 in uspec.B_region:
                # hit B while still "good so far" → success
                bits[i_succ] = 1
            elif s2 not in uspec.A_region:
                # left A before hitting B → fail
                bits[i_fail] = 1
            # else: still in A, not in B → ongoing

        return (s2, *bits)

    def cost_aug(self, st: AugState, a: Action) -> float:
        return self.base.cost(st[0], a)

    @property
    def start_aug(self) -> AugState:
        total_bits = len(self.flags) + 2 * len(self.until_specs)
        if total_bits == 0:
            return (self.base.start,)
        return (self.base.start,) + tuple(0 for _ in range(total_bits))


# =========================
# Global LP with PCTL + Until
# =========================

def solve_lp_with_pctl_aug(
    mdp_aug: AugmentedMDP,
    p_goal_min: float,
    region_constraints: List[PCTLRegionConstraint],
    until_constraints: List[UntilConstraint],
):
    """
    Global LP over augmented MDP with:

      - P(true U GOAL) >= p_goal_min
      - region constraints on "ever visit region" flags
      - until constraints on P(A U B) for each UntilSpec

    Vectorized version:
      * one variable vector x_vec (size = #edges)
      * flow constraints A_flow @ x_vec = b
      * probability constraints via coefficient vectors
    """

    # -------------------------------------------------------
    # 1) Enumerate non-absorbing augmented states and edges
    # -------------------------------------------------------
    non_abs_states = [st for st in mdp_aug.states_aug
                      if not mdp_aug.is_absorbing_aug(st)]
    state_index = {st: i for i, st in enumerate(non_abs_states)}

    # edges = list of (st, a) from non-absorbing states
    edges: List[Tuple[AugState, Action]] = []
    for st in non_abs_states:
        for a in mdp_aug.actions_from_aug(st):
            edges.append((st, a))

    E = len(edges)
    if E == 0:
        # Degenerate case: no edges
        return 0.0, 0.0, {}, {}, {}, 0.0

    # Decision variables: one nonnegative occupational measure per edge
    x_vec = cp.Variable(E, nonneg=True)

    # -------------------------------------------------------
    # 2) Build flow conservation matrix A_flow and RHS b
    #     out(s) - in(s) = b_s   for all non-absorbing s
    # -------------------------------------------------------
    S = len(non_abs_states)
    A_flow = np.zeros((S, E))
    b = np.zeros(S)

    for e, (st, a) in enumerate(edges):
        i_from = state_index[st]
        A_flow[i_from, e] += 1.0  # out(s)

        st2 = mdp_aug.move_aug(st, a)
        # only subtract from row if successor is non-absorbing,
        # to match the original code that skipped the flow equation for absorbing states
        if (not mdp_aug.is_absorbing_aug(st2)) and (st2 in state_index):
            i_to = state_index[st2]
            A_flow[i_to, e] -= 1.0  # in(s)

    # RHS: +1 at start_aug row, 0 elsewhere (if start is non-absorbing)
    start_idx = state_index.get(mdp_aug.start_aug, None)
    if start_idx is not None:
        b[start_idx] = 1.0

    constraints = [A_flow @ x_vec == b]

    # -------------------------------------------------------
    # 3) Build coefficient vectors for probabilities
    #    goal_prob, region_flag_prob[name], until_prob[name]
    # -------------------------------------------------------
    goal_coeff = np.zeros(E)
    region_coeffs: Dict[str, np.ndarray] = {
        spec.name: np.zeros(E) for spec in mdp_aug.flags
    }
    until_coeffs: Dict[str, np.ndarray] = {
        uspec.name: np.zeros(E) for uspec in mdp_aug.until_specs
    }

    for e, (stp, a) in enumerate(edges):
        st2 = mdp_aug.move_aug(stp, a)
        s2 = st2[0]

        if s2 in mdp_aug.base.goal:
            # this edge's flow contributes to "reach goal"
            goal_coeff[e] = 1.0
            bits2 = st2[1:]

            # region flags: "ever visit region_i"
            for spec in mdp_aug.flags:
                idx = mdp_aug.flag_indices[spec.name]
                if bits2[idx] == 1:
                    region_coeffs[spec.name][e] = 1.0

            # Until formulas: success && not fail
            for uspec in mdp_aug.until_specs:
                i_succ = mdp_aug.until_success_idx[uspec.name]
                i_fail = mdp_aug.until_fail_idx[uspec.name]
                if bits2[i_succ] == 1 and bits2[i_fail] == 0:
                    until_coeffs[uspec.name][e] = 1.0

    # Turn coefficient vectors into CVXPY expressions
    goal_prob = goal_coeff @ x_vec
    region_flag_prob: Dict[str, cp.Expression] = {
        name: coeff @ x_vec for name, coeff in region_coeffs.items()
    }
    until_prob: Dict[str, cp.Expression] = {
        name: coeff @ x_vec for name, coeff in until_coeffs.items()
    }

    # Enforce P(true U GOAL) >= p_goal_min
    constraints.append(goal_prob >= p_goal_min)

    # Region constraints
    for c in region_constraints:
        expr = region_flag_prob[c.region_name]
        if c.kind == "visit_region_max":
            constraints.append(expr <= c.bound)
        elif c.kind == "visit_region_min":
            constraints.append(expr >= c.bound)
        else:
            raise ValueError(f"Unknown region constraint kind {c.kind}")

    # Until constraints
    for c in until_constraints:
        expr = until_prob[c.spec_name]
        if c.kind == "until_min":
            constraints.append(expr >= c.bound)
        elif c.kind == "until_max":
            constraints.append(expr <= c.bound)
        else:
            raise ValueError(f"Unknown until constraint kind {c.kind}")

    # -------------------------------------------------------
    # 4) Objective: minimize expected cost  c^T x_vec
    # -------------------------------------------------------
    cost_vec = np.array([
        mdp_aug.cost_aug(st, a) for (st, a) in edges
    ])
    obj = cp.Minimize(cost_vec @ x_vec)
    # -------------------------------------------------------
    # Debug: problem size (global LP)
    # -------------------------------------------------------
    num_nonabs_states = len(non_abs_states)
    num_edges = len(edges)         
    num_constraints = len(constraints)

    # print("=== Global LP size (PCTL+until) ===")
    # print(f"  non-absorbing aug states : {num_nonabs_states}")
    # print(f"  edges / x-vars           : {num_edges}")
    # print(f"  constraints              : {num_constraints}")

    # -------------------------------------------------------
    # 5) Solve
    # -------------------------------------------------------
    prob = cp.Problem(obj, constraints)
    t0 = time.perf_counter()
    prob.solve()
    t1 = time.perf_counter()
    solve_time = t1 - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("LP status:", prob.status)
        return None, None, None, None, None, solve_time

    x_val = x_vec.value
    # Map back to {(aug_state, action): value}
    x_opt = {
        (st, a): float(x_val[e]) for e, (st, a) in enumerate(edges)
    }
    region_flag_val = {
        name: float(expr.value) for name, expr in region_flag_prob.items()
    }
    until_val = {
        name: float(expr.value) for name, expr in until_prob.items()
    }

    return float(obj.value), float(goal_prob.value), x_opt, region_flag_val, until_val, solve_time


def recover_policy_from_x_aug(mdp_aug: AugmentedMDP, x_opt, tol=1e-8):
    """
    Turn augmented occupation measures x into a (possibly stochastic)
    policy over augmented states.
    """
    policy = {}
    for st in mdp_aug.states_aug:
        if mdp_aug.is_absorbing_aug(st):
            continue
        flows = [(a, x_opt.get((st, a), 0.0)) for a in mdp_aug.actions_from_aug(st)]
        total = sum(v for _, v in flows)
        if total > tol:
            policy[st] = {a: v / total for a, v in flows}
        else:
            policy[st] = {a: 0.0 for a in mdp_aug.actions_from_aug(st)}
    return policy


def print_policy_grid_z0(mdp_aug: AugmentedMDP, policy_aug):
    """
    Show policy for the state where all bits (region + until) are 0,
    as a grid over the physical states.
    """
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    total_bits = len(mdp_aug.flags) + 2 * len(mdp_aug.until_specs)

    for r in range(mdp_aug.base.N):
        row = ""
        for c in range(mdp_aug.base.N):
            s = (r, c)
            st0 = (s,) + tuple(0 for _ in range(total_bits))
            if s in mdp_aug.base.goal:
                row += " G  "
            elif st0 not in policy_aug:
                row += " ·  "
            else:
                probs = policy_aug[st0]
                if not probs:
                    row += " ·  "
                    continue
                best_a = max(probs, key=lambda a: probs[a])
                if probs[best_a] < 1e-6:
                    row += " ·  "
                else:
                    row += f" {arrow[best_a]}  "
        print(row)


def print_policy_for_flags(mdp_aug: AugmentedMDP, policy_aug, flag_tuple):
    """
    Show policy for an arbitrary bit-vector (region + until),
    such as "already inside A", "after success of A U B", etc.
    """
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for r in range(mdp_aug.base.N):
        row = ""
        for c in range(mdp_aug.base.N):
            s = (r, c)
            st = (s,) + flag_tuple
            if s in mdp_aug.base.goal:
                row += " G  " 
            elif st not in policy_aug:
                row += " ·  "

            else:
                probs = policy_aug[st]
                if not probs:
                    row += " ·  "
                    continue
                best_a = max(probs, key=probs.get)
                if probs[best_a] < 1e-6:
                    row += " ·  "
                else:
                    row += f" {arrow[best_a]}  "
        print(row)
        
def simulate_policy_aug(
    mdp_aug: AugmentedMDP,
    policy_aug,
    max_steps: int = 100):
    """
    Simulate deterministically the augmented policy by always taking
    the action with the highest probability in policy_aug[st].

    Returns:
      base_traj: list of physical states s_t
      aug_traj:  list of augmented states (s_t, bits_t)
    """
    st = mdp_aug.start_aug
    s = st[0]
    base_traj = [s]
    aug_traj = [st]

    for _ in range(max_steps):
        if mdp_aug.is_absorbing_aug(st):
            break

        if st not in policy_aug:
            # no policy defined → stop
            break
        probs = policy_aug[st]
        if not probs:
            break

        # greedy action wrt probabilities
        a = max(probs, key=probs.get)

        st_next = mdp_aug.move_aug(st, a)
        s_next = st_next[0]

        base_traj.append(s_next)
        aug_traj.append(st_next)

        st, s = st_next, s_next

    return base_traj, aug_traj
    


