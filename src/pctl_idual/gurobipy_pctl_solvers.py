from dataclasses import dataclass
from typing import Dict, Tuple, Set, List, Optional
import cvxpy as cp
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

from pctl_idual.gridworld import GridWorld, State, Action  

AugState = Tuple  # (s, z1, z2, ...)
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
    kind: str   # "visit_region_max" or "visit_region_min"
    region_name: str
    bound: float


@dataclass
class UntilSpec:
    """
    Formula:  name :   A U B

    A_region = states where A holds
    B_region = states where B holds

    Semantics: along a run, the formula succeeds if you hit B
    while A has held at all previous steps; fails if you ever
    leave A before hitting B.
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
    kind: str   # "until_min" or "until_max"
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
        self.until_armed_idx: Dict[str, int] = {}
        self.until_success_idx: Dict[str, int] = {}
        self.until_fail_idx: Dict[str, int] = {}

        offset = len(self.flags)
        for i, uspec in enumerate(self.until_specs):
          base = offset + 3 * i
          self.until_armed_idx[uspec.name]   = base + 0
          self.until_success_idx[uspec.name] = base + 1
          self.until_fail_idx[uspec.name]    = base + 2

        total_bits = len(self.flags) + 3 * len(self.until_specs)
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
        # you can tweak this if you want until-formulas to keep “running”
        return self.base.is_goal(st[0])

    def actions_from_aug(self, st: AugState) -> List[Action]:
        if self.is_absorbing_aug(st):
            return []
        return self.base.actions_from(st[0])

    def move_aug(self, st: AugState, a: Action) -> AugState:
          """
          Deterministic next-state (kept for debugging).

          IMPORTANT: the LP must use transitions_aug() below,
          otherwise you IGNORE slip/stochasticity.
          """
          s = st[0]
          bits = list(st[1:])
          s2 = self.base.move(s, a)  # deterministic intended move
          bits2 = self._update_bits_from_next_state(bits, s2)
          # If we reached the base goal and this until never succeeded, mark failure
          if self.base.is_goal(s2):
              for uspec in self.until_specs:
                  i_succ = self.until_success_idx[uspec.name]
                  i_fail = self.until_fail_idx[uspec.name]
                  if bits2[i_succ] == 0:
                      bits2[i_fail] = 1
          return (s2, *bits2)
    def _update_bits_from_next_state(self, bits: List[int], s2: State) -> List[int]:
      # --- region flags ---
      for i, spec in enumerate(self.flags):
        if s2 in spec.region:
            bits[i] = 1

      # --- until bits (armed/success/fail) ---
      for uspec in self.until_specs:
        i_arm  = self.until_armed_idx[uspec.name]
        i_succ = self.until_success_idx[uspec.name]
        i_fail = self.until_fail_idx[uspec.name]

        # if already decided, don't change anything
        if bits[i_succ] == 1 or bits[i_fail] == 1:
            continue

        # If not armed yet: arm once you are in A (depending on your CVXPY rule)
        # Common choice: arm when you first enter A.
        if bits[i_arm] == 0:
            if s2 in uspec.A_region:
                bits[i_arm] = 1
            # if not armed and not in A, usually do nothing (not "failed" yet)
            if s2 in uspec.B_region:
                bits[i_fail] = 1
            continue    
        # armed: enforce staying in A until hitting B
        if s2 in uspec.B_region:
            bits[i_succ] = 1
            bits[i_fail] = 0
        elif s2 not in uspec.A_region:
            bits[i_fail] = 1
            bits[i_succ] = 0    

      return bits
    
    def transitions_aug(self, st: AugState, a: Action) -> Dict[AugState, float]:
          """
          Stochastic augmented transition kernel.

          Uses base.transitions(s,a) (this includes slip!) and updates bits per successor.
            Returns a dict {aug_state: prob}.
          """
          s = st[0]
          bits0 = list(st[1:])
          dist: Dict[AugState, float] = {}

          base_dist = self.base.transitions(s, a)  # {s2: p}
          for s2, p in base_dist.items():
              bits2 = self._update_bits_from_next_state(bits0.copy(), s2)

              if self.base.is_goal(s2):
                  for uspec in self.until_specs:
                      i_succ = self.until_success_idx[uspec.name]
                      i_fail = self.until_fail_idx[uspec.name]
                      if bits2[i_succ] == 0:
                          bits2[i_fail] = 1

              st2 = (s2, *bits2)
              dist[st2] = dist.get(st2, 0.0) + float(p)  

          # optional numeric safety
          total = sum(dist.values())
          if total > 0 and abs(total - 1.0) > 1e-12:
            for k in list(dist.keys()):
                dist[k] /= total

          return dist    

    def cost_aug(self, st: AugState, a: Action) -> float:
        return self.base.cost(st[0], a)

    @property
    def start_aug(self) -> AugState:
        total_bits = len(self.flags) + 3 * len(self.until_specs)
        if total_bits == 0:
            return (self.base.start,)
        return (self.base.start,) + tuple(0 for _ in range(total_bits))


# =========================
# Global LP with PCTL + Until
# =========================


def solve_lp_with_pctl_aug_gurobi(
    mdp_aug,
    p_goal_min: float,
    region_constraints: List,
    until_constraints: List,
    verbose: bool = False,
    env = None
):
    """
    Global LP over augmented MDP with:
      - P(reach GOAL) >= p_goal_min
      - region constraints on "ever visit region" flags
      - until constraints on P(A U B) success

    Returns:
      (J, p_goal, x_opt, region_probs, until_probs, solve_time)
    """

    # -------------------------------------------------------
    # 1) Enumerate non-absorbing augmented states and edges
    # -------------------------------------------------------
    non_abs_states = [st for st in mdp_aug.states_aug if not mdp_aug.is_absorbing_aug(st)]
    S = len(non_abs_states)
    if S == 0:
        return 0.0, 0.0, {}, {}, {}, 0.0

    state_index = {st: i for i, st in enumerate(non_abs_states)}

    edges: List[Tuple[Tuple, str]] = []
    for st in non_abs_states:
        for a in mdp_aug.actions_from_aug(st):
            edges.append((st, a))
    E = len(edges)
    if E == 0:
        return 0.0, 0.0, {}, {}, {}, 0.0

    # Build adjacency lists for flow constraints (memory-safe, no dense matrices)
    out_edges = [[] for _ in range(S)]
    in_edges  = [[] for _ in range(S)]
    b = np.zeros(S, dtype=float)

    start_idx = state_index.get(mdp_aug.start_aug, None)
    if start_idx is not None:
        b[start_idx] = 1.0

    for e, (st, a) in enumerate(edges):
        i_from = state_index[st]
        out_edges[i_from].append(e)

        for st2, p in mdp_aug.transitions_aug(st, a).items():
          if (not mdp_aug.is_absorbing_aug(st2)) and (st2 in state_index):
            i_to = state_index[st2]
            in_edges[i_to].append((e, float(p)))

    # -------------------------------------------------------
    # 2) Build coefficient vectors for probabilities
    # -------------------------------------------------------
    goal_coeff = np.zeros(E, dtype=float)

    region_coeffs = {spec.name: np.zeros(E, float) for spec in mdp_aug.flags}
    until_coeffs  = {uspec.name: np.zeros(E, float) for uspec in (mdp_aug.until_specs or [])}

    # Region/until coeffs must be counted for *all* edges whose successor has the bit set,
    # not only those that go into GOAL.
    for e, (st, a) in enumerate(edges):
        bits = st[1:]  # predecessor bits

        for st2, p in mdp_aug.transitions_aug(st, a).items():
          p = float(p)
          s2 = st2[0]
          bits2 = st2[1:]

          if s2 in mdp_aug.base.goal:
            goal_coeff[e] += p

          for spec in mdp_aug.flags:
            idx = mdp_aug.flag_indices[spec.name]
            if bits[idx] == 0 and bits2[idx] == 1:
                region_coeffs[spec.name][e] += p

          # Until success: success=1 and fail=0
          for uspec in (mdp_aug.until_specs or []):
            i_succ = mdp_aug.until_success_idx[uspec.name]
            i_fail = mdp_aug.until_fail_idx[uspec.name]

            if (bits[i_succ] == 0 and bits[i_fail] == 0
                    and bits2[i_succ] == 1 and bits2[i_fail] == 0):
                until_coeffs[uspec.name][e] += p
    name = "G2U_G3"
    if name in until_coeffs:
      print("\n[GUROBI DEBUG]")
      print("until_coeff nnz:", int(np.count_nonzero(until_coeffs[name])))
      print("until_coeff sum :", float(until_coeffs[name].sum()))
      print("until_coeff max :", float(until_coeffs[name].max()))
    else:
      print("\n[GUROBI DEBUG] missing key", name, "keys=", list(until_coeffs.keys()))
    

    # -------------------------------------------------------
    # 3) Objective cost vector
    # -------------------------------------------------------
    cost_vec = np.array([mdp_aug.cost_aug(st, a) for (st, a) in edges], dtype=float)

    # -------------------------------------------------------
    # 4) Build + solve in Gurobi
    # -------------------------------------------------------
    t0 = time.perf_counter()

    m = gp.Model("shortest_path_lp", env=env) if env is not None else gp.Model("pctl_global_lp")
    m.Params.OutputFlag = 1 if verbose else 0
    # Optional LP params that sometimes help:
    m.Params.Method = 2        # barrier
    m.Params.Crossover = 0
    m.Params.Presolve = 2

    x = m.addMVar(shape=E, lb=0.0, name="x")
    m.setObjective(cost_vec @ x, GRB.MINIMIZE)

    # Flow constraints: for each non-absorbing aug-state i:
    #   sum_out x_e - sum_in x_e = b[i]
    for i in range(S):
        expr = gp.LinExpr()
        # out
        if out_edges[i]:
            expr += x[out_edges[i]].sum()
        # in
        if in_edges[i]:
            expr -= gp.quicksum(p * x[e] for (e, p) in in_edges[i])
        m.addConstr(expr == float(b[i]), name=f"flow[{i}]")

    # Goal constraint
    m.addConstr(goal_coeff @ x >= float(p_goal_min), name="goal_min")

    # Region constraints
    for c in region_constraints:
        coeff = region_coeffs[c.region_name]
        if c.kind == "visit_region_max":
            m.addConstr(coeff @ x <= float(c.bound), name=f"reg_max[{c.region_name}]")
        elif c.kind == "visit_region_min":
            m.addConstr(coeff @ x >= float(c.bound), name=f"reg_min[{c.region_name}]")
        else:
            raise ValueError(f"Unknown region constraint kind {c.kind}")

    # Until constraints
    for c in until_constraints:
        coeff = until_coeffs[c.spec_name]
        if c.kind == "until_min":
            m.addConstr(coeff @ x >= float(c.bound), name=f"until_min[{c.spec_name}]")
        elif c.kind == "until_max":
            m.addConstr(coeff @ x <= float(c.bound), name=f"until_max[{c.spec_name}]")
        else:
            raise ValueError(f"Unknown until constraint kind {c.kind}")

    m.optimize()
    grb_runtime = float(m.Runtime)

    t1 = time.perf_counter()
    solve_time = t1 - t0

    if m.Status != GRB.OPTIMAL:
        print("Gurobi status:", m.Status)
        return None, None, None, None, None, solve_time

    x_val = x.X  # numpy array

    # Map back to {(aug_state, action): value}
    x_opt = {(st, a): float(x_val[e]) for e, (st, a) in enumerate(edges)}

    # Compute probabilities from coeffs
    p_goal = float(goal_coeff @ x_val)

    region_vals = {name: float(coeff @ x_val) for name, coeff in region_coeffs.items()}
    until_vals  = {name: float(coeff @ x_val) for name, coeff in until_coeffs.items()}

    return float(m.ObjVal), p_goal, x_opt, region_vals, until_vals, solve_time, grb_runtime


def recover_policy_from_x_aug_gurobi(mdp_aug: AugmentedMDP, x_opt, tol=1e-8):
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


def print_policy_grid_z0_gurobi(mdp_aug: AugmentedMDP, policy_aug):
    """
    Show policy for the state where all bits (region + until) are 0,
    as a grid over the physical states.
    """
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    total_bits = len(mdp_aug.flags) + 3 * len(mdp_aug.until_specs)

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


def print_policy_for_flags_gurobi(mdp_aug: AugmentedMDP, policy_aug, flag_tuple):
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

import random

def simulate_policy_aug_gurobi_stochastic(mdp_aug, policy_aug, max_steps=200):
    st = mdp_aug.start_aug
    base_traj = [st[0]]
    aug_traj  = [st]

    for _ in range(max_steps):
        if mdp_aug.is_absorbing_aug(st):
            break
        if st not in policy_aug or not policy_aug[st]:
            break

        # sample action from policy
        ap = policy_aug[st]
        actions = list(ap.keys())
        probs   = list(ap.values())
        sprob = sum(probs)
        probs = [p/sprob for p in probs] if sprob > 0 else [1/len(probs)]*len(probs)
        a = random.choices(actions, weights=probs, k=1)[0]

        # sample next augmented state from transition kernel (includes slip)
        trans = mdp_aug.transitions_aug(st, a)  # dict: st2 -> p
        st2s = list(trans.keys())
        ps   = list(trans.values())
        st_next = random.choices(st2s, weights=ps, k=1)[0]

        st = st_next
        aug_traj.append(st)
        base_traj.append(st[0])

    return base_traj, aug_traj
