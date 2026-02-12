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
class AugmentedMDPBaseline:
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
        # Absorb ONLY at base goal.
        # Do NOT absorb on flags, otherwise you change goal reachability semantics.
      return self.base.is_goal(st[0])

    def actions_from_aug(self, st: AugState) -> List[Action]:
      if self.is_absorbing_aug(st):
        return []
      return self.base.actions_from(st[0])

    def _update_bits_from_next_state(self, bits: List[int], s2: State) -> List[int]:


          # --- simple "visited region" flags (monotone: 0 -> 1 and stays 1) ---
          for i, spec in enumerate(self.flags):
            if s2 in spec.region:
                bits[i] = 1

          # --- A U B formula bits (success/fail, also monotone) ---
          for uspec in self.until_specs:
            i_arm  = self.until_armed_idx[uspec.name]
            i_succ = self.until_success_idx[uspec.name]
            i_fail = self.until_fail_idx[uspec.name]

            # already decided for this formula?
            if bits[i_succ] == 1 or bits[i_fail] == 1:
                continue

            if bits[i_arm] == 0:
            # not armed yet: arm when entering A
                    if s2 in uspec.A_region:
                        bits[i_arm] = 1

                    # If you want "must go through A first", you can mark fail when reaching B while not armed.
                    # Otherwise, just do nothing here.
                    if s2 in uspec.B_region:
                        bits[i_fail] = 1   # keep if you want "must pass A first"
                    continue
            # Armed: now enforce staying in A until hitting B
            if s2 in uspec.B_region:
              bits[i_succ] = 1
              bits[i_fail] = 0
            elif s2 not in uspec.A_region:
              bits[i_fail] = 1
              bits[i_succ] = 0        

          return bits

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

def solve_lp_with_pctl_aug_baseline(
    mdp_aug: AugmentedMDPBaseline,
    p_goal_min: float,
    region_constraints: List[PCTLRegionConstraint],
    until_constraints: List[UntilConstraint],
):
    """
    Global LP over augmented MDP with:

      - P(reach goal) >= p_goal_min
      - region constraints on P(ever visit region) for monotone visit-flags
      - until constraints on P(A U B) for each UntilSpec

    IMPORTANT:
      This version is STOCHASTIC-correct: it uses mdp_aug.transitions_aug(st,a),
      so slip_prob affects feasibility and probabilities.

    Trick for "ever visit" with monotone bits:
      P(ever visit region_i) == total probability mass on edges that flip bit_i 0->1.
      (Because it can happen at most once.)
    """

    # -------------------------------------------------------
    # 1) Enumerate non-absorbing augmented states and edges
    # -------------------------------------------------------
    non_abs_states = [st for st in mdp_aug.states_aug
                      if not mdp_aug.is_absorbing_aug(st)]
    state_index = {st: i for i, st in enumerate(non_abs_states)}

    edges: List[Tuple[AugState, Action]] = []
    for st in non_abs_states:
        for a in mdp_aug.actions_from_aug(st):
            edges.append((st, a))

    E = len(edges)
    if E == 0:
        return 0.0, 0.0, {}, {}, {}, 0.0

    x_vec = cp.Variable(E, nonneg=True)

    # -------------------------------------------------------
    # 2) Flow constraints (stochastic)
    #   out(st) - sum_{prev, a} P(st | prev,a) x(prev,a) = 1{st=start}
    # -------------------------------------------------------
    S = len(non_abs_states)
    A_flow = np.zeros((S, E))
    b = np.zeros(S)

    for e, (st, a) in enumerate(edges):
        i_from = state_index[st]
        A_flow[i_from, e] += 1.0  # outflow from st

        for st2, p in mdp_aug.transitions_aug(st, a).items():
            if (not mdp_aug.is_absorbing_aug(st2)) and (st2 in state_index):
                i_to = state_index[st2]
                A_flow[i_to, e] -= float(p)

    start_idx = state_index.get(mdp_aug.start_aug, None)
    if start_idx is not None:
        b[start_idx] = 1.0

    constraints = [A_flow @ x_vec == b]

    # -------------------------------------------------------
    # 3) Probability coefficient vectors
    # -------------------------------------------------------
    goal_coeff = np.zeros(E)

    region_coeffs: Dict[str, np.ndarray] = {
        spec.name: np.zeros(E) for spec in mdp_aug.flags
    }
    until_coeffs: Dict[str, np.ndarray] = {
        uspec.name: np.zeros(E) for uspec in mdp_aug.until_specs
    }

    for e, (st, a) in enumerate(edges):
        bits = st[1:]  # predecessor bits

        for st2, p in mdp_aug.transitions_aug(st, a).items():
            s2 = st2[0]
            bits2 = st2[1:]
            p = float(p)

            # reach-goal probability: probability mass on edges that enter goal
            if s2 in mdp_aug.base.goal:
                goal_coeff[e] += p

            # ever-visit region_i: count the *first* time bit flips 0->1
            for spec in mdp_aug.flags:
                idx = mdp_aug.flag_indices[spec.name]
                if bits[idx] == 0 and bits2[idx] == 1:
                    region_coeffs[spec.name][e] += p

            # until success probability: count the transition that triggers success first time
            for uspec in mdp_aug.until_specs:
                i_succ = mdp_aug.until_success_idx[uspec.name]
                i_fail = mdp_aug.until_fail_idx[uspec.name]
                if (bits[i_succ] == 0 and bits[i_fail] == 0
                        and bits2[i_succ] == 1 and bits2[i_fail] == 0):
                    until_coeffs[uspec.name][e] += p
            name = "G2U_G3"
    print("\n[CVXPY DEBUG]")
    print("until_coeff nnz:", int(np.count_nonzero(until_coeffs[name])))
    print("until_coeff sum :", float(until_coeffs[name].sum()))
    print("until_coeff max :", float(until_coeffs[name].max()))

    goal_prob = goal_coeff @ x_vec
    region_prob = {name: coeff @ x_vec for name, coeff in region_coeffs.items()}
    until_prob  = {name: coeff @ x_vec for name, coeff in until_coeffs.items()}

    # -------------------------------------------------------
    # 4) Add constraints
    # -------------------------------------------------------
    constraints.append(goal_prob >= float(p_goal_min))

    for rc in region_constraints:
        expr = region_prob[rc.region_name]
        if rc.kind == "visit_region_max":
            constraints.append(expr <= float(rc.bound))
        elif rc.kind == "visit_region_min":
            constraints.append(expr >= float(rc.bound))
        else:
            raise ValueError(f"Unknown region constraint kind='{rc.kind}'")

    for uc in until_constraints:
        expr = until_prob[uc.spec_name]
        if uc.kind == "until_min":
            constraints.append(expr >= float(uc.bound))
        elif uc.kind == "until_max":
            constraints.append(expr <= float(uc.bound))
        else:
            raise ValueError(f"Unknown until constraint kind='{uc.kind}'")

    # -------------------------------------------------------
    # 5) Objective: minimize expected cumulative cost
    # -------------------------------------------------------
    c_vec = np.zeros(E)
    for e, (st, a) in enumerate(edges):
        c_vec[e] = mdp_aug.cost_aug(st, a)

    objective = cp.Minimize(c_vec @ x_vec )
    prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    # print("Until constraints:", [(uc.kind, uc.spec_name, uc.bound) for uc in until_constraints])
    # print("Has key G2U_G3 in until_prob?", "G2U_G3" in until_prob)
    prob.solve(solver=cp.MOSEK, verbose=False)
    t1 = time.perf_counter()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # infeasible/other → keep return shape consistent
        return float("inf"), 0.0, {}, {k: 0.0 for k in region_prob}, {k: 0.0 for k in until_prob}, (t1 - t0)

    x_opt = x_vec.value
    x_opt_dict = {edges[i]: float(x_opt[i]) for i in range(E) if x_opt[i] > 1e-12}

    J = float((c_vec @ x_vec).value)
    p_goal_out = float(goal_prob.value)
    region_probs_out = {k: float(v.value) for k, v in region_prob.items()}
    until_probs_out  = {k: float(v.value) for k, v in until_prob.items()}

    return J, p_goal_out, x_opt_dict, region_probs_out, until_probs_out, (t1 - t0)



def recover_policy_from_x_aug(mdp_aug: AugmentedMDPBaseline, x_opt, tol=1e-8):
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


def print_policy_grid_z0(mdp_aug: AugmentedMDPBaseline, policy_aug):
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


def simulate_policy_aug(
    mdp_aug: AugmentedMDPBaseline,
    policy_aug,
    max_steps: int = 100,
    seed: int = 0,
    greedy: bool = False,
):
    rng = np.random.default_rng(seed)

    st = mdp_aug.start_aug
    base_traj = [st[0]]
    aug_traj  = [st]

    for t in range(max_steps):
        if mdp_aug.is_absorbing_aug(st):
            break

        probs = policy_aug.get(st, None)
        if probs is None:
            break

        actions = list(probs.keys())
        p = np.array([probs[a] for a in actions], dtype=float)
        if p.sum() <= 1e-12:
            p = np.ones(len(actions)) / len(actions)
        else:
            p /= p.sum()

        if greedy:
            a = actions[int(np.argmax(p))]
        else:
            i = rng.choice(len(actions), p=p)
            a = actions[i]

        dist = mdp_aug.transitions_aug(st, a)
        items = list(dist.items())
        next_states = [s2 for s2, _ in items]
        next_probs  = np.array([p2 for _, p2 in items], dtype=float)
        next_probs  = next_probs / next_probs.sum()

        j = rng.choice(len(next_states), p=next_probs)
        st = next_states[j]

        base_traj.append(st[0])
        aug_traj.append(st)

    return base_traj, aug_traj