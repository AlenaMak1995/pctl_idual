# -*- coding: utf-8 -*-
"""idual_solvers.py

i-dual style LP solvers on the augmented MDP used for PCTL constraints.

This version:
- Computes occupation measures x(s,z,a) on an envelope S_hat with fringe F.
- Tracks:
    * goal_prob          ~= P(true U GOAL)
    * region_flag_prob   ~= P(ever visit region_i before GOAL)
    * until_prob         ~= P(success of A U B before GOAL)
"""

from typing import Dict, Tuple, Set, List, Optional
import cvxpy as cp
import numpy as np

AugState = Tuple
Action = str


def compute_in_flow_aug(
    mdp_aug,
    x_opt: Dict[Tuple[AugState, Action], float],
) -> Dict[AugState, float]:
    """
    Compute in-flow for each augmented state under occupation measure x_opt.
    """
    in_flow: Dict[AugState, float] = {st: 0.0 for st in mdp_aug.states_aug}
    for (stp, a), val in x_opt.items():
        if val is None:
            continue
        st2 = mdp_aug.move_aug(stp, a)
        in_flow[st2] += float(val)
    return in_flow


def solve_lp3_aug(
    mdp_aug,
    S_hat: Set[AugState],
    F: Set[AugState],
    p_goal_min: float,
    extra_constraints,
    H: Optional[Dict[AugState, float]] = None,
    H_region_upper: Optional[Dict[str, Dict[AugState, float]]] = None,
    H_region_lower: Optional[Dict[str, Dict[AugState, float]]] = None,
    H_until_upper: Optional[Dict[str, Dict[AugState, float]]] = None,
    H_until_lower: Optional[Dict[str, Dict[AugState, float]]] = None,
):
    """
    Local LP (LP3-style) on the current envelope S_hat with fringe F.
    """
    enforce_pctl = (len(F) == 0)

    # ------------------------------------------------------------------
    # 1) Decision variables x(st,a) for interior states
    # ------------------------------------------------------------------
    x: Dict[Tuple[AugState, Action], cp.Variable] = {}
    for st in S_hat:
        if mdp_aug.is_absorbing_aug(st) or st in F:
            continue
        for a in mdp_aug.actions_from_aug(st):
            x[(st, a)] = cp.Variable(nonneg=True)

    def in_expr(st: AugState):
        return sum(
            var
            for (stp, a), var in x.items()
            if mdp_aug.move_aug(stp, a) == st
        )

    def out_expr(st: AugState):
        return sum(
            var
            for (stp, a), var in x.items()
            if stp == st
        )

    constraints: List[cp.Constraint] = []

    # ------------------------------------------------------------------
    # 2) Flow conservation on interior states
    #     out(s) - in(s) = b_s,  with b_start = 1, others 0
    # ------------------------------------------------------------------
    for st in S_hat:
        if mdp_aug.is_absorbing_aug(st) or st in F:
            continue
        b = 1.0 if st == mdp_aug.start_aug else 0.0
        constraints.append(out_expr(st) - in_expr(st) == b)

    # ------------------------------------------------------------------
    # 3) PCTL tracking on edges that ENTER the base-goal states
    # ------------------------------------------------------------------
    goal_prob = cp.Constant(0.0)
    region_flag_prob = {
        spec.name: cp.Constant(0.0) for spec in mdp_aug.flags
    }
    until_prob = {
        uspec.name: cp.Constant(0.0) for uspec in mdp_aug.until_specs
    }

    for (stp, a), var in x.items():
        st2 = mdp_aug.move_aug(stp, a)
        s2 = st2[0]

        # Only count probability when we step into the absorbing goal region
        if s2 in mdp_aug.base.goal:
            goal_prob += var

            bits2 = st2[1:]

            # Region "ever visited" flags
            for spec in mdp_aug.flags:
                idx = mdp_aug.flag_indices[spec.name]
                if bits2[idx] == 1:
                    region_flag_prob[spec.name] += var

            # Until success/fail bits
            for uspec in mdp_aug.until_specs:
                i_succ = mdp_aug.until_success_idx[uspec.name]
                i_fail = mdp_aug.until_fail_idx[uspec.name]
                if bits2[i_succ] == 1 and bits2[i_fail] == 0:
                    until_prob[uspec.name] += var

    region_prob_for_constraints = region_flag_prob
    if H_region_upper is not None and len(F) > 0:
        region_prob_upper = dict(region_flag_prob)

        for region_name, H_st in H_region_upper.items():
            expr = region_prob_upper.get(
                region_name, cp.Constant(0.0)
            )

            # Add heuristic fringe contribution
            fringe_terms = []
            for st in F:
                h = H_st.get(st, 0.0)
                if h != 0.0:
                    fringe_terms.append(h * in_expr(st))

            if fringe_terms:
                expr = expr + sum(fringe_terms)

            region_prob_upper[region_name] = expr

    region_prob_lower = region_flag_prob
    if H_region_lower is not None and len(F) > 0:
        region_prob_lower = dict(region_flag_prob)

        for region_name, H_st in H_region_lower.items():
            expr = region_prob_lower.get(
                region_name, cp.Constant(0.0)
            )
            fringe_terms = []
            for st in F:
                h = H_st.get(st, 0.0)
                if h != 0.0:
                    fringe_terms.append(h * in_expr(st))

            if fringe_terms:
                expr = expr + sum(fringe_terms)

            region_prob_lower[region_name] = expr

   # --- Heuristic-extended until probabilities ---
    until_prob_upper = until_prob
    if H_until_upper is not None and len(F) > 0:
        until_prob_upper = dict(until_prob)
        for spec_name, H_st in H_until_upper.items():
            expr = until_prob_upper.get(spec_name, cp.Constant(0.0))
            fringe_terms = []
            for st in F:
                h = H_st.get(st, 0.0)
                if h != 0.0:
                    fringe_terms.append(h * in_expr(st))
            if fringe_terms:
                expr = expr + sum(fringe_terms)
            until_prob_upper[spec_name] = expr

    until_prob_lower = until_prob
    if H_until_lower is not None and len(F) > 0:
        until_prob_lower = dict(until_prob)
        for spec_name, H_st in H_until_lower.items():
            expr = until_prob_lower.get(spec_name, cp.Constant(0.0))
            fringe_terms = []
            for st in F:
                h = H_st.get(st, 0.0)
                if h != 0.0:
                    fringe_terms.append(h * in_expr(st))
            if fringe_terms:
                expr = expr + sum(fringe_terms)
            until_prob_lower[spec_name] = expr                
    # ------------------------------------------------------------------
    # 4) Enforce probability / PCTL constraints
    # ------------------------------------------------------------------
    enforce_final = (len(F) == 0)

    if enforce_final:
        # Enforce P(true U GOAL) >= p_goal_min
        if p_goal_min > 0.0:
            constraints.append(goal_prob >= p_goal_min)

    # Region / until constraints
    for c in extra_constraints:
        if c.kind == "visit_region_max":
            # For <= constraints, use upper-bound heuristics
            # even when F is non-empty.
            if H_region_upper is not None and len(F) > 0:
                expr = region_prob_upper[c.region_name]
            elif enforce_final:
                expr = region_flag_prob[c.region_name]
            else:
                continue
            constraints.append(expr <= c.bound)

        elif c.kind == "visit_region_min":
            # For >= constraints, enforce them when the envelope is full
            if H_region_lower is not None and len(F) > 0:
                expr = region_prob_lower[c.region_name]
            elif enforce_final:
                expr = region_flag_prob[c.region_name]
            else:
                continue
            constraints.append(expr >= c.bound)

        elif c.kind == "until_min":
            # P(until_spec) >= bound
            if H_until_lower is not None and len(F) > 0:
                expr = until_prob_lower[c.spec_name]
            elif enforce_final:
                expr = until_prob[c.spec_name]
            else:
                continue
            constraints.append(expr >= c.bound)

        elif c.kind == "until_max":
            # P(until_spec) <= bound
            if H_until_upper is not None and len(F) > 0:
                expr = until_prob_upper[c.spec_name]
            elif enforce_final:
                expr = until_prob[c.spec_name]
            else:
                continue
            constraints.append(expr <= c.bound)

        else:
            raise ValueError(f"Unknown constraint kind {c.kind}")

    # ------------------------------------------------------------------
    # Debug: LP3 local problem size
    # ------------------------------------------------------------------
    num_x_vars = len(x)
    num_states_local = len({st for (st, a) in x.keys()})
    num_constraints = len(constraints)

    # print(f"[LP3] local interior states: {num_states_local}")
    # print(f"[LP3] x vars (st,a):         {num_x_vars}")
    # print(f"[LP3] constraints:           {num_constraints}")
    # ------------------------------------------------------------------
    # 5) Objective: base cost + optional heuristic on fringe
    # ------------------------------------------------------------------
    costs_list: List[float] = []
    vars_list: List[cp.Expression] = []
    for (st, a), var in x.items():
        costs_list.append(mdp_aug.cost_aug(st, a))
        vars_list.append(var)

    if vars_list:
        costs_vec = np.array(costs_list)
        vars_vec = cp.hstack(vars_list)
        obj_expr = costs_vec @ vars_vec
    else:
        # Degenerate case (no interior variables): objective is zero
        obj_expr = cp.Constant(0.0)
       
    if H is not None and F:
        fringe_cost = 0
        for st in F:
            fringe_cost += H.get(st, 0.0) * in_expr(st)
        obj_expr = obj_expr + fringe_cost

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)
    prob.solve(solver=cp.MOSEK)

    if prob.status not in ("optimal", "optimal_inaccurate"):
      print("LP3 status:", prob.status)
      return None, 0.0, {}, {}, 0.0, None
    obj_value = prob.value    

    x_opt = {
        (st, a): (var.value if var.value is not None else 0.0)
        for (st, a), var in x.items()
    }
    region_flag_val = {
        name: float(expr.value) for name, expr in region_flag_prob.items()
    }
    until_val = {
        name: float(expr.value) for name, expr in until_prob.items()
    }

    return x_opt, float(goal_prob.value), region_flag_val, until_val, prob.solver_stats.solve_time, obj_value

def fmt_aug_state(st):
    """
    st is an augmented state like ( (r,c), z1, z2, ... ).
    Return a string for flags "((r,c), z=(z1,z2,...))".
    """
    s = st[0]
    flags = st[1:]
    return f"{s} z={flags}"


def print_state_set(name, states, max_print=30):
    states = list(states)
    print(f"{name} (|{name}| = {len(states)})")

    if not states:
        print("   ∅")
        return

    # sort by row, col, then flags
    states.sort(key=lambda st: (st[0][0], st[0][1]) + st[1:])

    for i, st in enumerate(states):
        if i == max_print:
            print(f"   ... ({len(states) - max_print} more)")
            break
        print("   ", fmt_aug_state(st))

def i_dual_aug_trevizan(
    mdp_aug,
    p_goal_min: float,
    extra_constraints,
    H: Optional[Dict[AugState, float]] = None,
    H_region_upper: Optional[Dict[str, Dict[AugState, float]]] = None,
    H_region_lower: Optional[Dict[str, Dict[AugState, float]]] = None,
    H_until_upper: Optional[Dict[str, Dict[AugState, float]]] = None,
    H_until_lower: Optional[Dict[str, Dict[AugState, float]]] = None,
    tol: float = 1e-8,
):
    """
    Trevizan i-dual on the augmented MDP.

    - Grows an envelope S_hat with fringe F and relevant fringe FR.
    - Loop condition: while FR is nonempty.
    - When FR becomes empty, stop: the remaining fringe is deemed
      irrelevant under the current LP + heuristic.
    - The final solution is the last LP3 solution.

    With H_region_upper != None, we can safely enforce visit_region_max
    constraints (e.g. P(eventually G4) <= bound) in partial envelopes,
    by adding heuristic fringe contributions to region probabilities.
    """
    import time

    start_aug = mdp_aug.start_aug

    S_hat: Set[AugState] = {start_aug}
    F: Set[AugState] = {start_aug}
    FR: Set[AugState] = {start_aug}

    G_abs: Set[AugState] = {
        st for st in mdp_aug.states_aug if mdp_aug.is_absorbing_aug(st)
    }

    it = 0
    total_time = 0.0
    last_lp_time = 0.0

    x_last = None
    goal_prob_last = 0.0
    region_flag_last = {}
    until_last = {}
    obj_last = None

    while FR:
        # print(f"\n=== Trevizan Iteration {it} ===")
        # print("S_hat size:", len(S_hat))
        # print("F size    :", len(F))
        # print("FR size   :", len(FR))

        # 1) Expand only FR
        N: Set[AugState] = set()
        for st in FR:
            for a in mdp_aug.actions_from_aug(st):
                st2 = mdp_aug.move_aug(st, a)
                if st2 not in S_hat:
                    N.add(st2)

        # print(f"  New states N: {len(N)}")
        # if len(N) <= 30:
        #     print_state_set("N (newly discovered)", N, max_print=30)

        S_hat |= N
        F = (F - FR) | (N - G_abs)
        hatG = F | (G_abs & S_hat)

        # print("  Updated S_hat size:", len(S_hat))
        # print("  Updated F size    :", len(F))
        # print("  hatG size         :", len(hatG))

        # 2) Solve local LP3 with heuristic terminal costs 
        t0 = time.perf_counter()
        x_opt, goal_prob, region_flag_val, until_val, solve_t, obj_cur = solve_lp3_aug(
            mdp_aug,
            S_hat,
            F,
            p_goal_min=p_goal_min,
            extra_constraints=extra_constraints,
            H=H,
            H_region_upper=H_region_upper,
            H_region_lower=H_region_lower,
            H_until_upper=H_until_upper,
            H_until_lower=H_until_lower,
        )
        t1 = time.perf_counter()
        last_lp_time = solve_t
        total_time += (t1 - t0)

        if x_opt is None:
            print("[i-dual Trevizan] LP3 infeasible; stopping.")
            break

        x_last = x_opt
        goal_prob_last = goal_prob
        region_flag_last = region_flag_val
        until_last = until_val
        obj_last = obj_cur

        # 3) Update relevant fringe: states in F with positive in-flow
        in_flow = compute_in_flow_aug(mdp_aug, x_opt)
        FR = {st for st in F if in_flow.get(st, 0.0) > tol}
        # print("  New FR size after LP:", len(FR))

        it += 1

    if x_last is None:
        print("[i-dual Trevizan] No feasible LP solution found; returning no policy.")
        return None, 0.0, {}, {}, total_time, last_lp_time, it, len(S_hat), obj_last

    return (
        x_last,
        goal_prob_last,
        region_flag_last,
        until_last,
        total_time,
        last_lp_time,
        it,
        len(S_hat),
        obj_last,
    )

def compute_max_prob_visit_region_before_goal(
    mdp_aug,
    region_name: str,
    tol: float = 1e-6,
    max_iter: int = 1000,
):
    """
    Heuristic on the augmented MDP:

        H[st_aug] = max_π P^π( "ever visited region_name before GOAL" | start = st_aug )

    where "ever visited region_name" is encoded via the region flag bit
    in the augmented state, and GOAL is mdp_aug.base.goal.

    This is an UPPER bound: for any policy π,
        P^π(eventually visit region_name before GOAL | st) <= H[st].

    That makes it safe to use for constraints of the form
        P(eventually region_name) <= bound
    using the heuristic at the fringe.
    """
    flag_idx = mdp_aug.flag_indices[region_name]

    V = {}

    # Initialize boundary conditions
    for st in mdp_aug.states_aug:
        s = st[0]
        bits = st[1:]

        if mdp_aug.is_absorbing_aug(st):
            # If we are in an absorbing GOAL state and the region flag is 1,
            # then the event "visited region before GOAL" has already happened.
            if (s in mdp_aug.base.goal) and (bits[flag_idx] == 1):
                V[st] = 1.0
            else:
                # Either a non-goal absorbing or a goal without a region visited
                V[st] = 0.0
        else:
            # Non-absorbing: initialize to 0, then iteratively improve
            V[st] = 0.0

    # Value iteration: maximal probability of success
    for _ in range(max_iter):
        delta = 0.0
        for st in mdp_aug.states_aug:
            if mdp_aug.is_absorbing_aug(st):
                continue

            q_vals = []
            for a in mdp_aug.actions_from_aug(st):
                st2 = mdp_aug.move_aug(st, a)
                q_vals.append(V[st2])

            if not q_vals:
                continue  

            new_v = max(q_vals)
            delta = max(delta, abs(new_v - V[st]))
            V[st] = new_v

        if delta < tol:
            break

    return V  

def compute_min_prob_visit_region_before_goal(
    mdp_aug,
    region_name: str,
    tol: float = 1e-6,
    max_iter: int = 1000,
):
    """
    Heuristic on the augmented MDP:

        H[st_aug] = min_π P^π( "ever visited region_name before GOAL" | start = st_aug )

    where "ever visited region_name" is encoded via the region flag bit
    in the augmented state, and GOAL is mdp_aug.base.goal.

    Interpretation:
    --------------
    - H[st] is the best (smallest) probability of ever visiting region_name
      before GOAL, assuming we can pick the actions optimally from st onward.
    - This lets us say: if H[st] <= bound, then there exists at least one
      “good” continuation policy from st that respects the bound.

    We will use this in LP3 for constraints of the form:

        P(ever visit region_name) <= bound

    by combining flows into fringe states with these H-values.
    """

    flag_idx = mdp_aug.flag_indices[region_name]

    V = {}

    # Initialize boundary conditions
    for st in mdp_aug.states_aug:
        s = st[0]
        bits = st[1:]

        if mdp_aug.is_absorbing_aug(st):
            # If we are in an absorbing GOAL state and the region flag is 1,
            # then the event "visited region before GOAL" has already happened.
            if (s in mdp_aug.base.goal) and (bits[flag_idx] == 1):
                V[st] = 1.0
            else:
                # Either a non-goal absorbing or a goal without a region visited
                V[st] = 0.0
        else:
            # Non-absorbing: start from 0 and iteratively improve
            V[st] = 0.0

    # Value iteration: minimal probability of success 
    for _ in range(max_iter):
        delta = 0.0
        for st in mdp_aug.states_aug:
            if mdp_aug.is_absorbing_aug(st):
                continue

            q_vals = []
            for a in mdp_aug.actions_from_aug(st):
                st2 = mdp_aug.move_aug(st, a)
                q_vals.append(V[st2])

            if not q_vals:
                continue 

            new_v = min(q_vals)
            delta = max(delta, abs(new_v - V[st]))
            V[st] = new_v

        if delta < tol:
            break

    return V 

def compute_max_prob_until_before_goal(
    mdp_aug,
    spec_name: str,
    tol: float = 1e-6,
    max_iter: int = 1000,
):
    """
    H[st_aug] = max_π P^π( "success of until spec_name before GOAL" | st_aug )

    'Success of until' is exactly what until_prob[spec_name] counts in the LP:
    arriving in GOAL with (succ_bit = 1, fail_bit = 0) for this spec.
    """

    i_succ = mdp_aug.until_success_idx[spec_name]
    i_fail = mdp_aug.until_fail_idx[spec_name]

    V = {}

    # Boundary conditions
    for st in mdp_aug.states_aug:
        s = st[0]
        bits = st[1:]

        if mdp_aug.is_absorbing_aug(st):
            # If we are in GOAL and "until succeeded" (succ=1, fail=0),
            # count that as success.
            if (s in mdp_aug.base.goal) and (bits[i_succ] == 1 and bits[i_fail] == 0):
                V[st] = 1.0
            else:
                V[st] = 0.0
        else:
            V[st] = 0.0

    # Value iteration: maximal probability
    for _ in range(max_iter):
        delta = 0.0
        for st in mdp_aug.states_aug:
            if mdp_aug.is_absorbing_aug(st):
                continue

            q_vals = []
            for a in mdp_aug.actions_from_aug(st):
                st2 = mdp_aug.move_aug(st, a)
                q_vals.append(V[st2])

            if not q_vals:
                continue

            new_v = max(q_vals)
            delta = max(delta, abs(new_v - V[st]))
            V[st] = new_v

        if delta < tol:
            break

    return V  


def compute_min_prob_until_before_goal(
    mdp_aug,
    spec_name: str,
    tol: float = 1e-6,
    max_iter: int = 1000,
):
    """
    H[st_aug] = min_π P^π( "success of until spec_name before GOAL" | st_aug )

    This is the BEST (smallest) success probability we can force by choosing
    actions adversarially.
    """

    i_succ = mdp_aug.until_success_idx[spec_name]
    i_fail = mdp_aug.until_fail_idx[spec_name]

    V = {}

    for st in mdp_aug.states_aug:
        s = st[0]
        bits = st[1:]

        if mdp_aug.is_absorbing_aug(st):
            if (s in mdp_aug.base.goal) and (bits[i_succ] == 1 and bits[i_fail] == 0):
                V[st] = 1.0
            else:
                V[st] = 0.0
        else:
            V[st] = 0.0

    # Value iteration: minimal probability
    for _ in range(max_iter):
        delta = 0.0
        for st in mdp_aug.states_aug:
            if mdp_aug.is_absorbing_aug(st):
                continue

            q_vals = []
            for a in mdp_aug.actions_from_aug(st):
                st2 = mdp_aug.move_aug(st, a)
                q_vals.append(V[st2])

            if not q_vals:
                continue

            new_v = min(q_vals)
            delta = max(delta, abs(new_v - V[st]))
            V[st] = new_v

        if delta < tol:
            break

    return V
    
