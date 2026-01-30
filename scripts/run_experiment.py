import argparse
import yaml

from pctl_idual.gridworld import (
    make_grid_world,
    make_4x4_world,
    make_20x20_world,
    make_NxN_world,
    make_4x4_pctl_world,
)

from pctl_idual.dp_solvers import (
    value_iteration_shortest_path,
    greedy_policy_from_V,
    simulate_policy,
)

from pctl_idual.lp_solvers import (
    solve_shortest_path_lp,
    recover_policy_from_x,
    print_policy_grid,
    collapse_augmented_policy_to_base
)

from pctl_idual.pctl_solvers import (
    RegionFlagSpec,
    PCTLRegionConstraint,
    UntilSpec,
    UntilConstraint,
    AugmentedMDP,
    solve_lp_with_pctl_aug,
    recover_policy_from_x_aug,
    print_policy_grid_z0,
    print_policy_for_flags,
    simulate_policy_aug
)

from pctl_idual.idual_solvers import (
    compute_in_flow_aug,
    solve_lp3_aug,
    i_dual_aug_trevizan,
    compute_max_prob_visit_region_before_goal,
    compute_min_prob_visit_region_before_goal,
    compute_max_prob_until_before_goal,
    compute_min_prob_until_before_goal,
)


from pctl_idual.gurobipy_lp_solver import (
    solve_shortest_path_lp_gurobi,
    recover_policy_from_x_gurobi,
    print_policy_grid_gurobi

)

from pctl_idual.gurobipy_pctl_solvers import (
    solve_lp_with_pctl_aug_gurobi,
    recover_policy_from_x_aug_gurobi,
    print_policy_grid_z0_gurobi,
    print_policy_for_flags_gurobi,
    simulate_policy_aug_gurobi
)
def build_mdp(mdp_cfg):
    world = mdp_cfg["world"]

    if world == "4x4":
        mdp = make_4x4_world()

    elif world == "20x20":
        mdp = make_20x20_world()

    elif world == "NxN":
        return make_NxN_world(
            N=mdp_cfg["N"],
            start=tuple(mdp_cfg["start"]),
            goal={tuple(g) for g in mdp_cfg["goal"]},
            slip_prob=mdp_cfg.get("slip_prob", 0.0),
        )

    elif world == "custom":
        rect_costs = [tuple(rc) for rc in mdp_cfg.get("rect_costs", [])]
        return make_grid_world(
            N=mdp_cfg["N"],
            start=tuple(mdp_cfg["start"]),
            goal={tuple(g) for g in mdp_cfg["goal"]},
            default_cost=float(mdp_cfg.get("default_cost", 1.0)),
            rect_costs=rect_costs,
            slip_prob=float(mdp_cfg.get("slip_prob", 0.0)),
        )
    elif world == "4x4_pctl":
        mdp = make_4x4_pctl_world()


    else:
        raise ValueError(f"Unknown world type: {world}")

    return mdp

def parse_region(spec, N):
    """
    Parse a region spec from YAML into a set of (row, col) cells.

    Supported formats:
      - {"rect": [r_min, c_min, r_max, c_max]}  (inclusive)
      - {"cells": [[r,c], [r,c], ...]}
      - {"union": [ <region_spec>, <region_spec>, ... ]}  (recursive)
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Bad region spec (expected dict): {spec}")

    # --- rectangle ---
    if "rect" in spec:
        r1, c1, r2, c2 = spec["rect"]
        if not (0 <= r1 <= r2 < N and 0 <= c1 <= c2 < N):
            raise ValueError(f"rect out of bounds for N={N}: {spec['rect']}")
        return {(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)}

    # --- explicit cells ---
    if "cells" in spec:
        cells = set()
        for rc in spec["cells"]:
            r, c = rc
            if not (0 <= r < N and 0 <= c < N):
                raise ValueError(f"cell out of bounds for N={N}: {rc}")
            cells.add((r, c))
        return cells

    # --- union of specs (recursive) ---
    if "union" in spec:
        parts = spec["union"]
        if not isinstance(parts, list) or len(parts) == 0:
            raise ValueError(f"union must be a non-empty list: {spec}")
        region = set()
        for part in parts:
            region |= parse_region(part, N)  # recursion
        return region

    raise ValueError(f"Bad region spec: {spec}")

def build_augmented_from_pctl_cfg(mdp, pctl_cfg):
    p_goal_min = float(pctl_cfg.get("p_goal_min", 1.0))

    # Build regions dict from flags
    regions = {}
    for f in pctl_cfg.get("flags", []):
        name = f["name"]
        regions[name] = parse_region(f["region"], mdp.N)

    flags = [RegionFlagSpec(name, regions[name]) for name in regions]

    # Until specs
    until_specs = []
    for us in pctl_cfg.get("until_specs", []):
        until_specs.append(
            UntilSpec(
                us["name"],
                A_region=regions[us["A"]],
                B_region=regions[us["B"]],
            )
        )

    mdp_aug = AugmentedMDP(mdp, flags=flags, until_specs=until_specs)

    # Region constraints
    region_constraints = []
    for rc in pctl_cfg.get("region_constraints", []):
        region_constraints.append(
            PCTLRegionConstraint(rc["type"], rc["region"], float(rc["p"]))
        )

    # Until constraints
    until_constraints = []
    for uc in pctl_cfg.get("until_constraints", []):
        until_constraints.append(
            UntilConstraint(uc["type"], uc["until"], float(uc["p"]))
        )

    extra_constraints = region_constraints + until_constraints
    return mdp_aug, p_goal_min, extra_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # --- Load config ---
    cfg = yaml.safe_load(open(args.config, "r"))
    mdp = build_mdp(cfg["mdp"])

    run_cfg = cfg.get("run", {})
    solver = run_cfg.get("solver", "both").lower()
    solver = run_cfg.get("solver", "both").lower()
    if solver not in ("dp", "lp", "both", "pctl_lp", "idual_trevizan"):
        raise ValueError(
            "Unknown run.solver. Use dp | lp | both | pctl_lp | idual_trevizan."
        )


    lp_cfg = cfg.get("lp", {})
    lp_solver = lp_cfg.get("solver", "MOSEK")

    if solver in ("dp", "both"):
        V = value_iteration_shortest_path(mdp)
        print("DP (unconstrained) optimal cost from START:", V[mdp.start])

        pi = greedy_policy_from_V(mdp, V)
        traj = simulate_policy(mdp, pi)

        print("Trajectory under DP (unconstrained):")
        print(traj)

    if solver in ("lp", "both"):
        J_lp, x_opt, t_lp = solve_shortest_path_lp(mdp, solver=lp_solver)
        print("LP cost:", J_lp, "   solve time:", t_lp)

        pi_lp = recover_policy_from_x(mdp, x_opt)
        print("\nPolicy from LP:")
        print_policy_grid(mdp, pi_lp)

    if solver == "pctl_lp":
        pctl_cfg = cfg.get("pctl", {})
        backend = pctl_cfg.get("backend", "cvxpy").lower()
        if backend not in ("cvxpy", "gurobi"):
            raise ValueError("pctl.backend must be cvxpy or gurobi")

        p_goal_min = float(pctl_cfg.get("p_goal_min", 1.0))

        # Build regions dict from flags
        regions = {}
        for f in pctl_cfg.get("flags", []):
            name = f["name"]
            regions[name] = parse_region(f["region"], mdp.N)

        flags = [RegionFlagSpec(name, regions[name]) for name in regions]

        # Until specs
        until_specs = []
        for us in pctl_cfg.get("until_specs", []):
            until_specs.append(
                UntilSpec(
                    us["name"],
                    A_region=regions[us["A"]],
                    B_region=regions[us["B"]],
                )
            )

        mdp_aug = AugmentedMDP(mdp, flags=flags, until_specs=until_specs)

        # Region constraints
        region_constraints = []
        for rc in pctl_cfg.get("region_constraints", []):
            region_constraints.append(
                PCTLRegionConstraint(rc["type"], rc["region"], float(rc["p"]))
            )

        # Until constraints
        until_constraints = []
        for uc in pctl_cfg.get("until_constraints", []):
            until_constraints.append(
                UntilConstraint(uc["type"], uc["until"], float(uc["p"]))
            )

        # ---- solve ----
        if backend == "cvxpy":
            J_pctl, p_goal, x_opt_aug, region_probs, until_probs, t_pctl = solve_lp_with_pctl_aug(
                mdp_aug,
                p_goal_min=p_goal_min,
                region_constraints=region_constraints,
                until_constraints=until_constraints,
            )
            policy_recover = recover_policy_from_x_aug
            policy_sim = simulate_policy_aug

        else:  # backend == "gurobi"
            J_pctl, p_goal, x_opt_aug, region_probs, until_probs, t_pctl = solve_lp_with_pctl_aug_gurobi(
                mdp_aug,
                p_goal_min=p_goal_min,
                region_constraints=region_constraints,
                until_constraints=until_constraints,
                verbose=False,
                env=None,  # or pass GRB_ENV
            )
            policy_recover = recover_policy_from_x_aug_gurobi
            policy_sim = simulate_policy_aug_gurobi   # if you wrote this; otherwise use simulate_policy_aug

        if x_opt_aug is None:
            print("\n[Global PCTL+Until LP] No feasible policy.")
        else:
            print("\n=== Global LP with PCTL + Until ===")
            print("backend:", backend)
            print("Optimal expected cost:", float(J_pctl))
            print("P(reach GOAL):", float(p_goal))
            for name, val in (region_probs or {}).items():
                print(f"P(ever visit {name}):", float(val))
            for name, val in (until_probs or {}).items():
                print(f"P({name}):", float(val))
            print("Solve time:", round(t_pctl, 3), "s")

            policy_aug = policy_recover(mdp_aug, x_opt_aug)
            base_traj_pctl, aug_traj_pctl = policy_sim(mdp_aug, policy_aug)

            print("\nTrajectory under PCTL-constrained policy (base states):")
            print(base_traj_pctl)

            base_policy = collapse_augmented_policy_to_base(mdp_aug, policy_aug)
            print("\nFinal policy (collapsed to base MDP):")
            print_policy_grid(mdp, base_policy)

    if solver == "idual_trevizan":
        pctl_cfg = cfg.get("pctl", {})
        idual_cfg = cfg.get("idual", {})

        mdp_aug, p_goal_min, extra_constraints = build_augmented_from_pctl_cfg(mdp, pctl_cfg)

        # --- Base DP to build cost heuristic ---
        V_base = value_iteration_shortest_path(mdp)

        # cost-to-go heuristic defined on augmented states by physical component
        H_cost = {st_aug: float(V_base.get(st_aug[0], 0.0)) for st_aug in mdp_aug.states_aug}

        # --- Region / Until heuristics (computed on augmented MDP) ---
        H_region_lower = {}
        H_region_upper = {}
        H_until_lower = {}
        H_until_upper = {}

        if idual_cfg.get("use_region_heuristic_lower", True):
            # For visit_region_min constraints, a useful heuristic is max achievable visit prob
            for f in pctl_cfg.get("flags", []):
                name = f["name"]
                H_region_lower[name] = compute_max_prob_visit_region_before_goal(mdp_aug, name)

        if idual_cfg.get("use_region_heuristic_upper", False):
            # For visit_region_max constraints, a useful heuristic is min achievable visit prob
            for f in pctl_cfg.get("flags", []):
                name = f["name"]
                H_region_upper[name] = compute_min_prob_visit_region_before_goal(mdp_aug, name)

        if idual_cfg.get("use_until_heuristic_lower", True):
            for us in pctl_cfg.get("until_specs", []):
                uname = us["name"]
                H_until_lower[uname] = compute_max_prob_until_before_goal(mdp_aug, uname)

        if idual_cfg.get("use_until_heuristic_upper", True):
            for us in pctl_cfg.get("until_specs", []):
                uname = us["name"]
                H_until_upper[uname] = compute_min_prob_until_before_goal(mdp_aug, uname)

        # Choose which cost heuristic to pass
        H_to_pass = H_cost if idual_cfg.get("use_cost_heuristic", True) else None

        (
            x_final,
            goal_prob_final,
            region_flag_final,
            until_final,
            total_time,
            final_lp_time,
            n_iters,
            envelope_size,
            obj_final,
        ) = i_dual_aug_trevizan(
            mdp_aug,
            p_goal_min=p_goal_min,
            extra_constraints=extra_constraints,
            H=H_to_pass,
            H_region_upper=H_region_upper,
            H_region_lower=H_region_lower,
            H_until_upper=H_until_upper,
            H_until_lower=H_until_lower,
        )

        if x_final is None:
            print("\n[i-dual] No feasible policy under these constraints.")
        else:
            print("\n=== i-dual (Trevizan-style) with PCTL + Until ===")
            print("P(reach GOAL):", float(goal_prob_final))
            for name, val in region_flag_final.items():
                print(f"[i-dual] P(ever visit {name}):", float(val))
            for name, val in until_final.items():
                print(f"[i-dual] P({name}):", float(val))

            print("\n=== timing summary ===")
            print("total i-dual time:", float(total_time))
            print("final LP time:", float(final_lp_time))
            print("iterations:", int(n_iters))
            print("final envelope size:", int(envelope_size))
            print("i-dual objective:", float(obj_final))

            policy_aug = recover_policy_from_x_aug(mdp_aug, x_final)
            base_traj, aug_traj = simulate_policy_aug(mdp_aug, policy_aug)
            print("\nTrajectory under i-dual policy (base states):")
            print(base_traj)

            base_policy = collapse_augmented_policy_to_base(mdp_aug, policy_aug)
            print("\nFinal policy (collapsed to base MDP):")
            print_policy_grid(mdp, base_policy)


    if solver not in ("dp", "lp", "both", "pctl_lp", "idual_trevizan"):
        raise ValueError("Unknown run.solver. Use dp | lp | both | pctl_lp | idual_trevizan.")




if __name__ == "__main__":
    main()
