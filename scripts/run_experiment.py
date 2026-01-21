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
    spec can be:
      - "all": whole grid
      - {"cells": [[r,c], ...]}
      - {"rect": [r0, c0, r1, c1]}  inclusive bounds like your make_grid_world uses
    """
    if spec == "all":
        return {(r, c) for r in range(N) for c in range(N)}

    if isinstance(spec, dict) and "cells" in spec:
        return {tuple(x) for x in spec["cells"]}

    if isinstance(spec, dict) and "rect" in spec:
        r0, c0, r1, c1 = spec["rect"]
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    raise ValueError(f"Bad region spec: {spec}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # --- Load config ---
    cfg = yaml.safe_load(open(args.config, "r"))
    mdp = build_mdp(cfg["mdp"])

    run_cfg = cfg.get("run", {})
    solver = run_cfg.get("solver", "both").lower()
    if solver not in ("dp", "lp", "both", "pctl_lp"):
        raise ValueError("Unknown run.solver. Use dp | lp | both | pctl_lp.")

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

        J_pctl, p_goal, x_opt_aug, region_probs, until_probs, t_pctl = solve_lp_with_pctl_aug(
            mdp_aug,
            p_goal_min=p_goal_min,
            region_constraints=region_constraints,
            until_constraints=until_constraints,
        )

        if x_opt_aug is None:
            print("\n[Global PCTL+Until LP] No feasible policy.")
        else:
            print("\n=== Global LP with PCTL + Until ===")
            print("Optimal expected cost:", float(J_pctl))
            print("P(reach GOAL):", float(p_goal))
            for name, val in region_probs.items():
                print(f"P(ever visit {name}):", float(val))
            for name, val in until_probs.items():
                print(f"P({name}):", float(val))
            print("Solve time:", round(t_pctl, 3), "s")

            policy_aug = recover_policy_from_x_aug(mdp_aug, x_opt_aug)
            base_traj_pctl, aug_traj_pctl = simulate_policy_aug(mdp_aug, policy_aug)
            print("\nTrajectory under PCTL-constrained policy (base states):")
            print(base_traj_pctl)

            base_policy = collapse_augmented_policy_to_base(mdp_aug, policy_aug)
            print("\nFinal policy (collapsed to base MDP):")
            print_policy_grid(mdp, base_policy)
    

    if solver not in ("dp", "lp", "both", "pctl_lp"):
        raise ValueError(f"Unknown run.solver={solver}. Use dp | lp | both | pctl_lp.")



if __name__ == "__main__":
    main()
