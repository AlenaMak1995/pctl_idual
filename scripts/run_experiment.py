import argparse
import yaml

from pctl_idual.gridworld import (
    make_grid_world,
    make_4x4_world,
    make_20x20_world,
    make_NxN_world,
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

    else:
        raise ValueError(f"Unknown world type: {world}")

    return mdp

from pctl_idual.dp_solvers import (
    value_iteration_shortest_path,
    greedy_policy_from_V,
    simulate_policy,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # --- Load config ---
    cfg = yaml.safe_load(open(args.config, "r"))
    mdp = build_mdp(cfg["mdp"])

    # --- DP: unconstrained shortest path ---
    V = value_iteration_shortest_path(mdp)
    print("DP (unconstrained) optimal cost from START:", V[mdp.start])

    pi = greedy_policy_from_V(mdp, V)
    traj = simulate_policy(mdp, pi)

    print("Trajectory under DP (unconstrained):")
    print(traj)


if __name__ == "__main__":
    main()
