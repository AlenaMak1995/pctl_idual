# pctl_idual

This repository provides:
- a GridWorld Markov Decision Process (MDP) framework,
- a Dynamic Programming (DP) solver for shortest-path problems (ground truth),
- LP- and i-dual–based solvers (with PCTL constraints).

The DP solver serves as an unconstrained baseline and ground truth
against which LP and i-dual methods are compared.

---

## 1. GridWorld MDP

We consider a finite-state GridWorld MDP where:
- states are grid cells `(row, col)`,
- actions move the agent in the four cardinal directions,
- transitions may be stochastic (slip probability),
- costs are incurred per cell (with optional rectangular cost regions),
- goal states are absorbing.

### Example GridWorld (4×4)

![4x4 grid](experiments/figures/grid_4x4.png)

### Example GridWorld (20×20)

![4x4 grid](experiments/figures/grid_20x20.png)

Start state: $s_0 = (19,0)$
Target/goal state: $g = (0,19)$

Cost structure (cost of \emph{entering} the successor cell):
Blue region $L$: very cheap cells, cost $0.1$
Red region $H$: expensive cells, cost $10$
 All other cells (including $G2$ and $G3$): default cost $1$

---

## 2. Dynamic Programming (Ground Truth)

The DP solver computes the **unconstrained shortest-path value function**
using value iteration:

\[
V(s) = \min_a \sum_{s'} P(s' \mid s, a)\left[c(s,a,s') + V(s')\right]
\]

This provides the optimal expected cost-to-go from every state.

From the value function, we extract a greedy policy and simulate a trajectory.

---

## 3. Running a DP experiment

### Step 1: Create a YAML config

```yaml
exp_name: dp_custom

mdp:
  world: "custom"
  N: 50
  start: [49, 0]
  goal: [[0, 49]]
  slip_prob: 0.0
  default_cost: 1.0
  rect_costs:
    - [16, 16, 19, 19, 0.1]
    - [5, 5, 14, 14, 10.0]



