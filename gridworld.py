# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Set, List

Action = str
State = Tuple[int, int]
Region = Set[State]

ACTIONS: List[Action] = ["U", "D", "L", "R"]
DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@dataclass
class GridWorld:
    """
    Base Grid MDP.

    The possible changes:
    - N (grid size)
    - define an arbitrary goal region
    - define arbitrary cost structure via cost_cell
    - define transition probabilities via slip_prob 
    """
    N: int
    start: State
    goal: Region
    cost_cell: Callable[[int, int], float]
    # probability of "slipping" to a random other action
    slip_prob: float = 0.0   

    def __post_init__(self):
        self.states: List[State] = [
            (r, c) for r in range(self.N) for c in range(self.N)
        ]

    # --- geometry helpers with grid borders ---

    def clamp(self, r: int, c: int) -> State:
        return max(0, min(self.N - 1, r)), max(0, min(self.N - 1, c))

    def move(self, s: State, a: Action) -> State:
        """Deterministic movement for a single action."""
        dr, dc = DELTA[a]
        r, c = s
        return self.clamp(r + dr, c + dc)

    # --- MDP interface ---

    def is_goal(self, s: State) -> bool:
        return s in self.goal

    def actions_from(self, s: State) -> List[Action]:
        if self.is_goal(s):
            return []
        return ACTIONS

    def transitions(self, s: State, a: Action) -> Dict[State, float]:
        """
        P(s' | s, a): transition probabilities.

        - with prob 1 - slip_prob: go in direction a
        - with prob slip_prob: choose uniformly among other actions
        """
        if self.is_goal(s):
            return {s: 1.0}

        next_main = self.move(s, a)

        if self.slip_prob <= 0:
            return {next_main: 1.0}

        others = [b for b in ACTIONS if b != a]
        p_main = 1.0 - self.slip_prob
        p_slip_each = self.slip_prob / len(others)

        probs: Dict[State, float] = {}
        # main move
        probs[next_main] = probs.get(next_main, 0.0) + p_main
        # slips
        for b in others:
            s2 = self.move(s, b)
            probs[s2] = probs.get(s2, 0.0) + p_slip_each
        return probs

    def cost(self, s: State, a: Action) -> float:
        """
        Immediate cost for taking a in state s.
        Current setup: cost of the next cell.
        """
        s2 = self.move(s, a)
        r2, c2 = s2
        return self.cost_cell(r2, c2)


# =========================
# Construction + debugging
# =========================

def make_grid_world(
    N: int,
    start: State,
    goal: Region,
    default_cost: float = 1.0,
    cell_costs: Dict[State, float] | None = None,
    rect_costs: List[tuple[int, int, int, int, float]] | None = None,
    slip_prob: float = 0.0,
) -> GridWorld:
    """
    Helper to build a grid with arbitrarily expensive/cheap regions.

    rect_costs: list of (r0, c0, r1, c1, cost) rectangles.
    """
    cell_costs = cell_costs or {}
    rect_costs = rect_costs or []

    def cost_cell(r: int, c: int) -> float:
        # 1) explicit cell overrides
        if (r, c) in cell_costs:
            return cell_costs[(r, c)]
        # 2) rectangular regions
        for (r0, c0, r1, c1, cost) in rect_costs:
            if r0 <= r <= r1 and c0 <= c <= c1:
                return cost
        # 3) default
        return default_cost

    return GridWorld(
        N=N,
        start=start,
        goal=goal,
        cost_cell=cost_cell,
        slip_prob=slip_prob,
    )


def print_cost_grid(mdp: GridWorld, digits: int = 1):
    """ Print the cost of each cell to visually check obstacles/expensive regions."""
    for r in range(mdp.N):
        row = []
        for c in range(mdp.N):
            row.append(f"{mdp.cost_cell(r, c):.{digits}f}")
        print("\t".join(row))
    print()

def make_4x4_world() -> GridWorld:
    N = 4
    goal = {(0, 3)}
    start = (3, 0)
    cell_cost_dict = {
        (1, 0): 10,
        (1, 1): 20,
        (1, 2): 5,
        (2, 3): 8,
    }

    def cost_cell(r, c):
        # default cost 1, override some cells
        return cell_cost_dict.get((r, c), 1.0)

    return GridWorld(N=N, start=start, goal=goal, cost_cell=cost_cell)

def make_4x4_pctl_world() -> GridWorld:
    N = 4
    goal = {(0, 3)}
    start = (3, 0)
    cell_cost_dict = {
        (1, 0): 10,
        (1, 1): 5,
        (1, 2): 8,
        (2, 1): 3,
        (2, 3): 5,
        (3, 2): 1,
    }

    def cost_cell(r, c):
        return cell_cost_dict.get((r, c), 1.0)
    return GridWorld(N=N, start=start, goal=goal, cost_cell=cost_cell)
    
def make_20x20_world() -> GridWorld:
    N = 20
    start = (19, 0)
    goal = {(0, 19)}

    def cost_cell(r, c):
        if 16 <= r <= 19 and 16 <= c <= 19:
            return 0.1
        if 5 <= r <= 14 and 5 <= c <= 14:
            return 10.0
        return 1.0

    return GridWorld(N=N, start=start, goal=goal, cost_cell=cost_cell)


def make_NxN_world(N: int) -> GridWorld:
    start = (N-1, 0)
    goal  = {(0, N-1)}

    def cost_cell(r, c):
        # cheap top-right corner
        if N-4 <= r <= N-1 and N-4 <= c <= N-1:
            return 0.1
        # expensive central block
        if N//4 <= r <= 3*N//4 and N//4 <= c <= 3*N//4:
            return 10.0
        return 1.0

    return GridWorld(N=N, start=start, goal=goal, cost_cell=cost_cell)

