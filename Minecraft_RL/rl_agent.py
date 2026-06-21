#!/usr/bin/env python3
"""
rl_agent.py - the learning core, shared by the trainer/evaluator (mine_sim.py,
evaluate.py) and the live deploy brain (rl_brain.py).

Tabular Q-learning, from scratch (no ML libraries). It works because we compress
the world into a small discrete observation.

WHAT THE AGENT SEES (the observation)
-------------------------------------
Identical in the sim and in-game so a trained policy transfers:
    ore_dir        : compass bucket toward the nearest VISIBLE ore (8 dirs + NONE)
    ore_near       : is that ore within mining reach? (0/1)
    lava N/E/S/W   : is lava in each of the 4 cardinal neighbours? (0/1 each)

So 9 x 2 x 16 = 288 possible states. Per-direction lava (rather than a single
"lava nearby" bit) is what lets the agent learn to steer AROUND lava instead of
just knowing it's somewhere close.

ACTIONS
-------
    MOVE_N, MOVE_E, MOVE_S, MOVE_W, MINE, WAIT

The agent learns the MINING sub-task. Returning home when full is scripted in the
brain (not learnable from this observation, which doesn't encode where home is).

BASELINES (for comparison in evaluate.py)
-----------------------------------------
    random_action()   - the floor.
    heuristic_action  - the "told" policy: head to the nearest ore and mine it.
                        It is given the same observation but only acts on the ore
                        part; it knows nothing about lava. The learned agent beats
                        it by using the lava information it was never told to use.
"""

import json
import math
import random

# --- Action space ------------------------------------------------------------
ACTIONS = ["MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W", "MINE", "WAIT"]

# Minecraft convention: +X = East, +Z = South, so North is -Z.
MOVE_VECTORS = {
    "MOVE_N": (0, -1),
    "MOVE_E": (1, 0),
    "MOVE_S": (0, 1),
    "MOVE_W": (-1, 0),
}

_COMPASS = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]

# Which cardinal move best approaches each 8-way ore bucket (diagonals pick one).
_DIR_TO_MOVE = {
    "N": "MOVE_N", "S": "MOVE_S", "E": "MOVE_E", "W": "MOVE_W",
    "NE": "MOVE_E", "SE": "MOVE_E", "NW": "MOVE_W", "SW": "MOVE_W",
}


def bucket_direction(dx, dz):
    """Map a (dx, dz) vector to one of 8 compass buckets, or NONE if zero."""
    if dx == 0 and dz == 0:
        return "NONE"
    angle = math.degrees(math.atan2(dz, dx))   # 0 = East, +90 = South
    idx = int(((angle + 22.5) % 360) // 45)
    return _COMPASS[idx]


def encode_state(ore_dir, ore_near, lava_dirs):
    """Pack the observation into a string key. lava_dirs = (N, E, S, W) booleans."""
    ln, le, ls, lw = (1 if b else 0 for b in lava_dirs)
    return f"{ore_dir},{int(bool(ore_near))},{ln}{le}{ls}{lw}"


# --- baseline policies -------------------------------------------------------
def heuristic_action(ore_dir, ore_near):
    """The 'told' policy: seek the nearest ore and mine it. Ignores lava entirely."""
    if ore_near:
        return "MINE"
    if ore_dir == "NONE":
        return "WAIT"
    return _DIR_TO_MOVE.get(ore_dir, "WAIT")


def random_action():
    return random.choice(ACTIONS)


# --- the learner -------------------------------------------------------------
class QLearningAgent:
    """Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))."""

    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.2):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _row(self, state):
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in ACTIONS}
        return self.q[state]

    def choose(self, state, greedy=False):
        row = self._row(state)
        if not greedy and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        best = max(row.values())
        return random.choice([a for a, v in row.items() if v == best])

    def update(self, state, action, reward, next_state, done):
        row = self._row(state)
        future = 0.0 if done else self.gamma * max(self._row(next_state).values())
        row[action] += self.alpha * (reward + future - row[action])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"q": self.q, "meta": {"alpha": self.alpha, "gamma": self.gamma}}, f, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.q = json.load(f).get("q", {})
        return self
