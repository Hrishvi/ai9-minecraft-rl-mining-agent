#!/usr/bin/env python3
"""
mine_sim.py - a tiny gridworld that mirrors the Minecraft mining task. This is
where the agent learns (fast), before rl_brain.py deploys the policy in-game.

THE ABSTRACTION
---------------
A WxH grid. ORE cells are solid (can't walk into, must MINE from an adjacent
cell). LAVA cells kill on contact. The agent walks empty cells. An episode ends
on lava death, clearing all ore, or running out of steps.

The observation is computed the same way rl_brain.py computes it from live game
state - nearest visible ore + reach + per-direction lava - so the learned policy
is meaningful in-game.

USAGE
-----
    python3 mine_sim.py                       # quick train -> qtable.json
    python3 mine_sim.py --watch               # ...and render one greedy episode
    python3 mine_sim.py --watch-every 1000 --delay 0.12   # WATCH IT LEARN: render
                                              # a greedy episode every 1000 episodes
    (for the train-vs-heuristic comparison + chart, use evaluate.py)
"""

import argparse
import random
import time

from rl_agent import MOVE_VECTORS, QLearningAgent, bucket_direction, encode_state

EMPTY, ORE, LAVA = 0, 1, 2

# Reward design
R_STEP = -0.02
R_SHAPE = 0.30      # moved closer to nearest ore
R_BLOCKED = -0.20   # walked into ore / off the edge
R_MISMINE = -0.50   # MINE with nothing in reach
R_ORE = 10.0
R_CLEAR = 5.0
R_LAVA = -12.0      # stepped into lava (death)


class MineGrid:
    def __init__(self, size=8, n_ore=5, n_lava=6, max_steps=120):
        self.size = size
        self.n_ore = n_ore
        self.n_lava = n_lava
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.grid = [[EMPTY] * self.size for _ in range(self.size)]
        free = [(x, z) for x in range(self.size) for z in range(self.size)]
        random.shuffle(free)
        self.ax, self.az = free.pop()
        for _ in range(self.n_ore):
            x, z = free.pop()
            self.grid[x][z] = ORE
        for _ in range(self.n_lava):
            x, z = free.pop()
            self.grid[x][z] = LAVA
        self.ore_left = self.n_ore
        self.steps = 0
        return self.observe()

    # --- helpers -------------------------------------------------------------
    def _in_bounds(self, x, z):
        return 0 <= x < self.size and 0 <= z < self.size

    def _ore_cells(self):
        return [(x, z) for x in range(self.size) for z in range(self.size)
                if self.grid[x][z] == ORE]

    def _nearest_ore(self):
        ores = self._ore_cells()
        if not ores:
            return None
        return min(ores, key=lambda c: abs(c[0] - self.ax) + abs(c[1] - self.az))

    def _lava_dirs(self):
        out = []
        for d in ("MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W"):
            dx, dz = MOVE_VECTORS[d]
            x, z = self.ax + dx, self.az + dz
            out.append(self._in_bounds(x, z) and self.grid[x][z] == LAVA)
        return tuple(out)

    def observe_structured(self):
        """Return (ore_dir, ore_near, lava_dirs) - used by the heuristic baseline."""
        nearest = self._nearest_ore()
        lava = self._lava_dirs()
        if nearest is None:
            return ("NONE", False, lava)
        dx, dz = nearest[0] - self.ax, nearest[1] - self.az
        return (bucket_direction(dx, dz), (abs(dx) + abs(dz)) <= 1, lava)

    def observe(self):
        ore_dir, ore_near, lava = self.observe_structured()
        return encode_state(ore_dir, ore_near, lava)

    def _dist_to_ore(self):
        nearest = self._nearest_ore()
        return None if nearest is None else abs(nearest[0] - self.ax) + abs(nearest[1] - self.az)

    def _adjacent_ore(self):
        for dx, dz in MOVE_VECTORS.values():
            x, z = self.ax + dx, self.az + dz
            if self._in_bounds(x, z) and self.grid[x][z] == ORE:
                return (x, z)
        return None

    # --- transition ----------------------------------------------------------
    def step(self, action):
        self.steps += 1
        reward = R_STEP
        done = False

        if action in MOVE_VECTORS:
            dx, dz = MOVE_VECTORS[action]
            nx, nz = self.ax + dx, self.az + dz
            if not self._in_bounds(nx, nz):
                reward += R_BLOCKED
            elif self.grid[nx][nz] == LAVA:
                reward += R_LAVA
                done = True
            elif self.grid[nx][nz] == ORE:
                reward += R_BLOCKED
            else:
                before = self._dist_to_ore()
                self.ax, self.az = nx, nz
                if before is not None:
                    after = self._dist_to_ore()
                    if after is not None and after < before:
                        reward += R_SHAPE

        elif action == "MINE":
            target = self._adjacent_ore()
            if target is not None:
                self.grid[target[0]][target[1]] = EMPTY
                self.ore_left -= 1
                reward += R_ORE
                if self.ore_left == 0:
                    reward += R_CLEAR
                    done = True
            else:
                reward += R_MISMINE

        if self.steps >= self.max_steps:
            done = True
        return self.observe(), reward, done

    # --- pretty print --------------------------------------------------------
    def render(self):
        glyph = {EMPTY: ".", ORE: "O", LAVA: "L"}
        for z in range(self.size):
            row = ["A" if (x, z) == (self.ax, self.az) else glyph[self.grid[x][z]]
                   for x in range(self.size)]
            print(" ".join(row))


def play_and_render(env, agent, delay=0.0, label="", max_frames=120):
    """Play one greedy episode, rendering each frame (with an optional pause)."""
    if label:
        print(f"\n----- {label} -----")
    state = env.reset()
    done, total, frames = False, 0.0, 0
    while not done:
        if frames < max_frames:
            env.render()
            print()
            if delay:
                time.sleep(delay)
        frames += 1
        state, reward, done = env.step(state if False else agent.choose(state, greedy=True))
        total += reward
    env.render()
    result = "cleared all ore" if env.ore_left == 0 else f"{env.ore_left} ore left"
    print(f"  -> {result} in {env.steps} steps (return {total:.1f})\n")


def train(args):
    random.seed(args.seed)
    env = MineGrid(size=args.grid, n_ore=args.ore, n_lava=args.lava, max_steps=args.max_steps)
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=args.epsilon_start)
    window = []
    print(f"Training {args.episodes} episodes on {args.grid}x{args.grid} "
          f"({args.ore} ore, {args.lava} lava)...\n")
    print(f"{'episode':>8} | {'avg ore/ep':>10} | {'epsilon':>7}")
    print("-" * 32)
    for ep in range(1, args.episodes + 1):
        agent.epsilon = args.epsilon_start + (ep / args.episodes) * (args.epsilon_end - args.epsilon_start)
        s = env.reset()
        start = env.ore_left
        done = False
        while not done:
            a = agent.choose(s)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns
        window.append(start - env.ore_left)
        if ep % args.report_every == 0:
            print(f"{ep:>8} | {sum(window) / len(window):>10.2f} | {agent.epsilon:>7.3f}")
            window = []
        if args.watch_every and ep % args.watch_every == 0:
            play_and_render(env, agent, args.delay, label=f"episode {ep} (greedy snapshot)")
    agent.save(args.out)
    print(f"\nSaved policy to {args.out} ({len(agent.q)} states visited).")
    if args.watch:
        play_and_render(env, agent, args.delay, label="final greedy episode")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--ore", type=int, default=5)
    p.add_argument("--lava", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=120, dest="max_steps")
    p.add_argument("--epsilon-start", type=float, default=0.30, dest="epsilon_start")
    p.add_argument("--epsilon-end", type=float, default=0.02, dest="epsilon_end")
    p.add_argument("--report-every", type=int, default=1000, dest="report_every")
    p.add_argument("--watch", action="store_true", help="render one greedy episode at the end")
    p.add_argument("--watch-every", type=int, default=0, dest="watch_every",
                   help="render a greedy episode every N training episodes (watch it learn)")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between rendered frames")
    p.add_argument("--out", type=str, default="qtable.json")
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
