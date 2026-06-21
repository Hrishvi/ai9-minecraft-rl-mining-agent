#!/usr/bin/env python3
"""
evaluate.py - the "is it actually learning, and is it better than just being
told the rule?" experiment.

It trains the Q-learning miner (checkpointing greedy performance -> the learning
curve), then evaluates three policies on many fresh maps:

    random     - the floor.
    heuristic  - the "told" policy: go to the nearest ore and mine it; it knows
                 nothing about lava.
    learned    - the trained Q-table, acting greedily.

We score TWO things, because they tell different stories:
    gross ore  - blocks mined per episode (death still ends the episode).
    kept ore   - blocks you actually walk away with. In Minecraft, dying in lava
                 destroys your inventory, so a run that ends in lava banks 0. This
                 is the metric that matters, and it's where surviving pays off.
"""

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mine_sim import MineGrid, R_LAVA
from rl_agent import QLearningAgent, heuristic_action, random_action

DEATH_THRESHOLD = R_LAVA / 2.0  # a step reward below this only happens on lava death


def make_env(args):
    return MineGrid(size=args.grid, n_ore=args.ore, n_lava=args.lava, max_steps=args.max_steps)


def eval_policy(kind, env, episodes, agent=None):
    """Returns (gross_ore, kept_ore, death_rate) averaged over fresh maps."""
    gross, kept, deaths = [], [], 0
    for _ in range(episodes):
        state = env.reset()
        struct = env.observe_structured()
        start = env.ore_left
        done, died = False, False
        while not done:
            if kind == "random":
                action = random_action()
            elif kind == "heuristic":
                action = heuristic_action(struct[0], struct[1])
            else:
                action = agent.choose(state, greedy=True)
            state, reward, done = env.step(action)
            struct = env.observe_structured()
            if reward <= DEATH_THRESHOLD:
                died = True
        mined = start - env.ore_left
        gross.append(mined)
        kept.append(0 if died else mined)   # die in lava -> lose the haul
        if died:
            deaths += 1
    n = len(gross)
    return sum(gross) / n, sum(kept) / n, deaths / n


def train(env, agent, episodes, eps_start, eps_end, checkpoint_every, eval_n):
    """Train, recording greedy KEPT-ore at each checkpoint (the learning curve)."""
    xs, ys = [], []
    for ep in range(1, episodes + 1):
        agent.epsilon = eps_start + (ep / episodes) * (eps_end - eps_start)
        state = env.reset()
        done = False
        while not done:
            action = agent.choose(state)
            nxt, reward, done = env.step(action)
            agent.update(state, action, reward, nxt, done)
            state = nxt
        if ep % checkpoint_every == 0:
            _, kept, _ = eval_policy("learned", env, eval_n, agent)
            xs.append(ep)
            ys.append(kept)
            print(f"  episode {ep:>5}: greedy kept-ore = {kept:.2f}")
    return xs, ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--ore", type=int, default=5)
    p.add_argument("--lava", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=120, dest="max_steps")
    p.add_argument("--eval-episodes", type=int, default=1500, dest="eval_episodes")
    p.add_argument("--checkpoint-every", type=int, default=1000, dest="checkpoint_every")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", type=str, default="qtable.json")
    p.add_argument("--plot", type=str, default="rl_vs_heuristic.png")
    args = p.parse_args()

    random.seed(args.seed)
    env = make_env(args)
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.3)  # lower alpha = steadier learning

    print(f"Training on {args.grid}x{args.grid} ({args.ore} ore, {args.lava} lava)...")
    checkpoint_eval = max(300, args.eval_episodes // 3)
    xs, ys = train(env, agent, args.episodes, 0.30, 0.02, args.checkpoint_every, checkpoint_eval)
    agent.save(args.out)

    print(f"\nEvaluating each policy on {args.eval_episodes} fresh maps...")
    rnd = eval_policy("random", env, args.eval_episodes)
    heu = eval_policy("heuristic", env, args.eval_episodes)
    lrn = eval_policy("learned", env, args.eval_episodes, agent)

    print(f"\n{'policy':>10} | {'gross ore':>9} | {'kept ore':>8} | {'death rate':>10}")
    print("-" * 48)
    for name, (g, k, d) in [("random", rnd), ("heuristic", heu), ("learned", lrn)]:
        print(f"{name:>10} | {g:>9.2f} | {k:>8.2f} | {d:>10.1%}")

    # --- chart ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(xs, ys, "-o", color="#2c7fb8", label="learned (RL)")
    ax1.axhline(heu[1], ls="--", color="#d95f0e", label="told heuristic")
    ax1.axhline(rnd[1], ls=":", color="#999999", label="random")
    ax1.set_xlabel("training episodes")
    ax1.set_ylabel("ore kept / episode")
    ax1.set_title("Learning curve (ore safely banked)")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    labels = ["random", "told\nheuristic", "learned\n(RL)"]
    kept_vals = [rnd[1], heu[1], lrn[1]]
    gross_vals = [rnd[0], heu[0], lrn[0]]
    rates = [rnd[2], heu[2], lrn[2]]
    bars = ax2.bar(labels, kept_vals, color=["#999999", "#d95f0e", "#2c7fb8"])
    for b, k, g, d in zip(bars, kept_vals, gross_vals, rates):
        ax2.text(b.get_x() + b.get_width() / 2, k + 0.05,
                 f"kept {k:.2f}\n(mined {g:.2f}, {d:.0%} died)",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("ore kept / episode")
    ax2.set_title(f"Head-to-head ({args.eval_episodes} maps)")
    ax2.set_ylim(0, max(kept_vals) * 1.45)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Learned policy vs a model that was only told the rule", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"\nSaved chart to {args.plot}")


if __name__ == "__main__":
    main()
