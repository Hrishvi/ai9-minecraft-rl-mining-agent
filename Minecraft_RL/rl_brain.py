#!/usr/bin/env python3
"""
rl_brain.py - the live brain. Drop-in replacement for mining_brain.py: run THIS
instead, and ORE-mode decisions come from the Q-table trained by mine_sim.py /
evaluate.py. STRIP mode and RETURNING are unchanged (nothing to learn there).

The observation - nearest visible ore (direction + reach) plus per-direction lava
- is built from the live state report exactly as the sim builds it, so the trained
policy applies. The sim is an abstraction, so expect imperfect in-game behaviour
(the sim-to-real gap) - that's the tradeoff for training in seconds.

USAGE
-----
    python3 mine_sim.py        # or evaluate.py : train -> qtable.json
    python3 rl_brain.py        # then run this and play with !mine
    python3 rl_brain.py --learn
"""

import argparse
import json
import math
import os
import socket
import sys

from rl_agent import MOVE_VECTORS, QLearningAgent, bucket_direction, encode_state

HOST = "127.0.0.1"
PORT = 5555
ARRIVE_RADIUS = 2.0
NEAR_DIST = 2.0

HOME_MESSAGE = "Inventory full, back at base!"
STRIP_DONE_MESSAGE = "Work area cleared!"
LAVA_BLOCKED_MESSAGE = "Stopped - remaining blocks border lava."
LAVA_PAUSE_MESSAGE = "Lava nearby - pausing for safety."

# Rough live reward, only used with --learn (the signal is noisy; sim is primary).
LR_ORE = 1.0
LR_HAZARD = -1.0
LR_STEP = -0.02

# action -> which lava_* flag would mean walking into lava
_LAVA_FLAG = {"MOVE_N": "lava_n", "MOVE_E": "lava_e", "MOVE_S": "lava_s", "MOVE_W": "lava_w"}


class RLBrain:
    def __init__(self, agent, learn=False):
        self.agent = agent
        self.learn = learn
        self.qtable_path = "qtable.json"

        self.mode = None
        self.target_blocks = set()
        self.home_base = None
        self.state = "IDLE"

        self._announced_home = False
        self._announced_done = False
        self._announced_hazard = False

        self._prev_state = None
        self._prev_action = None
        self._prev_empty = None
        self._updates = 0

    # ---------------------------------------------------------------- commands
    def set_targets(self, blocks, home):
        self.mode = "ore"
        self.target_blocks = set(blocks)
        self.home_base = (home["x"], home["y"], home["z"])
        self.state = "MINING"
        self._reset_episode()
        print(f"[rl] ORE job (learned policy). targets={sorted(self.target_blocks)}", flush=True)

    def set_strip(self, home, box):
        self.mode = "strip"
        self.home_base = (home["x"], home["y"], home["z"])
        self.state = "MINING"
        self._reset_episode()
        print(f"[rl] STRIP job (deterministic sweep). box={box}", flush=True)

    def stop(self):
        self.mode = None
        self.state = "IDLE"
        self._reset_episode()
        print("[rl] Stopped -> IDLE", flush=True)

    def _reset_episode(self):
        self._announced_home = False
        self._announced_done = False
        self._announced_hazard = False
        self._prev_state = None
        self._prev_action = None
        self._prev_empty = None

    # ------------------------------------------------------------ decision loop
    def decide(self, state_msg):
        player = state_msg.get("player", {})
        px, py, pz = player.get("x", 0.0), player.get("y", 0.0), player.get("z", 0.0)
        inv = state_msg.get("inventory", {})
        empty_slots = inv.get("empty_slots", 0)
        inventory_full = bool(inv.get("is_inventory_full", False))

        if self.state == "MINING":
            if inventory_full:
                self.state = "RETURNING"
                self._announced_home = False
                print("[rl] Inventory full. MINING -> RETURNING", flush=True)
                return self._goto_home_action()

            if self.mode == "strip":
                return self._decide_strip(state_msg)
            return self._decide_ore(px, py, pz, empty_slots, state_msg)

        if self.state == "RETURNING":
            if self.home_base is None:
                self.state = "IDLE"
                return {"action": "idle"}
            dist = _dist(px, py, pz, *self.home_base)
            if dist <= ARRIVE_RADIUS:
                self.state = "IDLE"
                if not self._announced_home:
                    self._announced_home = True
                    return {"action": "idle", "say": HOME_MESSAGE}
                return {"action": "idle"}
            return self._goto_home_action()

        return {"action": "idle"}

    # --------------------------------------------------------- learned ORE mode
    def _decide_ore(self, px, py, pz, empty_slots, state_msg):
        lava_dirs = (
            bool(state_msg.get("lava_n", False)),
            bool(state_msg.get("lava_e", False)),
            bool(state_msg.get("lava_s", False)),
            bool(state_msg.get("lava_w", False)),
        )
        nearest = self._nearest_target(px, pz, state_msg.get("targets", []))

        if nearest is None:
            obs = encode_state("NONE", False, lava_dirs)
        else:
            dx = (nearest["x"] + 0.5) - px
            dz = (nearest["z"] + 0.5) - pz
            ore_dir = bucket_direction(dx, dz)
            ore_near = math.sqrt(dx * dx + dz * dz) <= NEAR_DIST
            obs = encode_state(ore_dir, ore_near, lava_dirs)

        # Optional online update from the previous step.
        if self.learn and self._prev_state is not None:
            reward = LR_STEP + (LR_HAZARD if any(lava_dirs) else 0.0)
            if self._prev_empty is not None and empty_slots < self._prev_empty:
                reward += LR_ORE
            self.agent.update(self._prev_state, self._prev_action, reward, obs, False)
            self._updates += 1
            if self._updates % 200 == 0:
                self.agent.save(self.qtable_path)

        action = self.agent.choose(obs, greedy=not self.learn)

        # Hard safety net: never walk into a cell we can see is lava.
        if action in _LAVA_FLAG and state_msg.get(_LAVA_FLAG[action], False):
            action = "WAIT"

        self._prev_state, self._prev_action, self._prev_empty = obs, action, empty_slots

        if action in MOVE_VECTORS:
            dx, dz = MOVE_VECTORS[action]
            return {"action": "goto", "target": [px + dx, py, pz + dz]}
        if action == "MINE":
            if nearest is not None:
                return {"action": "mine", "target": [nearest["x"], nearest["y"], nearest["z"]]}
            return {"action": "idle"}
        return {"action": "idle"}  # WAIT

    # --------------------------------------------------- deterministic STRIP mode
    def _decide_strip(self, state_msg):
        if bool(state_msg.get("hazard", False)):
            if not self._announced_hazard:
                self._announced_hazard = True
                return {"action": "idle", "say": LAVA_PAUSE_MESSAGE}
            return {"action": "idle"}
        self._announced_hazard = False

        strip = state_msg.get("strip", {})
        nxt = strip.get("next")
        if not nxt:
            self.state = "IDLE"
            if not self._announced_done:
                self._announced_done = True
                if strip.get("blocked"):
                    return {"action": "idle", "say": LAVA_BLOCKED_MESSAGE}
                return {"action": "idle", "say": STRIP_DONE_MESSAGE}
            return {"action": "idle"}
        return {"action": "mine", "target": list(nxt)}

    # ----------------------------------------------------------------- helpers
    def _goto_home_action(self):
        return {"action": "goto", "target": list(self.home_base)}

    def _nearest_target(self, px, pz, candidates):
        matching = [c for c in candidates if c.get("id") in self.target_blocks]
        if not matching:
            return None
        return min(matching, key=lambda c: (c["x"] + 0.5 - px) ** 2 + (c["z"] + 0.5 - pz) ** 2)


def _dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def handle_message(brain, raw):
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    mtype = msg.get("type")
    if mtype == "set_targets":
        brain.set_targets(msg.get("blocks", []), msg.get("home", {}))
        return {"action": "idle"}
    if mtype == "set_strip":
        brain.set_strip(msg.get("home", {}), msg.get("box", {}))
        return {"action": "idle"}
    if mtype == "stop":
        brain.stop()
        return {"action": "idle"}
    if mtype == "state":
        return brain.decide(msg)
    return None


def serve(brain):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[rl] Listening on {HOST}:{PORT} - waiting for the mod...", flush=True)
        while True:
            conn, addr = server.accept()
            print(f"[rl] Mod connected from {addr}", flush=True)
            buffer = ""
            with conn:
                writer = conn.makefile("w", encoding="utf-8", newline="\n")
                try:
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buffer += chunk.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            action = handle_message(brain, line)
                            if action is not None:
                                writer.write(json.dumps(action) + "\n")
                                writer.flush()
                except (ConnectionResetError, BrokenPipeError):
                    pass
            print("[rl] Mod disconnected. Waiting for reconnect...", flush=True)


def main():
    p = argparse.ArgumentParser(description="Deploy a trained Q-learning policy into Minecraft.")
    p.add_argument("--qtable", type=str, default="qtable.json")
    p.add_argument("--learn", action="store_true", help="keep updating from live play (experimental)")
    args = p.parse_args()

    agent = QLearningAgent(epsilon=0.1 if args.learn else 0.0)
    if os.path.exists(args.qtable):
        agent.load(args.qtable)
        print(f"[rl] Loaded policy from {args.qtable} ({len(agent.q)} states).", flush=True)
    else:
        print(f"[rl] WARNING: {args.qtable} not found. Train first: python3 mine_sim.py", flush=True)

    brain = RLBrain(agent, learn=args.learn)
    brain.qtable_path = args.qtable
    try:
        serve(brain)
    except KeyboardInterrupt:
        if args.learn:
            agent.save(args.qtable)
        print("\n[rl] Shutting down.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
