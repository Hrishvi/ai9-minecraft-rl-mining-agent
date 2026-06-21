#!/usr/bin/env python3
"""
mining_brain.py - the decision brain for the companion mining agent.

ARCHITECTURE
------------
  - The Fabric mod is the CLIENT; this script is the SERVER (it listens on a
    local TCP port).
  - The mod streams JSON "state" lines every couple of ticks. The brain runs the
    state machine and streams back a single "action" line per state report.

NOTE ON "AI": this brain is hand-written automation - a state machine plus a
greedy nearest-target rule and a fixed sweep. It is "AI" only in the loose
game-dev sense; nothing here learns. It is, however, the right shape to bolt a
learned policy onto later: the state report is an observation, the action set is
an action space, and a reward function would slot in around decide().

TWO MODES (both line-of-sight legit, no X-ray):
  ORE   - "!mine a b c": mine the closest visible target ore.
  STRIP - "!strip" after "!corner1"/"!corner2": clear the whole selected box,
          block by block, collecting whatever ore turns up.

LAVA is treated as SAFETY, not detection. The mod skips lava-touching blocks in
STRIP (leaving a buffer wall) and flags a "hazard" when lava is next to the
player; this brain pauses on that flag. No hidden-lava scanning.

STATES
------
  IDLE      -> no active job.
  MINING    -> active work (ore vs strip by mode). Inventory full -> RETURNING.
               Hazard flag -> pause until it clears.
  RETURNING -> walk to Home Base; on arrival stop and announce once.

MESSAGE FORMATS
---------------
  In  (mod -> brain):
      {"type": "set_targets", "blocks": [...], "home": {...}}
      {"type": "set_strip", "home": {...}, "box": {...}}
      {"type": "stop"}
      {"type": "state", "player": {...}, "inventory": {...},
       "mode": "ore|strip|none", "hazard": bool,
       "targets": [...],                      # ORE mode
       "strip": {"next": [x,y,z] | null, "blocked": bool}}   # STRIP mode
  Out (brain -> mod):
      {"action": "mine"|"goto"|"idle", "target": [x,y,z]?, "say": "..."?}
"""

import json
import math
import socket
import sys

HOST = "127.0.0.1"
PORT = 5555
ARRIVE_RADIUS = 2.0
HOME_MESSAGE = "Inventory full, back at base!"
STRIP_DONE_MESSAGE = "Work area cleared!"
LAVA_BLOCKED_MESSAGE = "Stopped - remaining blocks border lava."
LAVA_PAUSE_MESSAGE = "Lava nearby - pausing for safety."


class MiningBrain:
    def __init__(self):
        self.mode = None
        self.target_blocks = set()
        self.home_base = None
        self.box = None
        self.state = "IDLE"
        self.current_target = None
        self._announced_home = False
        self._announced_done = False
        self._announced_hazard = False

    # ---------------------------------------------------------------- commands
    def set_targets(self, blocks, home):
        self.mode = "ore"
        self.target_blocks = set(blocks)
        self.home_base = (home["x"], home["y"], home["z"])
        self.state = "MINING"
        self.current_target = None
        self._reset_announcements()
        print(f"[brain] ORE job. targets={sorted(self.target_blocks)} "
              f"home={tuple(round(c, 1) for c in self.home_base)}", flush=True)

    def set_strip(self, home, box):
        self.mode = "strip"
        self.home_base = (home["x"], home["y"], home["z"])
        self.box = box
        self.state = "MINING"
        self.current_target = None
        self._reset_announcements()
        print(f"[brain] STRIP job. box={box} "
              f"home={tuple(round(c, 1) for c in self.home_base)}", flush=True)

    def stop(self):
        self.mode = None
        self.state = "IDLE"
        self.current_target = None
        print("[brain] Stopped -> IDLE", flush=True)

    def _reset_announcements(self):
        self._announced_home = False
        self._announced_done = False
        self._announced_hazard = False

    # ------------------------------------------------------ per-state decision
    def decide(self, state_msg):
        player = state_msg.get("player", {})
        px = player.get("x", 0.0)
        py = player.get("y", 0.0)
        pz = player.get("z", 0.0)
        inventory_full = bool(state_msg.get("inventory", {}).get("is_inventory_full", False))
        hazard = bool(state_msg.get("hazard", False))

        # ---- State: MINING (active work) -----------------------------------
        if self.state == "MINING":
            # Safety brake: lava beside the player -> pause, auto-resume when clear.
            if hazard:
                if not self._announced_hazard:
                    self._announced_hazard = True
                    print("[brain] Lava adjacent - pausing.", flush=True)
                    return {"action": "idle", "say": LAVA_PAUSE_MESSAGE}
                return {"action": "idle"}
            self._announced_hazard = False

            # Full bags -> head home (either mode).
            if inventory_full:
                self.state = "RETURNING"
                self.current_target = None
                self._announced_home = False
                print("[brain] Inventory full. MINING -> RETURNING", flush=True)
                return self._goto_home_action()

            if self.mode == "strip":
                return self._decide_strip(state_msg)
            return self._decide_ore(px, py, pz, state_msg)

        # ---- State: RETURNING ----------------------------------------------
        if self.state == "RETURNING":
            if self.home_base is None:
                self.state = "IDLE"
                return {"action": "idle"}
            dist = self._dist(px, py, pz, *self.home_base)
            if dist <= ARRIVE_RADIUS:
                self.state = "IDLE"
                if not self._announced_home:
                    self._announced_home = True
                    print(f"[brain] Arrived home ({dist:.1f}m). RETURNING -> IDLE", flush=True)
                    return {"action": "idle", "say": HOME_MESSAGE}
                return {"action": "idle"}
            return self._goto_home_action()

        # ---- State: IDLE ---------------------------------------------------
        return {"action": "idle"}

    # ----------------------------------------------------------- mode handlers
    def _decide_ore(self, px, py, pz, state_msg):
        nearest = self._nearest_target(px, py, pz, state_msg.get("targets", []))
        if nearest is None:
            self.current_target = None
            return {"action": "idle", "note": "no visible target ore nearby"}
        self.current_target = (nearest["x"], nearest["y"], nearest["z"])
        return {"action": "mine", "target": list(self.current_target)}

    def _decide_strip(self, state_msg):
        strip = state_msg.get("strip", {})
        nxt = strip.get("next")
        if not nxt:  # None / missing -> box done, or blocked by lava
            self.state = "IDLE"
            if not self._announced_done:
                self._announced_done = True
                if strip.get("blocked"):
                    print("[brain] Strip stopped: blocks border lava. -> IDLE", flush=True)
                    return {"action": "idle", "say": LAVA_BLOCKED_MESSAGE}
                print("[brain] Work area cleared. -> IDLE", flush=True)
                return {"action": "idle", "say": STRIP_DONE_MESSAGE}
            return {"action": "idle"}
        self.current_target = tuple(nxt)
        return {"action": "mine", "target": list(nxt)}

    # ----------------------------------------------------------------- helpers
    def _goto_home_action(self):
        return {"action": "goto", "target": list(self.home_base)}

    def _nearest_target(self, px, py, pz, candidates):
        matching = [c for c in candidates if c.get("id") in self.target_blocks]
        if not matching:
            return None
        matching.sort(key=lambda c: self._dist(px, py, pz, c["x"], c["y"], c["z"]))
        return matching[0]

    @staticmethod
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


def serve():
    brain = MiningBrain()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[brain] Listening on {HOST}:{PORT} - waiting for the mod...", flush=True)

        while True:
            conn, addr = server.accept()
            print(f"[brain] Mod connected from {addr}", flush=True)
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
            print("[brain] Mod disconnected. Waiting for reconnect...", flush=True)


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("\n[brain] Shutting down.", flush=True)
        sys.exit(0)
