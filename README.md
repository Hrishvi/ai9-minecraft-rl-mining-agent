# ai9-minecraft-rl-mining-agent

**Experimental Minecraft AI mining agent** combining a Fabric 1.21.11 companion mod (Java) with a Python decision/state-machine "brain" — a research/dev playground for agent architecture, proximity-based target selection, and simple return-home behavior loops. Not a finished product.

> ⚠️ **Status: dev beta / research sandbox.** This is a personal experiment, not a stable release. Things break, change shape without notice, and aren't guaranteed to work on your setup.

---

## What this is

A hybrid client-side agent for Minecraft (Fabric, 1.21.11) split into two halves:

- **Java companion mod** — runs inside the game as a Fabric client mod. Reads world/player/inventory state, executes movement and block-breaking primitives, and reports state out over local HTTP every couple of ticks.
- **Python brain** (`Minecraft_RL/`) — runs outside the game as a lightweight local HTTP server. Owns the decision-making: target prioritization (closest-ore selection) and a small state machine (mining vs. returning to a base location).

The split is intentional: the mod is "hands and eyes," Python is "the brain." This keeps the Java side dumb and stable, and lets the interesting agent-logic experimentation happen in Python where it's faster to iterate.

This project is being used to explore:
- Local low-latency game-state ↔ Python communication patterns
- Simple rule-based / state-machine agent decision logic as a stepping stone toward RL
- Multi-target prioritization and pathing heuristics in a 3D voxel environment

It is **not** a combat hack, not affiliated with Mojang/Microsoft, and not intended for use on servers whose rules prohibit automation — see [Disclaimer](#disclaimer--legal).

---

## Project Status

This repository is an active, unstable, work-in-progress experiment.

- ❌ No guarantee any given commit builds or runs correctly
- ❌ No guarantee of backward compatibility between commits
- ❌ No support, no SLAs, no promises
- ✅ Issues and PRs are welcome, but treat everything here as "use at your own risk"
- ✅ Built and tested against Minecraft **1.21.11** with **Fabric Loader** + **Fabric API**

If you're looking for a polished, ready-to-use Minecraft bot, this isn't it (yet, maybe never). If you're poking around game-agent architecture, local game-state APIs, or just curious how a Fabric mod talks to a Python process, you're in the right place.

---

## How it works (high level)

1. The Fabric mod runs client-side and watches the player's world: nearby blocks, inventory fill state, position.
2. Every few ticks, it POSTs a JSON snapshot of that state to a local Python HTTP server (`Minecraft_RL/`).
3. Python evaluates the snapshot against its current state (mining or returning home), picks a target or destination, and responds with the next action.
4. The mod executes that action in-game (move toward a point, break a block, stop) and the loop repeats.

No cloud calls, no telemetry — everything runs locally between the game client and a Python process on the same machine.

---

## Requirements

- Minecraft **1.21.11**
- [Fabric Loader](https://fabricmc.net/) + Fabric API
- Java 21 (required by Minecraft 1.21.x / Fabric)
- Python 3.9+ (standard library only)

---

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/ai9-minecraft-rl-mining-agent.git
   cd ai9-minecraft-rl-mining-agent
   ```
2. Build the mod:
   ```bash
   ./gradlew build
   ```
   Output jar will be under `build/libs/`. Drop it into your Fabric `mods/` folder.
3. Start the Python brain **before** launching Minecraft:
   ```bash
   cd Minecraft_RL
   python3 mining_agent.py
   ```
4. Launch Minecraft with Fabric, load into a world, and use the in-game chat command to start the agent.

---

## Roadmap / Ideas (unordered, unscheduled)

- [ ] Chest-deposit automation when returning home full
- [ ] Pathfinding upgrade (obstacle-aware navigation, e.g. Baritone integration)
- [ ] Actual RL-based decision policy instead of hand-written state machine
- [ ] Configurable scan radius / performance tuning
- [ ] Logging/replay tooling for agent behavior analysis

None of these are commitments — this list exists to track ideas, not promises.

---

## Disclaimer & Legal

- This project is an independent, fan-made experiment and is **not affiliated with, endorsed by, or associated with Mojang Studios or Microsoft**.
- Minecraft is a trademark of Mojang Studios.
- Automating gameplay (movement, mining, combat assistance, etc.) **violates the rules of most public/community Minecraft servers** and may violate the Minecraft EULA / server ToS depending on context. This project is intended for **single-player, private, or explicitly automation-permitting servers only**. You are responsible for how you use it.
- Provided **as-is**, with no warranty of any kind — see [LICENSE](LICENSE).

---

## Contributing

This is a personal research repo, but issues, forks, and PRs are welcome if you're interested in the same problem space. No formal process — open an issue if you want to discuss a change before sending a PR.

---

## License

See [LICENSE](LICENSE).
