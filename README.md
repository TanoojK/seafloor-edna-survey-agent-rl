# Underwater Robot RL: Seafloor eDNA Survey Agent

A reinforcement learning agent, trained from scratch with PPO, that controls a simulated underwater robot searching a seafloor for eDNA (environmental DNA) samples. eDNA is a real marine-biology technique for detecting which species are present without ever seeing the organism.

The robot has to find hotspots under an energy budget, stay clear of lethal vents and predator zones, and **make it back to its starting point with what it collected**. That makes it a round-trip planning problem, not just a search problem.

## Results

Final checkpoint (500k steps per environment, 8 environments, 4M environment steps in total, about 2 hours on a 12-core CPU). Evaluated on 50 fixed seeds per row, 3 canisters, no forced return starts, **sampling actions from the policy**:

| Hotspot distance limit | Returned to spawn | Out of energy | Hazard deaths | Samples (of 3) | Avg reward |
|---|---|---|---|---|---|
| 13 tiles | 41 / 50 | 6 | 3 | 1.88 | +18.9 |
| 20 tiles | 37 / 50 | 10 | 3 | 1.84 | +17.1 |
| 30 tiles (training maximum) | 41 / 50 | 7 | 2 | 1.80 | +19.8 |
| unlimited (any distance ≥ 8) | 36 / 50 | 12 | 2 | 1.76 | +17.9 |

Please read these with the following caveats:

- **One training run, one seed.** I have no estimate of run-to-run variation. With 50 episodes per row, a rate around 75% carries roughly ±12 percentage points of sampling noise, so the differences between rows are not meaningful.
- **The policy is stochastic.** Using the most likely action instead (`--deterministic`) is much worse: 14-22% returned and about 0.3 samples. The agent still relies on randomness to press SAMPLE at the right moment (a hotspot must be sampled from its exact cell).
- **The environment tells the robot when to go home** (see "Design notes" below). It still has to navigate back and avoid hazards, but the timing of the decision is partly handed over.
- It uses about 1.8 of 3 canisters on average. I haven't checked whether leaving canisters unused is the best trade-off or just a cautious one.

## How it works

**Environment** (`scripts/environment.py`): a procedurally generated 50×50 grid with Perlin-noise terrain, 3-5 eDNA hotspots per episode (Gaussian concentration fields), thermal vents (instant death, placed near hotspots to create risk/reward), predator zones (0.5% attack chance per step inside), 3 sample canisters, and an energy budget of 300 that has to cover the full round trip. The agent chooses from 9 actions: 8 moves and SAMPLE.

**Observation** (agent-centered):

- a 49×49 window centered on the robot (zero-padded at the map edge) with 6 channels: elevation, eDNA concentration, vent mask, predator mask, sampled-hotspot mask, and a **home beacon** (a wide Gaussian around spawn)
- 5 scalars: energy fraction, canister fraction, spawn offset (row, column), and an energy margin `(energy − 1.2 × distance_home) / max_energy`
- the home beacon is only shown when going home makes sense: all canisters used, or at least one sample taken and `energy < 1.5 × distance_home + 20`. Otherwise the channel is zero.

**Algorithm**: Proximal Policy Optimization written from scratch in PyTorch (no RL library): clipped surrogate objective, GAE (λ = 0.95), separate actor and critic networks (each a 3-layer CNN plus a small scalar embedding), entropy regularisation (0.01 → 0.001), learning rate annealed 3e-4 → 3e-5, gradient clipping applied separately to actor and critic. Rollouts come from 8 parallel worker processes (spawn-based, so it runs on Windows).

**Safety**: moves onto a visible vent are blocked by an action mask. The mask is stored with every transition and replayed during the update. Predator zones are not masked on purpose; they are a risk to trade off, not certain death.

**Training distribution**:

- Hotspot distance follows a curriculum: each episode draws its maximum distance from spawn uniformly between 10 and a ceiling that grows over the whole run (quadratically) to 30 tiles.
- 30% of episodes start in the **return phase** (agent already out on the map, no canisters, partly used tank). Without this, the return trip was almost never practised and training stalled at about 5% returns.
- Canister count is random (1-3) in training.

Full reward details are in [`REWARDS.md`](./REWARDS.md).

## What went wrong along the way

The interesting part of this project was diagnosing why training kept failing and fixing root causes instead of tuning hyperparameters.

- **Reward hacking.** The agent earned most of the "mission complete" reward by stepping out of spawn and straight back in. Fixed by making the return pay mostly through samples and requiring at least one sample.
- **A silent shaping bug.** The shaping discount was hardcoded separately from the agent's gamma; once they drifted, the agent could bank reward by loitering near a target. Fixed with a single source for gamma.
- **Policy collapse.** Entropy either stayed near maximum forever or collapsed onto a useless fixed behaviour, depending on the schedule.
- **Catastrophic forgetting under a one-way curriculum.** Fixed by drawing each episode's difficulty from a range instead of a single ramping value.
- **The agent couldn't navigate (the biggest one).** Every run plateaued at random-level sampling. The cause was the observation: the robot's own position was only two scalars (later a single pixel in its own channel), and a CNN followed by a fully connected layer is very slow to learn "where is the blob relative to me" from that. A scripted expert showed the task was easy. In a behaviour-cloning test with the same network and data, the agent-centered view reached about 72% accuracy and 0.76 samples per episode after 60 epochs, against about 31% and 0.0 for the global view. Switching training to an agent-centered window fixed the plateau.
- **Parking.** Once it could sample, the agent learned to stop and wait out the timeout instead of coming home, because idling cost almost nothing. Fixed by charging a full move's energy for any step that doesn't move the agent.
- **Never learning to return.** The return trip was rare in training, so it wasn't learned (0 of 40 successes when I placed the agent on a hotspot with no canisters). Fixed by starting 30% of episodes in the return phase, plus the conditional home beacon. This took returns from about 5% to about 80%.
- **A misleading evaluation.** An early evaluation script didn't apply the vent mask and showed 35% vent deaths; the training code was fine.

## Limitations

- Single run and seed; the numbers above are not averaged over training runs.
- The deterministic policy performs poorly (see Results).
- The environment decides when to show the home beacon.
- A small number of vent deaths (roughly 0-2% of logged training episodes) still occur despite the mask. I haven't found out why.
- 2D grid world only; no physics or currents. A 3D version is a possible next step.
- Hotspots farther than the 30-tile training maximum were only evaluated, not trained on.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pinned `torch` is enough for CPU training, which is what this project uses (the model is small and the time goes into the environment).

## Usage

Run everything from the `scripts/` folder.

Train from scratch (about 2 hours on 12 cores):
```bash
cd scripts
python train.py
python train.py --fast-test     # shorter proxy run
```

Evaluate a checkpoint (samples actions from the policy; optional last argument is the hotspot distance limit):
```bash
python eval_final.py checkpoints/ppo_step500000_final.pt 30
```

Watch one episode (writes a trajectory image and a GIF to `rollout_viz/`):
```bash
python visualize_rollout.py --checkpoint checkpoints/ppo_step500000_final.pt --seed 7
```

Run the scripted baseline that routes around obstacles and vents:
```bash
python test_survivability.py
```

Checkpoints are not stored in the repository.

## Example

Evaluated with 3 canisters; chosen from 20 seeds tried because the agent collects all 3 samples and returns home. A typical episode collects about 1.8

![Example trajectory](Images/example_trajectory.png)

![Example run](Images/example_run.gif)

## Repository layout

- `scripts/environment.py`: the environment, reward and observation
- `scripts/terrain_generator.py`, `object_placement.py`, `hazards.py`: procedural generation
- `scripts/networks.py`, `ppo.py`, `buffer.py`: actor-critic network and PPO
- `scripts/vec_env.py`: parallel environments
- `scripts/train.py`: training loop, curriculum and schedules
- `scripts/eval_final.py`, `visualize_rollout.py`, `visualize_environment.py`, `test_survivability.py`: evaluation and diagnostics
- `REWARDS.md`: reward and energy reference
