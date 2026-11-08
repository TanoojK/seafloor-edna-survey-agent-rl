# eDNA Seafloor Survey: Reward and Energy Reference

Every reward and penalty term in `scripts/environment.py`'s `step()`, in the order they are applied. All of them are summed into the single `reward` value returned each step.

This reflects the version that produced the results in the README. Things that changed from the first draft:

- the per-step energy drain is gone; instead, **any step that doesn't move the agent costs a full move's worth of energy** (see "Energy" below)
- the return bonus is `5 + 2.5 × samples` (was `0.5 + 2.5 × samples`)
- the "go home" shaping target can switch on early when energy runs low, not only when canisters run out
- 30% of training episodes start in the return phase (they are excluded from the logged statistics)
- SAMPLE works within 1 tile of an unsampled hotspot (it used to require the exact cell), and the action mask now also blocks moves that do nothing and SAMPLE where it cannot succeed

## Reward terms

| # | Event | Reward | When it fires |
|---|---|---|---|
| 1 | Per-step cost | **-0.04** | Every step. |
| 2 | Successful sample | **`concentration × 10.0`** | Pressing SAMPLE within 1 tile (8-neighbourhood or the same cell) of an unsampled hotspot, with at least one canister left. `concentration` is the eDNA field value at the robot's own cell, normalised so the strongest hotspot is 1.0, so it is slightly lower one tile off the peak. The closest unsampled hotspot in range is the one marked as sampled. |
| 3 | Re-sampling a sampled hotspot | **-1.0** | SAMPLE with only already-sampled hotspots in range. The mask normally prevents this (see below), so it only appears through the all-actions-allowed fallback. |
| 4 | SAMPLE anywhere else, or with 0 canisters | **0.0** | No reward. Also normally masked out. A non-moving step still costs energy (see below). |
| 5 | Hit a vent | **-20.0**, episode ends | Moving onto a vent cell. The action mask blocks this (see below), so it should not happen. A small number of vent endings (roughly 0-2% of logged training episodes) still appear; I haven't tracked down why. |
| 6 | Predator attack | **-20.0**, episode ends | Inside a predator zone and the per-step attack roll (0.5%) comes up. Not masked on purpose: it is a probabilistic risk the agent can trade off. |
| 7 | Out of energy | **-15.0**, episode ends | `energy <= 0`. |
| 8 | Returned to spawn | **`5 + 2.5 × samples_taken`**, episode ends | Back on the spawn cell after leaving it, with at least one sample taken. (Episodes that start in the return phase don't need a sample; see below.) |
| 9 | Timeout | **-20.0**, episode ends | Reached `max_steps` (500) without another ending. |
| 10 | Hotspot / home shaping | **`gamma × Φ(s') − Φ(s)`** | Every step, with `gamma = 0.99` (the same as the agent's). |
| 11 | Danger shaping | **`gamma × D(s') − D(s)`** | Every step. |

Both shaping potentials are set to 0 at terminal and truncated steps, so the shaping sums to a constant per episode and does not change which policy is optimal.

## Energy

The tank starts at `max_energy = 300`.

- **Moving:** cardinal step 1.0, diagonal step 1.5, plus a small elevation-based term (flat component 0.1). Measured on shortest safe paths, this came to about 1.1 energy per tile of straight-line distance.
- **Not moving:** any step that leaves the agent where it is costs **1.0 energy**. That covers SAMPLE (including a successful one), bumping into impassable terrain and walking off the edge. The mask now removes the pointless ones, but the cost still applies to a successful SAMPLE.
- There is no other drain. The earlier version used a small drain on every step (0.1), which didn't stop the agent from parking and waiting for the timeout. Charging for idle steps closed that gap: an agent that stops moving now runs out of energy before the 500-step limit.

## What the shaping potentials measure

What is added to the reward each step is the *change* in these values, not a fixed amount.

**Hotspot / home potential Φ:**

- Search mode: `-0.2 × (distance to the nearest unsampled hotspot)`.
- Return mode: `-0.2 × (distance to spawn)`.
- Return mode switches on when **canisters are empty**, or **after at least one sample when `energy < 1.5 × (distance to spawn) + 20`**.

**Danger potential D:** `-0.15 × max(0, 6.0 − distance to nearest vent cell)`. It is 0 beyond 6 tiles from every vent. Only vents count, not predator zones.

## Action masking

`get_action_mask()` returns a length-9 boolean array. An action is `False` if:

- it is a move onto a vent cell
- it is a move off the map or into impassable terrain (a no-op that would only cost energy)
- it is SAMPLE and the robot has no canister, or no unsampled hotspot is within 1 tile (`SAMPLE_RADIUS`)

If every action would be masked, all actions are allowed (a safeguard for a boxed-in robot). Logits for masked actions are set to `-1e8` before the softmax, and the mask is stored per step and replayed identically during the PPO update.

Why the extra masking: without it, a greedy policy could pick a move into a wall, or SAMPLE away from any hotspot, and repeat it forever, because the state never changed. The sampled policy escaped by chance. Masking those actions took the greedy policy from about 58% returns to about 92% on a 50-episode check, with no retraining. Offering SAMPLE only where it can succeed is a simplification, since a real sampler would not be told when it is in range.

Evaluation code must pass the mask too. The policy never trains the logits of masked actions, so an evaluation without the mask walks into vents (this caused a spurious 35% vent rate in an early evaluation script).

## Episode endings

| End reason | Reward | Notes |
|---|---|---|
| `returned_to_spawn` | `+7.5` to `+12.5` | The only success ending: `5 + 2.5 × samples`, so +7.5 with 1 sample and +12.5 with 3. (Return-phase starts get the flat +5.) |
| `out_of_energy` | `-15` | Ran the tank dry. |
| `predator_attack` | `-20` | Unmasked risk. |
| `vent` | `-20` | Rare; masked. |
| `max_steps` | `-20` | Rare now that idling costs energy. |

## Training-only details

- **Return-phase starts.** With probability `RETURN_START_PROB = 0.3`, an episode starts with the agent already placed 6-35 tiles from spawn, 0 canisters, a partly used tank (at least `1.5 × distance + 20`), and the two nearest hotspots marked as sampled. Returning home ends these episodes with the `5` base bonus and no sample requirement. They exist because the return trip was almost never practised otherwise.
- **Canister count.** Each episode starts with 1, 2 or 3 canisters, chosen at random (evaluation fixes it at 3).
- **Logging.** Episodes that started in the return phase are left out of the logged reward, sample and end-reason statistics, so those numbers describe normal episodes.

## Sanity checks to re-run after any reward change

- A do-nothing round trip should score much less than a real sampling run (the return rule requires a sample for this reason).
- Parking should end in `out_of_energy` well before `max_steps`. If `max_steps` shows up in the logs again, idling is too cheap.
- If vent endings climb in training logs despite masking, the mask is not being applied or replayed consistently somewhere.
- Keep `gamma` in the environment and in the agent the same value, from one source.
- With the mask in place, a greedy policy should not be able to repeat a no-op forever. If greedy evaluation episodes end in `out_of_energy` with the robot on a single cell, the mask is missing an action type.