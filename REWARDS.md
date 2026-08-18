# eDNA Seafloor Survey — Complete Reward Reference (Updated)

Every reward/penalty term in `environment.py`'s `step()` function, in the order they're
applied. All of these are added together into a single `reward` value returned each step.

**This version reflects all fixes made since the first draft**: the shaping discount
factor now uses `self.gamma` (passed in at construction, matching the PPO agent's
`gamma=0.99` — no more hardcoded mismatch), the shaping scale is `0.2`, and there's a
new danger-proximity shaping term. Vent contact is also now structurally prevented via
action masking (see `get_action_mask()`), so the vent penalty below exists as a
safety net but should essentially never fire against a masked policy.

| # | Event | Reward | When it fires |
|---|---|---|---|
| 1 | Per-step cost | **−0.04** | Every step, unconditionally — a small "efficiency tax" so idling isn't free. |
| 2 | Movement energy cost | **0.0 direct reward** (drains `self.energy`) | Cardinal move: `1.0` energy. Diagonal: `1.5` energy. Plus a small elevation-dependent adjustment. No direct reward — only matters via running out of energy (#6). |
| 3 | Successful sample | **`concentration × 10.0`** | Standing on an un-sampled hotspot, sampling, with ≥1 canister left. `concentration` is the eDNA field's local value (~0–1), so typically **0 to ~10**. |
| 4 | Re-sampling an already-sampled cell | **−1.0** | Sampling a hotspot you've already sampled. |
| 5 | Sampling elsewhere / 0 canisters left | **0.0** | No-op. |
| 6 | Hit a vent | **−20.0**, episode ends | Should essentially never fire now — action masking (`get_action_mask()`) makes it structurally impossible to choose a move that steps onto a known vent cell. Kept as a safety-net penalty in case of an edge case (e.g. a future change to hazard placement timing). |
| 7 | Killed by a predator | **−20.0**, episode ends | In a predator zone AND the per-step attack probability (default `0.5%`) rolls true. NOT masked — this is deliberate: it's a probabilistic risk, not certain death, so briefly passing through is a legitimate tradeoff for the policy to learn, not something to structurally prevent. |
| 8 | Ran out of energy | **−15.0**, episode ends | `self.energy <= 0`. |
| 9 | Returned home safely | **`0.5 + 2.5 × samples_taken`**, episode ends | Left spawn at some point, now back on it. Kept deliberately low as a flat/base amount — most of the value has to come from `samples_taken`, so a "safe but empty-handed" return isn't competitive with actually sampling. |
| 10 | Timed out (never returned home) | **−20.0**, episode ends | Reached `max_steps` (500) without dying or returning to spawn. |
| 11 | Potential-based **hotspot** shaping | **`self.gamma × new_potential − prev_potential`** | Every step. Dense "getting warmer toward the nearest un-sampled hotspot" signal (or toward spawn, once canisters are exhausted or everything's sampled). `gamma` now correctly matches the PPO agent's actual gamma (`0.99`) — see note above. Scale is `0.2`. |
| 12 | Potential-based **danger** shaping | **`self.gamma × new_danger_potential − prev_danger_potential`** | Every step. Dense "getting warm near a vent" penalty — 0 at ≥6 tiles from the nearest vent cell, ramping to `−0.15 × 6 = −0.9` right at a vent's edge. Only vents count (not predator zones — same reasoning as #7, probabilistic risk isn't something to shape away from entirely). |

## What the two shaping potentials actually measure

Both are functions of the *current state* — what's added to reward each step is the
*change* in these values (that's what "potential-based" means), not a fixed amount.

**Hotspot potential** (`_compute_potential()`):
- Canisters remain, un-sampled hotspot exists: `−(distance to nearest un-sampled hotspot) × 0.2`
- Canisters exhausted, or everything sampled: `−(distance to spawn) × 0.2` — the "goal" the shaping points toward switches to "go home" once sampling is no longer possible.

**Danger potential** (`_compute_danger_potential()`):
- `−0.15 × max(0, 6.0 − distance to nearest vent cell)` — zero once you're 6+ tiles from every vent, growing (more negative) as you approach one.
- Returns `0.0` immediately if the episode has no vents.

## Action masking (structural, not reward-based)

`get_action_mask()` returns a length-9 boolean array (`True` = allowed). Any of the 8
movement actions that would step directly onto a currently-visible vent cell is marked
`False` before the policy ever samples an action — logits for masked actions are set to
`-1e8` before the softmax, so their probability is ~0. This is enforced identically
during both action selection *and* the PPO update (the mask is stored per-step in the
buffer and replayed), so training stays consistent.

This is why the vent penalty (#6) is effectively a dead code path now — it's not that
the agent has learned to avoid vents, it's that it structurally cannot choose to walk
into one. Predator zones are deliberately left unmasked (see #7 and #12's notes).

## Episode-ending outcomes at a glance

| End reason | Reward impact | Notes |
|---|---|---|
| `returned_to_spawn` | `+0.5` to `+0.5 + 2.5×n` | The only "success" ending |
| `vent` | `−20.0` | Should be ~never observed now (masked) |
| `predator_attack` | `−20.0` | Genuine, unmasked risk |
| `out_of_energy` | `−15.0` | Ran the tank dry |
| `max_steps` (timeout) | `−20.0` | Took too long, never made it home |

## Sanity checks worth re-running after any reward-function change

- Does a **do-nothing round trip** score meaningfully *less* than a **real sampling run**?
- Does **lingering near but never reaching a hotspot until timeout** score *worse* than actually reaching it? (This was broken once — a hardcoded shaping gamma that drifted from the agent's real gamma. Always double check `self.gamma` is the single source of truth, never a second hardcoded number.)
- If vent deaths ever start appearing again in the logs despite masking — that's a red flag the mask isn't being applied/stored/replayed consistently somewhere, not that the policy "got unlucky."
