"""
Gymnasium environment for the eDNA seafloor survey task.

Ties together everything built so far:
    - terrain_v2.py    -> elevation generation + impassable mask + elevation step cost
    - placement.py     -> spawn + eDNA hotspot placement + concentration field
    - hazards.py       -> vent + predator zone placement, attack rolls

Action space: Discrete(9)
    0-7 -> movement in 8 directions (N, NE, E, SE, S, SW, W, NW)
    8   -> sample (only has an effect if standing on an un-sampled hotspot cell)

Observation space: Dict
    "maps":    (5, grid_size, grid_size) float32 -> elevation, eDNA field,
               vent mask, predator mask, sampled mask
    "scalars": (6,) float32 -> agent_row, agent_col, energy, canisters,
               spawn_row, spawn_col (all normalized to roughly [0, 1])

Episode ends when any of:
    - energy <= 0                          (ran out of power)
    - agent enters a vent cell             (certain damage)
    - a predator attack roll succeeds      (0.5% chance per step spent in a predator zone)
    - agent returns to spawn after having left it  (successful mission completion)
    - max_steps reached                    (safety cutoff, shouldn't normally trigger
                                             given the energy budget, but prevents any
                                             pathological infinite-loop edge case)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from terrain_generator import generate_terrain, compute_impassable_terrain, elevation_step_cost_flat
from object_placement import place_spawn, place_edna_hotspots, generate_edna_field
from hazards import place_vents, place_predator_zones, build_hazard_mask, hazard_type_at, roll_predator_attack


# 8 movement directions as (d_row, d_col), index matches action id 0-7
DIRECTIONS = [
    (-1, 0),   # 0: N
    (-1, 1),   # 1: NE
    (0, 1),    # 2: E
    (1, 1),    # 3: SE
    (1, 0),    # 4: S
    (1, -1),   # 5: SW
    (0, -1),   # 6: W
    (-1, -1),  # 7: NW
]
SAMPLE_ACTION = 8

CARDINAL_COST = 1.0
DIAGONAL_COST = 1.5
ELEVATION_FLAT_COST = 0.1


class SeafloorEDNAEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 50,
        max_energy: float = 300.0,
        n_canisters: int = 3,
        n_hotspots_range: tuple[int, int] = (3, 5),
        n_vents_max: int = 2,
        predator_attack_prob: float = 0.005,
        max_steps: int = 500,
        render_mode: str | None = None,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_energy = max_energy
        self.n_canisters_start = n_canisters
        self.n_hotspots_range = n_hotspots_range
        self.n_vents_max = n_vents_max
        self.predator_attack_prob = predator_attack_prob
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.gamma = gamma
        self.max_hotspot_dist = None

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "maps": spaces.Box(low=0.0, high=1.0, shape=(5, grid_size, grid_size), dtype=np.float32),
            "scalars": spaces.Box(low=-1.0, high=2.0, shape=(6,), dtype=np.float32),
        })

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #

    def reset(self, 
              seed: int | None = None, 
              options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._generate_episode()

        self.agent_pos = self.spawn
        self.energy = self.max_energy
        self.canisters = self.n_canisters_start
        self.sampled_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.has_left_spawn = False
        self.step_count = 0
        self.n_samples_taken = 0
        self._prev_potential = self._compute_potential()
        self._prev_danger_potential = self._compute_danger_potential()

        return self._get_obs(), self._get_info()
    
    def _compute_potential(self) -> float:
        """
        Potential function for reward shaping: negative distance to the nearest
        *reachable-for-sampling* hotspot, or to spawn once no more sampling is
        possible. Used as a potential-based shaping term (gamma * phi(s') -
        phi(s), added to reward each step) -- gives the agent dense signal
        toward its current goal without changing the optimal policy (Ng et
        al. 1999). Makes the existing sparse sampling/return-home reward much
        easier to discover via gradient signal.

        Canisters are limited (fewer than the max possible hotspot count), so
        "nearest unsampled hotspot" stops being a valid goal once canisters
        run out -- sampling there is physically impossible. Without this
        check, the shaping term kept pulling the agent toward hotspots it
        could never sample again instead of toward heading home, actively
        working against the return-to-spawn objective.
        """
        if self.canisters > 0:
            unsampled = [h for h in self.hotspots if self.sampled_mask[h] == 0]
        else:
            unsampled = []

        if unsampled:
            dists = [np.hypot(self.agent_pos[0] - h[0], self.agent_pos[1] - h[1]) for h in unsampled]
            return -min(dists) * 0.2  # scaled down so shaping doesn't dominate other rewards

        # no more sampling possible (out of canisters, or all hotspots sampled)
        # -> shape toward spawn instead, so the agent still gets dense signal
        # for the "get home safely" phase of the task
        dist_to_spawn = np.hypot(self.agent_pos[0] - self.spawn[0], self.agent_pos[1] - self.spawn[1])
        return -dist_to_spawn * 0.2

    def get_action_mask(self) -> np.ndarray:
        """Bool array, len 9 (True=allowed). Blocks only the 8 movement actions
        that would step directly onto a known-lethal vent cell -- vents are
        fully visible and static, so there's no reason to let the policy ever
        pick a move that's certain death; this makes that structurally
        impossible instead of relying on it being learned. Predator zones are
        NOT masked (probabilistic risk, a legitimate tradeoff). Sample action
        (8) is always allowed."""
        mask = np.ones(9, dtype=bool)
        r, c = self.agent_pos
        for a, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.vent_mask[nr, nc]:
                mask[a] = False
        return mask

    def _compute_danger_potential(self, danger_radius: float = 6.0, scale: float = 0.15) -> float:
        """
        Potential function for VENT-AVOIDANCE shaping: a dense, "getting warm
        near danger" penalty that ramps up as the agent gets closer to a vent,
        instead of danger only being felt at the moment of death (-20, once,
        rare). Same potential-based-shaping structure as _compute_potential()
        (telescopes cleanly given matching gamma), just pushing AWAY from
        something instead of TOWARD something.

        0 penalty at danger_radius tiles or farther from the nearest vent
        cell; ramps linearly to -scale*danger_radius right at a vent's edge.
        Predator zones are deliberately excluded -- they're a probabilistic
        risk, not certain death, so brief exposure is a legitimate tradeoff,
        not something to shape the agent away from entirely.
        """
        if not np.any(self.vent_mask):
            return 0.0
        vent_cells = np.argwhere(self.vent_mask)
        dists = np.hypot(vent_cells[:, 0] - self.agent_pos[0], vent_cells[:, 1] - self.agent_pos[1])
        nearest = dists.min()
        return -scale * max(0.0, danger_radius - nearest)

    def _generate_episode(self, 
                          max_regen_attempts: int = 10):
        """Runs the full generation pipeline, retrying if spawn/hotspot placement fails."""
        for _ in range(max_regen_attempts):
            elevation = generate_terrain(grid_size=self.grid_size, seed=int(self._rng.integers(0, 1_000_000)))
            impassable = compute_impassable_terrain(elevation)

            spawn = place_spawn(impassable, rng=self._rng)
            if spawn is None:
                continue

            n_hotspots = int(self._rng.integers(*self.n_hotspots_range))
            hotspots = place_edna_hotspots(impassable, spawn, n_hotspots=n_hotspots, rng=self._rng)
            if hotspots is None or len(hotspots) == 0:
                continue

            edna_field = generate_edna_field(self.grid_size, hotspots, rng=self._rng)
            vents = place_vents(hotspots, impassable, n_vents=min(self.n_vents_max, len(hotspots)), rng=self._rng)
            predators = place_predator_zones(impassable, spawn, hotspots, rng=self._rng)

            self.elevation = elevation
            self.impassable = impassable
            self.spawn = spawn
            self.hotspots = hotspots
            self.edna_field = edna_field
            self.vents = vents
            self.predators = predators
            self.vent_mask = build_hazard_mask(self.grid_size, vents) if vents else \
                np.zeros((self.grid_size, self.grid_size), dtype=bool)
            self.predator_mask = build_hazard_mask(self.grid_size, predators) if predators else \
                np.zeros((self.grid_size, self.grid_size), dtype=bool)
            self.all_hazards = vents + predators
            return

        raise RuntimeError("Failed to generate a valid episode after max_regen_attempts -- "
                            "check placement constraints, they may be too strict for this grid size.")

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, 
             action: int):
        self.step_count += 1
        reward = -0.04  # small per-step cost, encourages efficiency
        terminated = False
        truncated = False
        info = {}

        if action in range(8):
            reward += self._handle_movement(action)
        elif action == SAMPLE_ACTION:
            reward += self._handle_sample()
        else:
            raise ValueError(f"Invalid action {action}, expected 0-8")

        # -- hazard checks happen AFTER movement resolves, based on new position --
        hazard = hazard_type_at(self.agent_pos, self.all_hazards)
        if hazard == "vent":
            reward -= 20.0
            terminated = True
            info["end_reason"] = "vent"
        elif hazard == "predator":
            if roll_predator_attack(self.predator_attack_prob, rng=self._rng):
                reward -= 20.0
                terminated = True
                info["end_reason"] = "predator_attack"

        # -- energy depletion --
        if not terminated and self.energy <= 0:
            terminated = True
            reward -= 15.0
            info["end_reason"] = "out_of_energy"

        # -- return to spawn (success condition) --
        if not terminated and self.has_left_spawn and self.agent_pos == self.spawn:
            terminated = True
            # small flat bonus for a safe return (worth having, since dying already
            # costs -15/-20, so "come home safely" deserves *some* credit on its
            # own) -- but kept low on purpose. A large flat bonus here made a
            # trivial one-tile-out-and-back round trip a competitive terminal
            # strategy versus actually risking a real sampling run, since it
            # collected most of the reward for near-zero cost/risk. Sampling
            # reward now has to carry almost all of the incentive to explore.
            completion_bonus = 0.5 + 2.5 * self.n_samples_taken
            reward += completion_bonus
            info["end_reason"] = "returned_to_spawn"
            info["samples_collected"] = self.n_samples_taken

# -- safety cutoff --
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            reward -= 20.0
            info["end_reason"] = "max_steps"

        # -- potential-based distance shaping (dense signal toward nearest unsampled hotspot) --
        new_potential = self._compute_potential()
        reward += self.gamma * new_potential - self._prev_potential
        self._prev_potential = new_potential

        # -- potential-based DANGER shaping (dense signal to steer away from vents) --
        new_danger_potential = self._compute_danger_potential()
        reward += self.gamma * new_danger_potential - self._prev_danger_potential
        self._prev_danger_potential = new_danger_potential

        full_info = self._get_info()
        full_info.update(info)
        return self._get_obs(), reward, terminated, truncated, full_info

    def _handle_movement(self, 
                         action: int) -> float:
        d_row, d_col = DIRECTIONS[action]
        new_row = self.agent_pos[0] + d_row
        new_col = self.agent_pos[1] + d_col

        # out of bounds -> blocked, no-op, no extra cost
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return 0.0

        new_pos = (new_row, new_col)

        # impassable terrain -> blocked, no-op, no extra cost (wasted a step, not punished further)
        if self.impassable[new_pos]:
            return 0.0

        is_diagonal = (d_row != 0 and d_col != 0)
        move_cost = DIAGONAL_COST if is_diagonal else CARDINAL_COST
        elev_cost = elevation_step_cost_flat(self.elevation, self.agent_pos, new_pos, flat_cost=ELEVATION_FLAT_COST)
        total_cost = move_cost + elev_cost

        self.energy -= total_cost
        self.agent_pos = new_pos
        if self.agent_pos != self.spawn:
            self.has_left_spawn = True

        return 0.0  # movement itself has no direct reward, only the energy cost (applied above)

    def _handle_sample(self) -> float:
        r, c = self.agent_pos
        # not standing on a hotspot cell -> sampling does nothing
        is_hotspot_cell = any((r, c) == h for h in self.hotspots)
        if not is_hotspot_cell or self.canisters <= 0:
            return 0.0

        if self.sampled_mask[r, c] > 0:
            return -1.0  # already sampled this exact cell 

        concentration = float(self.edna_field[r, c])
        self.sampled_mask[r, c] = 1.0
        self.canisters -= 1
        self.n_samples_taken += 1
        return concentration * 10.0


    def _get_obs(self):
        maps = np.stack([
            self.elevation,
            self.edna_field,
            self.vent_mask.astype(np.float32),
            self.predator_mask.astype(np.float32),
            self.sampled_mask,
        ], axis=0).astype(np.float32)

        scalars = np.array([
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.energy / self.max_energy,
            self.canisters / self.n_canisters_start,
            self.spawn[0] / self.grid_size,
            self.spawn[1] / self.grid_size,
        ], dtype=np.float32)

        return {"maps": maps, "scalars": scalars}

    def _get_info(self):
        return {
            "energy": self.energy,
            "canisters": self.canisters,
            "samples_taken": self.n_samples_taken,
            "agent_pos": self.agent_pos,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        # lightweight placeholder -- full matplotlib rendering comes later as a
        # separate GIF-export utility, not baked into the env's hot path
        return None