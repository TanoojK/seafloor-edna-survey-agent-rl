"""
Hazard system for the environment.

Two hazard types:

    - "Thermal vents": placed near a random subset of eDNA hotspots (risk-reward tension --
      high-value sampling areas are sometimes near genuine danger). Stepping onto
      a vent cell ends episode immediately.

    - "Predator zones": placed independently of hotspots, in small radius zones.
      Each step spent standing in a predator zone has a small (default 0.5%)
      chance of triggering an attack -> episode ends immediately. Risk compounds
      the longer the agent lingers, but a quick pass-through is comparatively safe.
"""

import numpy as np

DEFAULT_PREDATOR_ATTACK_PROB = 0.005  # 0.5% per step spent inside a predator zone

def _generate_blob_mask(
    grid_size: int,
    center: tuple[int, int],
    base_radius: float,
    irregularity: float = 0.4,
    n_harmonics: int = 3,
    rng=None,
) -> np.ndarray: #Generates an irregular blob-shaped boolean mask 
    rng = rng or np.random.default_rng()
    cr, cc = center

    # random harmonic coefficients: amplitude shrinks with harmonic order, random phase each
    amplitudes = [irregularity * base_radius / (k + 1) for k in range(1, n_harmonics + 1)]
    phases = rng.uniform(0, 2 * np.pi, size=n_harmonics)

    rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    dr = rows - cr
    dc = cols - cc
    dist = np.hypot(dr, dc)
    theta = np.arctan2(dr, dc)

    boundary_radius = np.full_like(theta, base_radius, dtype=np.float64)
    for k, (amp, phase) in enumerate(zip(amplitudes, phases), start=1):
        boundary_radius += amp * np.sin(k * theta + phase)

    boundary_radius = np.clip(boundary_radius, base_radius * 0.4, None)  # avoid degenerate near-zero spots
    return dist <= boundary_radius


def place_vents(
    hotspots: list[tuple[int, int]],
    impassable_mask: np.ndarray,
    n_vents: int,
    offset_range: tuple[float, float] = (5.0, 8.0),
    radius: float = 1.5,
    min_hotspot_clearance: float = 2.5,
    max_attempts: int = 100,
    rng=None,
) -> list[dict]:
    """Places vents near a random subset of hotspots (offset by a random distance).

    `min_hotspot_clearance` enforces a hard safety buffer: the vent's irregular
    blob mask is never allowed to reach within this many tiles of the hotspot
    cell itself. Without this, the offset distance and the blob's random
    irregularity (which can push its radius well past its nominal `radius`)
    would sometimes overlap the hotspot's only sane approach path, making
    high-value hotspots effectively unreachable without near-certain death --
    that's "danger nearby", not "danger you can route around".
    """
    rng = rng or np.random.default_rng()
    grid_size = impassable_mask.shape[0]

    if n_vents > len(hotspots):
        n_vents = len(hotspots)
    chosen_hotspots = rng.choice(len(hotspots), size=n_vents, replace=False) if n_vents > 0 else []

    vents = []
    for idx in chosen_hotspots:
        hr, hc = hotspots[idx]
        placed = False
        for _ in range(max_attempts):
            dist = rng.uniform(*offset_range)
            angle = rng.uniform(0, 2 * np.pi)
            vr = int(round(hr + dist * np.sin(angle)))
            vc = int(round(hc + dist * np.cos(angle)))

            if not (0 <= vr < grid_size and 0 <= vc < grid_size):
                continue
            if impassable_mask[vr, vc]:
                continue

            mask = _generate_blob_mask(grid_size, (vr, vc), radius, irregularity=0.4, rng=rng)

            # hard safety buffer -- reject if the blob reaches too close to the hotspot
            rows, cols = np.nonzero(mask)
            if len(rows) > 0:
                dists_to_hotspot = np.hypot(rows - hr, cols - hc)
                if dists_to_hotspot.min() < min_hotspot_clearance:
                    continue

            vents.append({"center": (vr, vc), "radius": radius, "type": "vent", "mask": mask})
            placed = True
            break

        if not placed:
            continue  # skip this vent, don't hold up the whole generation pipeline

    return vents


def place_predator_zones(
    impassable_mask: np.ndarray,
    spawn: tuple[int, int],
    hotspots: list[tuple[int, int]],
    n_zones: tuple[int, int] = (0, 3),
    radius_range: tuple[int, int] = (2, 5),
    min_dist_from_spawn: float = 6.0,
    edge_margin: float = 0.1,
    max_attempts: int = 200,
    rng=None,
) -> list[dict]: 
    
    """Places predator zones independently of hotspots (random location), avoiding
    spawn and avoiding directly overlapping a hotspot's exact cell (so sampling
    itself never has a forced attack roll. """
    rng = rng or np.random.default_rng()
    grid_size = impassable_mask.shape[0]
    zones = []

    #restrict zone centers to the middle 80% of the grid
    margin = int(grid_size * edge_margin)
    low, high = margin, grid_size - margin

    n_zones_this_episode = int(rng.integers(n_zones[0], n_zones[1] + 1))

    for _ in range(n_zones_this_episode):
        placed = False
        for _ in range(max_attempts):
            r = rng.integers(low, high)
            c = rng.integers(low, high)
            radius = rng.uniform(radius_range[0], radius_range[1])

            if impassable_mask[r, c]:
                continue
            if _dist((r, c), spawn) < min_dist_from_spawn:
                continue
            if any((r, c) == h for h in hotspots):
                continue

            mask = _generate_blob_mask(grid_size, (int(r), int(c)), radius, irregularity=0.45, rng=rng)
            zones.append({"center": (int(r), int(c)), "radius": radius, "type": "predator", "mask": mask})
            placed = True
            break

        if not placed:
            continue  # skip, non-critical

    return zones


def build_hazard_mask(grid_size: int, 
                      hazards: list[dict]) -> np.ndarray:
    
    """Combines all hazard zones (vents + predator) into one boolean occupancy mask,
    using each hazard's precomputed irregular blob mask."""
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for hazard in hazards:
        mask |= hazard["mask"]
    return mask


def hazard_type_at(pos: tuple[int, int], 
                   hazards: list[dict]) -> str | None:
    """
    Returns "vent", "predator", or None for the given position, using each
    hazard's precomputed irregular blob mask. If a cell is covered by both a
    vent and predator zone (rare, possible with random placement), vent takes
    priority since it's certain damage, not probabilistic.
    """
    r, c = pos
    found_predator = False
    for hazard in hazards:
        if hazard["mask"][r, c]:
            if hazard["type"] == "vent":
                return "vent"
            found_predator = True
    return "predator" if found_predator else None


def roll_predator_attack(prob: float = DEFAULT_PREDATOR_ATTACK_PROB,
                        rng=None) -> bool:
    """Called once per step when the agent's position is inside a predator zone."""
    rng = rng or np.random.default_rng()
    return rng.random() < prob


def _dist(a: tuple[int, int],
          b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))