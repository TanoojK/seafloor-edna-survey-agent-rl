#Placement logic for spawn point and eDNA hotspots.
import numpy as np

def place_spawn(impassable_mask: np.ndarray,
                edge_margin_spawn: float = 0.2, 
                max_attempts: int = 100, 
                rng=None) -> tuple[int, int] | None:
    """
    Returns (row, col) for a random passable spawn location, or None if no
    valid spot was found within max_attempts (caller should regenerate terrain).
    """
    rng = rng or np.random.default_rng()
    grid_size = impassable_mask.shape[0]
    margin_spawn = int(grid_size * edge_margin_spawn)
    low, high = margin_spawn, grid_size - margin_spawn

    for _ in range(max_attempts):
        r = rng.integers(low, high)
        c = rng.integers(low, high)
        if not impassable_mask[r, c]:
            return (int(r), int(c))
    return None


def place_edna_hotspots(
    impassable_mask: np.ndarray,
    spawn: tuple[int, int],
    n_hotspots: int,
    min_dist_from_spawn: float = 8.0,
    max_dist_from_spawn: float | None = None,
    min_dist_between_hotspots: float = 6.0,
    max_attempts_per_hotspot: int = 200,
    edge_margin_hotspots: float = 0.05,
    rng=None,
) -> list[tuple[int, int]] | None:
    """
    Returns a list of (row, col) positions for eDNA hotspots, or None if it
    couldn't place all of them within the attempt budget (caller should
    regenerate terrain/spawn and retry).
    """
    rng = rng or np.random.default_rng()
    grid_size = impassable_mask.shape[0]
    hotspots: list[tuple[int, int]] = []
    margin_hotspots = int(grid_size * edge_margin_hotspots)
    low, high = margin_hotspots, grid_size - margin_hotspots

    for _ in range(n_hotspots):
        placed = False
        for _ in range(max_attempts_per_hotspot):
            r, c = rng.integers(low, high, size=2)
            candidate = (int(r), int(c))

            if impassable_mask[r, c]:
                continue
            if _dist(candidate, spawn) < min_dist_from_spawn:
                continue
            if max_dist_from_spawn is not None and _dist(candidate, spawn) > max_dist_from_spawn:
                continue
            if any(_dist(candidate, h) < min_dist_between_hotspots for h in hotspots):
                continue

            hotspots.append(candidate)
            placed = True
            break

        if not placed:
            return None  # couldn't satisfy constraints -- caller regenerates

    return hotspots


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def generate_edna_field(
    grid_size: int,
    hotspots: list[tuple[int, int]],
    sigma_range: tuple[float, float] = (3.0, 6.0),
    rng=None,
) -> np.ndarray:
    """
    Builds a continuous eDNA concentration field as a sum of Gaussians centered
    at each hotspot. Returned field is normalized to [0, 1].
    """
    rng = rng or np.random.default_rng()
    field = np.zeros((grid_size, grid_size), dtype=np.float32)
    rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")

    for (hr, hc) in hotspots:
        sigma = rng.uniform(*sigma_range)
        field += np.exp(-((rows - hr) ** 2 + (cols - hc) ** 2) / (2 * sigma ** 2))

    field = field / (field.max() + 1e-8)
    return field.astype(np.float32)