"""
Corrected direct-path survivability test.

The original version in the project notes walked in a straight line (sign of
delta row/col) toward the nearest hotspot. That's not "the best a sensible
policy could do" -- it gets stuck indefinitely against impassable terrain,
which silently inflated the max_steps bucket and hid the true death rate.

This version uses Dijkstra's algorithm to find the actual CHEAPEST path (by
energy cost, matching the environment's real move costs -- cardinal=1.0,
diagonal=1.5, plus a small elevation adjustment) from the agent's current cell
to its target, avoiding impassable cells, and takes the first step of that
path each turn. This is a step up from plain BFS (shortest path by tile
count), which can pick a route that's fewer tiles but MORE energy, since
diagonal moves cost more than cardinal ones. Recomputed every step since
sampling can change the target.
"""

import heapq
import numpy as np
import sys
sys.path.insert(0, "/home/claude")
from environment import SeafloorEDNAEnv, CARDINAL_COST, DIAGONAL_COST, ELEVATION_FLAT_COST
from terrain_generator import elevation_step_cost_flat

DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
IS_DIAGONAL = [False, True, False, True, False, True, False, True]


def dijkstra_next_step(start, target, blocked, elevation):
    """Returns the first move (as an action index 0-7) along the CHEAPEST
    (by energy cost, matching the real environment's move costs) 8-connected
    path from start to target, avoiding any cell marked in `blocked`.
    Returns None if unreachable."""
    if start == target:
        return None
    grid_size = blocked.shape[0]
    best_cost = {start: 0.0}
    first_move = {}
    # min-heap of (cost_so_far, tie_breaker, position)
    heap = [(0.0, 0, start)]
    counter = 1  # tie-breaker so the heap never tries to compare tuples directly

    while heap:
        cost, _, cur = heapq.heappop(heap)
        if cost > best_cost.get(cur, float("inf")):
            continue  # stale entry, a cheaper path to `cur` was already found
        if cur == target:
            break
        for a, (dr, dc) in enumerate(DIRS):
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue
            if blocked[nr, nc]:
                continue
            move_cost = DIAGONAL_COST if IS_DIAGONAL[a] else CARDINAL_COST
            elev_cost = elevation_step_cost_flat(elevation, cur, (nr, nc), flat_cost=ELEVATION_FLAT_COST)
            new_cost = cost + move_cost + elev_cost

            if new_cost < best_cost.get((nr, nc), float("inf")):
                best_cost[(nr, nc)] = new_cost
                first_move[(nr, nc)] = first_move.get(cur, a) if cur != start else a
                heapq.heappush(heap, (new_cost, counter, (nr, nc)))
                counter += 1

    if target not in first_move:
        return None  # unreachable
    return first_move[target]


def direct_path_action(env):
    """A 'sensible, near-optimal but simple' policy: routes around impassable
    terrain AND around vents (certain death, and fully visible in the vent-mask
    observation channel from step 1 -- any reasonable policy avoids what it can
    see is lethal), using the actual cheapest-energy path (Dijkstra), not just
    the fewest-tiles path. Predator zones are left walkable: the risk is small
    per step (0.5%) and probabilistic, so accepting brief exposure while
    passing through is a legitimate trade-off, not an oversight."""
    ar, ac = env.agent_pos
    unsampled = [h for h in env.hotspots if env.sampled_mask[h] == 0] if env.canisters > 0 else []
    target = env.spawn if not unsampled else min(
        unsampled, key=lambda h: np.hypot(ar - h[0], ac - h[1])
    )
    if (ar, ac) == target:
        return 8 if target != env.spawn else 0  # 0 (N) is a no-op fallback; spawn-arrival terminates anyway

    blocked = env.impassable | env.vent_mask
    move = dijkstra_next_step((ar, ac), target, blocked, env.elevation)
    if move is None:
        # hotspot unreachable without crossing a vent (rare, given clearance
        # buffer) -- fall back to allowing vent cells rather than idling forever
        move = dijkstra_next_step((ar, ac), target, env.impassable, env.elevation)
        if move is None:
            return 8
    return move


def run(n_episodes=200, seed=0):
    env = SeafloorEDNAEnv(grid_size=50, max_steps=500)
    rng = np.random.default_rng(seed)

    end_reasons = []
    samples = []
    stuck = 0

    for ep in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        reason = None
        for t in range(500):
            a = direct_path_action(env)
            _, r, term, trunc, info = env.step(a)
            if term or trunc:
                reason = info.get("end_reason", "unknown")
                break
        if reason is None:
            reason = "no_end"
        end_reasons.append(reason)
        samples.append(info.get("samples_taken", 0))

    from collections import Counter
    counts = Counter(end_reasons)
    died = counts.get("vent", 0) + counts.get("predator_attack", 0)
    print(f"episodes: {n_episodes}")
    print(f"end reasons: {dict(counts)}")
    print(f"avg samples: {np.mean(samples):.2f}")
    print(f"death rate (vent+predator): {died / n_episodes:.1%}")
    print(f"  vent-only rate: {counts.get('vent', 0) / n_episodes:.1%}")
    print(f"  predator-only rate: {counts.get('predator_attack', 0) / n_episodes:.1%}")
    print(f"success rate (returned_to_spawn): {counts.get('returned_to_spawn', 0) / n_episodes:.1%}")


if __name__ == "__main__":
    run()