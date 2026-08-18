import numpy as np
import matplotlib.pyplot as plt

from terrain_generator import generate_terrain, compute_impassable_terrain
from object_placement import place_spawn, place_edna_hotspots, generate_edna_field
from hazards import place_vents, place_predator_zones


def generate_full_episode(grid_size: int = 50, 
                          seed: int | None = None, 
                          n_hotspots_range=(3, 5)):
    
    """Runs the full generation pipeline for one episode and returns everything needed to render it."""
    rng = np.random.default_rng(seed)

    elevation = generate_terrain(grid_size=grid_size, seed=seed)
    impassable = compute_impassable_terrain(elevation)

    spawn = place_spawn(impassable, rng=rng)
    n_hotspots = int(rng.integers(n_hotspots_range[0], n_hotspots_range[1] + 1))
    hotspots = place_edna_hotspots(impassable, spawn, n_hotspots=n_hotspots, rng=rng)
    edna_field = generate_edna_field(grid_size, hotspots, rng=rng) if hotspots else np.zeros((grid_size, grid_size))

    vents = place_vents(hotspots, impassable, n_vents=min(2, len(hotspots)), rng=rng)
    predators = place_predator_zones(impassable, spawn, hotspots, rng=rng)

    return {
        "elevation": elevation,
        "impassable": impassable,
        "spawn": spawn,
        "hotspots": hotspots,
        "edna_field": edna_field,
        "vents": vents,
        "predators": predators,
        "grid_size": grid_size,
    }


def render_episode(ax, 
                   ep: dict, 
                   title: str = ""):
    """Renders one generated episode onto a given matplotlib axis."""
    grid_size = ep["grid_size"]

    # base terrain
    ax.imshow(ep["elevation"], cmap="terrain", vmin=0, vmax=1)

    # eDNA concentration field (soft blue glow)
    if ep["hotspots"]:
        ax.imshow(ep["edna_field"], cmap="Blues", alpha=0.35)

    # impassable terrain -> solid dark grey
    impassable_overlay = np.zeros((grid_size, grid_size, 4))
    impassable_overlay[ep["impassable"]] = [0.1, 0.1, 0.1, 0.75]
    ax.imshow(impassable_overlay)

    # vents -> bright orange
    for v in ep["vents"]:
        overlay = np.zeros((grid_size, grid_size, 4))
        overlay[v["mask"]] = [1.0, 0.55, 0.0, 0.8]
        ax.imshow(overlay)

    # predator zones -> bright magenta
    for p in ep["predators"]:
        overlay = np.zeros((grid_size, grid_size, 4))
        overlay[p["mask"]] = [1.0, 0.0, 0.8, 0.5]
        ax.imshow(overlay)

    # eDNA hotspots -> gold stars
    for (r, c) in ep["hotspots"]:
        ax.plot(c, r, marker="*", color="gold", markersize=14, markeredgecolor="black", zorder=5)

    # spawn -> lime circle
    if ep["spawn"]:
        ax.plot(ep["spawn"][1], ep["spawn"][0], marker="o", color="lime",
                 markersize=12, markeredgecolor="black", zorder=5)

    n_hotspots = len(ep["hotspots"])
    n_vents = len(ep["vents"])
    n_predators = len(ep["predators"])
    ax.set_title(f"{title}\nhotspots={n_hotspots}  vents={n_vents}  predators={n_predators}", fontsize=10)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_one_episode(seed: int | None = None, 
                          grid_size: int = 50, 
                          save_path: str = "../Images/Environment_check/latest_full_env_check.png"):
    """
    Visualizes a single generated episode.
    Pass seed=None (or omit it) for a fresh random episode each run.
    Pass a specific integer seed to reproduce the same episode again.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 1_000_000))

    ep = generate_full_episode(grid_size=grid_size, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 7))
    render_episode(ax, ep, title=f"Seed {seed}")

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markeredgecolor='black', markersize=10, label='Spawn'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=14, label='eDNA hotspot'),
        plt.Rectangle((0, 0), 1, 1, fc=[1.0, 0.55, 0.0, 0.8], label='Vent (certain damage)'),
        plt.Rectangle((0, 0), 1, 1, fc=[1.0, 0.0, 0.8, 0.5], label='Predator zone (0.5%/step)'),
        plt.Rectangle((0, 0), 1, 1, fc=[0.1, 0.1, 0.1, 0.75], label='Impassable terrain'),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"Saved {save_path} (seed={seed})")
    return ep


if __name__ == "__main__":
    seed = int(input("Enter seed (or leave blank for random): ") or np.random.default_rng().integers(0, 1_000_000))
    visualize_one_episode(seed=seed)