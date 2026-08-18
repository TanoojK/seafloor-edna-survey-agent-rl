"""
Visualizes a single episode (rollout) of the agent so training problems --
reward hacking, hazard-avoidance failures, energy mismanagement, stuck/looping
behavior, etc. -- are actually visible instead of inferred from aggregate stats.

Produces two outputs per run:
    1. A static trajectory plot: the full path overlaid on the terrain/hazard
       map, colored by time (dark -> light = early -> late), with sample
       events (green X) and the death/end location (red X or lime circle for
       a safe return) marked.
    2. An animated GIF: frame-by-frame playback with a live readout of step,
       reward, energy, canisters, and the running total reward -- useful for
       spotting things like "it clearly saw the vent and walked in anyway" or
       "it's oscillating between two cells instead of committing to a path."

Usage:
    # trained checkpoint
    python visualize_rollout.py --checkpoint checkpoints/ppo_final.pt --seed 7

    # no checkpoint -> uses an untrained (random-weight) policy, useful as a
    # sanity check on the environment/rendering itself
    python visualize_rollout.py --seed 7

    # random ACTION policy instead of a network at all (fastest sanity check)
    python visualize_rollout.py --seed 7 --random-policy

Outputs are written to ./rollout_viz/ by default (trajectory_seedN.png,
rollout_seedN.gif).
"""

import argparse
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from environment import SeafloorEDNAEnv
from ppo import PPOAgent


def parse_checkpoint_step(path):
    """Best-effort extraction of the training step count from a checkpoint's
    filename (e.g. 'ppo_step300000.pt' -> 300000, 'ppo_step500000_final.pt'
    -> 500000). Returns None if the filename doesn't follow that pattern --
    the plot just omits the step count rather than erroring out."""
    match = re.search(r"step(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


# ------------------------------------------------------------------ #
# Rollout collection
# ------------------------------------------------------------------ #

def to_tensor(obs):
    return torch.from_numpy(obs["maps"]), torch.from_numpy(obs["scalars"])


def run_episode(env, agent=None, max_steps=500, deterministic=False, random_policy=False, seed=None):
    """Runs one episode and records everything needed to visualize it.

    agent=None + random_policy=True -> uniform random actions (fastest, no
    network needed at all -- good for checking the env/renderer in isolation).
    agent=None + random_policy=False -> not allowed, agent is required unless
    random_policy is set.
    deterministic=True -> take argmax action instead of sampling (shows the
    policy's "best guess" rather than its stochastic behavior; useful once
    training has converged enough that sampling noise obscures the strategy).
    """
    obs, info = env.reset(seed=seed)

    record = {
        "positions": [env.agent_pos],
        "actions": [],
        "rewards": [],
        "energy": [env.energy],
        "canisters": [env.canisters],
        "samples_taken": [0],
        "end_reason": None,
        # static episode layout, captured once at reset
        "elevation": env.elevation.copy(),
        "impassable": env.impassable.copy(),
        "spawn": env.spawn,
        "hotspots": list(env.hotspots),
        "edna_field": env.edna_field.copy(),
        "vents": env.vents,
        "predators": env.predators,
        "grid_size": env.grid_size,
    }

    maps, scalars = to_tensor(obs)

    for _ in range(max_steps):
        if random_policy:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                maps_b = maps.unsqueeze(0).to(agent.device)
                scalars_b = scalars.unsqueeze(0).to(agent.device)
                dist, value = agent.model.forward(maps_b, scalars_b)
                action = (torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()).item()

        obs, reward, terminated, truncated, info = env.step(action)

        record["actions"].append(action)
        record["rewards"].append(reward)
        record["positions"].append(env.agent_pos)
        record["energy"].append(env.energy)
        record["canisters"].append(env.canisters)
        record["samples_taken"].append(info.get("samples_taken", 0))

        maps, scalars = to_tensor(obs)

        if terminated or truncated:
            record["end_reason"] = info.get("end_reason", "unknown")
            break

    if record["end_reason"] is None:
        record["end_reason"] = "did_not_terminate"  # ran max_steps in this script without hitting env's own cutoff

    return record


# ------------------------------------------------------------------ #
# Static background (terrain + hazards + hotspots), shared by both outputs
# ------------------------------------------------------------------ #

def draw_background(ax, rec):
    grid_size = rec["grid_size"]

    ax.imshow(rec["elevation"], cmap="terrain", vmin=0, vmax=1)

    if rec["hotspots"]:
        ax.imshow(rec["edna_field"], cmap="Blues", alpha=0.30)

    impassable_overlay = np.zeros((grid_size, grid_size, 4))
    impassable_overlay[rec["impassable"]] = [0.1, 0.1, 0.1, 0.75]
    ax.imshow(impassable_overlay)

    for v in rec["vents"]:
        overlay = np.zeros((grid_size, grid_size, 4))
        overlay[v["mask"]] = [1.0, 0.55, 0.0, 0.8]
        ax.imshow(overlay)

    for p in rec["predators"]:
        overlay = np.zeros((grid_size, grid_size, 4))
        overlay[p["mask"]] = [1.0, 0.0, 0.8, 0.5]
        ax.imshow(overlay)

    for (r, c) in rec["hotspots"]:
        ax.plot(c, r, marker="*", color="gold", markersize=14, markeredgecolor="black", zorder=5)

    ax.plot(rec["spawn"][1], rec["spawn"][0], marker="o", color="lime",
             markersize=12, markeredgecolor="black", zorder=5)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_xticks([])
    ax.set_yticks([])


# ------------------------------------------------------------------ #
# Output 1: static trajectory plot
# ------------------------------------------------------------------ #

def plot_trajectory(rec, save_path, checkpoint_step=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_background(ax, rec)

    positions = rec["positions"]
    xs = [p[1] for p in positions]
    ys = [p[0] for p in positions]

    # path colored by time: build line segments and color by step index
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="cool", linewidth=2.2, zorder=6)
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)

    # mark sample events: step i where samples_taken increased
    samples = rec["samples_taken"]
    for i in range(1, len(samples)):
        if samples[i] > samples[i - 1]:
            r, c = positions[i]
            ax.plot(c, r, marker="X", color="chartreuse", markersize=13,
                     markeredgecolor="black", zorder=7)

    # mark the end location
    end_r, end_c = positions[-1]
    end_reason = rec["end_reason"]
    if end_reason == "returned_to_spawn":
        ax.plot(end_c, end_r, marker="o", markersize=16, markerfacecolor="none",
                 markeredgecolor="lime", markeredgewidth=3, zorder=8)
    else:
        ax.plot(end_c, end_r, marker="X", color="red", markersize=16,
                 markeredgecolor="black", markeredgewidth=1.5, zorder=8)

    n_samples_final = samples[-1]
    total_reward = sum(rec["rewards"])
    n_steps = len(rec["actions"])
    step_tag = f"checkpoint_step={checkpoint_step} | " if checkpoint_step is not None else ""
    ax.set_title(
        f"{step_tag}end_reason={end_reason} | steps={n_steps} | samples={n_samples_final} | "
        f"total_reward={total_reward:.2f}",
        fontsize=11,
    )

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markeredgecolor='black', markersize=10, label='Spawn'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=14, label='eDNA hotspot'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='chartreuse', markeredgecolor='black', markersize=12, label='Sample taken'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='black', markersize=12, label='Death location'),
        plt.Rectangle((0, 0), 1, 1, fc=[1.0, 0.55, 0.0, 0.8], label='Vent'),
        plt.Rectangle((0, 0), 1, 1, fc=[1.0, 0.0, 0.8, 0.5], label='Predator zone'),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory plot: {save_path}")


# ------------------------------------------------------------------ #
# Output 2: animated GIF, frame-by-frame playback
# ------------------------------------------------------------------ #

def animate_rollout(rec, save_path, fps=8, checkpoint_step=None):
    fig, ax = plt.subplots(figsize=(7, 7.5))
    draw_background(ax, rec)

    positions = rec["positions"]
    xs = [p[1] for p in positions]
    ys = [p[0] for p in positions]

    (trail_line,) = ax.plot([], [], color="deepskyblue", linewidth=2, zorder=6, alpha=0.8)
    (agent_dot,) = ax.plot([], [], marker="o", color="white", markersize=11,
                            markeredgecolor="black", markeredgewidth=1.5, zorder=9)
    info_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=9, color="white",
        bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.3"),
    )

    n_frames = len(positions)
    cum_reward = np.cumsum([0.0] + rec["rewards"])

    def update(frame):
        trail_line.set_data(xs[: frame + 1], ys[: frame + 1])
        agent_dot.set_data([xs[frame]], [ys[frame]])
        step = frame
        energy = rec["energy"][frame]
        canisters = rec["canisters"][frame]
        samples = rec["samples_taken"][frame]
        reward_so_far = cum_reward[frame]
        step_reward = rec["rewards"][frame - 1] if frame > 0 else 0.0
        step_tag = f"ckpt step: {checkpoint_step}\n" if checkpoint_step is not None else ""
        info_text.set_text(
            f"{step_tag}"
            f"step {step}/{n_frames - 1}\n"
            f"energy: {energy:.1f}\n"
            f"canisters: {canisters}\n"
            f"samples: {samples}\n"
            f"step reward: {step_reward:+.2f}\n"
            f"total reward: {reward_so_far:+.2f}"
        )
        if frame == n_frames - 1 and rec["end_reason"] is not None:
            ax.set_title(f"END: {rec['end_reason']}", fontsize=12, color="darkred")
        return trail_line, agent_dot, info_text

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(save_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved rollout animation: {save_path}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a saved model state_dict (e.g. checkpoints/ppo_final.pt). "
                              "If omitted, uses an untrained network (or --random-policy).")
    parser.add_argument("--random-policy", action="store_true",
                         help="Ignore any network entirely and take uniform random actions. "
                              "Fastest way to sanity-check the environment/renderer.")
    parser.add_argument("--deterministic", action="store_true",
                         help="Take the argmax action instead of sampling from the policy. "
                              "Useful for a converged policy to see its 'best guess' strategy.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed, for reproducible episodes.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="./rollout_viz")
    parser.add_argument("--no-gif", action="store_true", help="Skip the (slower) GIF export, only make the static plot.")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    env = SeafloorEDNAEnv(grid_size=args.grid_size, max_steps=args.max_steps)

    agent = None
    if not args.random_policy:
        agent = PPOAgent(n_map_channels=5, n_scalars=6, n_actions=9, grid_size=args.grid_size)
        if args.checkpoint:
            agent.model.load_state_dict(torch.load(args.checkpoint, map_location=agent.device))
            agent.model.eval()
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print("No --checkpoint given -- using an UNTRAINED network (random weights).")

    rec = run_episode(
        env, agent=agent, max_steps=args.max_steps,
        deterministic=args.deterministic, random_policy=args.random_policy, seed=args.seed,
    )

    seed_tag = args.seed if args.seed is not None else "rand"
    ckpt_step = parse_checkpoint_step(args.checkpoint) if args.checkpoint else None
    step_suffix = f"_step{ckpt_step}" if ckpt_step is not None else ""

    plot_trajectory(rec, os.path.join(args.outdir, f"trajectory_seed{seed_tag}{step_suffix}.png"),
                     checkpoint_step=ckpt_step)
    if not args.no_gif:
        animate_rollout(rec, os.path.join(args.outdir, f"rollout_seed{seed_tag}{step_suffix}.gif"),
                         fps=args.fps, checkpoint_step=ckpt_step)

    print(f"\nend_reason={rec['end_reason']}  steps={len(rec['actions'])}  "
          f"samples={rec['samples_taken'][-1]}  total_reward={sum(rec['rewards']):.2f}")


if __name__ == "__main__":
    main()