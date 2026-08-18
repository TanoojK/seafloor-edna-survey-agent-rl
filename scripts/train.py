"""
Training script for the eDNA seafloor survey agent.

Collects experience in fixed-size rollouts (not necessarily aligned to episode
boundaries -- an episode can end mid-rollout, in which case the buffer just
keeps going into the next episode, which is standard PPO practice), then runs
a PPO update once the rollout buffer is full.

Usage:
    python train.py                  # full run (500k steps, 50x50 grid)
    python train.py --fast-test      # small/fast proxy run for quickly checking
                                      # whether a change helps, before committing
                                      # to a full run (see FAST_TEST_* config below)

Checkpoints are saved periodically to ./checkpoints/ -- if training on Colab,
point CHECKPOINT_DIR at your mounted Google Drive so a disconnect doesn't lose
your run.
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import SeafloorEDNAEnv
from ppo import PPOAgent

# ---------------------------------------------------------------------- #
# Config 
# ---------------------------------------------------------------------- #
TOTAL_TIMESTEPS = 500_000
ROLLOUT_LENGTH = 2048          # steps collected before each PPO update
CHECKPOINT_EVERY = 100_000      # timesteps between checkpoint saves
CHECKPOINT_DIR = "./checkpoints"
LOG_EVERY_EPISODES = 100      # print a summary every N completed episodes

GRID_SIZE = 50
N_MAP_CHANNELS = 5             # elevation, eDNA field, vent mask, predator mask, sampled mask
N_SCALARS = 6                  # agent_row, agent_col, energy, canisters, spawn_row, spawn_col

ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END = 0.006    # was 0.003 -- raised slightly; entropy was crashing to
                             # near-zero (0.1-0.6) even before reaching the old floor
CURRICULUM_START_DIST = 10.0
CURRICULUM_END_DIST = None      # None = grows up to grid_size * 0.7
CURRICULUM_END_FRACTION = 0.75

# ---- fast-test mode: a shorter proxy run for quickly checking whether a
# change helps, in minutes instead of the ~20+ min a full run takes. Bugs like
# late-training forgetting already show up within the first ~1/4 of a full
# run's timesteps, so a shortened version tends to surface the same signal
# fast. Only trust a FULL run to confirm a fix actually works end to end --
# this is for ruling things OUT quickly, not for a final verdict.
#
# NOTE: grid_size is deliberately NOT shrunk here. Shrinking it seemed like an
# obvious extra speedup, but hotspot placement has several distance constants
# (min_dist_from_spawn, min_dist_between_hotspots, curriculum start/ceiling)
# that all interact -- scaling one without the others made hotspot placement
# fail 100% of the time on a 20x20 grid, and even partial rescaling still hit
# geometric infeasibility (can't fit N hotspots that are all close to spawn
# AND far apart from each other in a small ring). Not worth the fragility for
# a speed gain -- timestep count alone is a safer, still-useful lever. ----#
FAST_TEST_TOTAL_TIMESTEPS = 100_000
FAST_TEST_GRID_SIZE = GRID_SIZE   # NOT shrunk -- see note below
FAST_TEST_CHECKPOINT_DIR = "./checkpoints_fast_test"

# ---- auto-stop: if avg_samples is still ~0 this far into a run, something is
# almost certainly broken (see project history -- this has happened multiple
# times). Stop early instead of burning the rest of the timestep budget and
# only finding out at the very end. ---- #
STUCK_CHECK_AFTER_TIMESTEPS = 150_000   # don't judge before this many steps
STUCK_CHECK_SCALE_WITH_FAST_TEST = True  # scale the above proportionally in --fast-test
STUCK_AVG_SAMPLES_THRESHOLD = 0.01       # "still ~0" cutoff


def to_tensor(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a single env observation (numpy dict) into torch tensors for the agent."""
    maps = torch.from_numpy(obs["maps"])
    scalars = torch.from_numpy(obs["scalars"])
    return maps, scalars


def train(fast_test: bool = False):
    total_timesteps = FAST_TEST_TOTAL_TIMESTEPS if fast_test else TOTAL_TIMESTEPS
    grid_size = FAST_TEST_GRID_SIZE if fast_test else GRID_SIZE
    checkpoint_dir = FAST_TEST_CHECKPOINT_DIR if fast_test else CHECKPOINT_DIR
    stuck_check_after = (
        int(STUCK_CHECK_AFTER_TIMESTEPS * (total_timesteps / TOTAL_TIMESTEPS))
        if (fast_test and STUCK_CHECK_SCALE_WITH_FAST_TEST) else STUCK_CHECK_AFTER_TIMESTEPS
    )

    if fast_test:
        print(f"*** FAST-TEST MODE: {total_timesteps} steps (full {grid_size}x{grid_size} grid, "
              f"not shrunk -- see comment above FAST_TEST_GRID_SIZE), "
              f"stuck-check at {stuck_check_after} steps. Not a substitute for a full run. ***")

    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SeafloorEDNAEnv(grid_size=grid_size, max_steps=500, gamma=0.99)
    agent = PPOAgent(
        n_map_channels=N_MAP_CHANNELS,
        n_scalars=N_SCALARS,
        n_actions=9,
        grid_size=grid_size,
        gamma=0.99,
    )

    env.max_hotspot_dist = CURRICULUM_START_DIST  # first episode starts easy
    obs, info = env.reset()
    maps, scalars = to_tensor(obs)

    episode_reward = 0.0
    episode_length = 0
    episode_rewards_log = []       # completed-episode total rewards, for plotting
    episode_end_reasons = []
    episode_samples_log = []       # samples collected per completed episode

    timesteps_done = 0
    episodes_done = 0
    start_time = time.time()

    metrics_log_path = os.path.join(checkpoint_dir, "metrics_log.csv")
    if not os.path.exists(metrics_log_path):
        with open(metrics_log_path, "w") as f:
            f.write("timesteps,episodes,avg_reward,avg_samples,entropy,value_loss,policy_loss,elapsed_s\n")

    while timesteps_done < total_timesteps:
        # ---- collect one rollout ---- #
        for _ in range(ROLLOUT_LENGTH):
            action_mask = torch.from_numpy(env.get_action_mask())
            action_int, action_t, log_prob, value = agent.select_action(maps, scalars, action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            agent.store_transition(maps, scalars, action_t, log_prob, value, reward, done, action_mask)

            episode_reward += reward
            episode_length += 1
            timesteps_done += 1

            if done:
                episode_rewards_log.append(episode_reward)
                episode_end_reasons.append(info.get("end_reason", "unknown"))
                episode_samples_log.append(info.get("samples_taken", 0))
                episodes_done += 1

                if episodes_done % LOG_EVERY_EPISODES == 0:
                    recent = episode_rewards_log[-LOG_EVERY_EPISODES:]
                    recent_reasons = episode_end_reasons[-LOG_EVERY_EPISODES:]
                    recent_samples = episode_samples_log[-LOG_EVERY_EPISODES:]
                    elapsed = time.time() - start_time
                    print(f"[{timesteps_done:>8}/{total_timesteps}] "
                          f"episode {episodes_done:>5} | "
                          f"avg_reward(last {LOG_EVERY_EPISODES})={np.mean(recent):>7.2f} | "
                          f"avg_samples={np.mean(recent_samples):.2f} | "
                          f"reasons={_summarize_reasons(recent_reasons)} | "
                          f"elapsed={elapsed:.0f}s")

                # ---- curriculum: pick THIS episode's difficulty randomly up to
                # the current ceiling (not pinned at the ceiling), so easy
                # episodes stay mixed in and the "reach a nearby hotspot" skill
                # doesn't get forgotten as the ceiling grows ---- #
                progress = min(timesteps_done / total_timesteps, 1.0)
                curriculum_progress = min(progress / CURRICULUM_END_FRACTION, 1.0)
                if CURRICULUM_END_DIST is None:
                    curriculum_ceiling = CURRICULUM_START_DIST + curriculum_progress * (
                        env.grid_size * 0.7 - CURRICULUM_START_DIST
                    )
                else:
                    curriculum_ceiling = CURRICULUM_START_DIST + curriculum_progress * (
                        CURRICULUM_END_DIST - CURRICULUM_START_DIST
                    )
                env.max_hotspot_dist = float(np.random.uniform(CURRICULUM_START_DIST, curriculum_ceiling))

                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs

            maps, scalars = to_tensor(obs)

            if timesteps_done >= total_timesteps:
                break

        # ---- entropy annealing: decays across the FULL run (raw progress),
        # deliberately NOT tied to curriculum_progress. An earlier version tied
        # them together on the theory that "exploration shouldn't shrink faster
        # than the task gets harder" -- but that meant entropy hit its floor at
        # the same time curriculum hit full difficulty (75%), leaving the
        # hardest final stretch of training with almost no exploration left to
        # recover from any drift. Decoupling leaves a residual exploration
        # budget through the end of training. ---- #
        progress = min(timesteps_done / total_timesteps, 1.0)
        agent.entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress

        # ---- PPO update using everything collected this rollout ---- #
        update_stats = agent.update(maps, scalars, last_done=done)
        print(f"    [update] entropy={update_stats['entropy']:.4f} "
              f"value_loss={update_stats['value_loss']:.4f} "
              f"policy_loss={update_stats['policy_loss']:.4f}")

        # ---- persist metrics so a checkpoint's training stage is checkable
        # later (e.g. from visualize_rollout.py) without having to have kept
        # the console log around ---- #
        recent = episode_rewards_log[-LOG_EVERY_EPISODES:] if episode_rewards_log else [0.0]
        recent_samples = episode_samples_log[-LOG_EVERY_EPISODES:] if episode_samples_log else [0.0]
        avg_samples_now = float(np.mean(recent_samples))
        with open(metrics_log_path, "a") as f:
            f.write(f"{timesteps_done},{episodes_done},{np.mean(recent):.4f},"
                    f"{avg_samples_now:.4f},{update_stats['entropy']:.4f},"
                    f"{update_stats['value_loss']:.4f},{update_stats['policy_loss']:.4f},"
                    f"{time.time() - start_time:.1f}\n")

        # ---- auto-stop: bail out early if it's clearly stuck, instead of
        # burning the rest of the run and only finding out at the end ---- #
        if timesteps_done >= stuck_check_after and avg_samples_now < STUCK_AVG_SAMPLES_THRESHOLD:
            print(f"\n!!! STUCK: avg_samples={avg_samples_now:.4f} after {timesteps_done} steps "
                  f"(threshold={STUCK_AVG_SAMPLES_THRESHOLD}, checked after {stuck_check_after} steps). "
                  f"Stopping early -- something is very likely broken, not just 'needs more training'. "
                  f"Checkpoint saved below for inspection. !!!\n")
            torch.save(agent.model.state_dict(), os.path.join(checkpoint_dir, f"ppo_step{timesteps_done}_STUCK.pt"))
            _plot_rewards(episode_rewards_log, checkpoint_dir)
            return

        # ---- periodic checkpoint ---- #
        if timesteps_done % CHECKPOINT_EVERY < ROLLOUT_LENGTH:
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_step{timesteps_done}.pt")
            torch.save(agent.model.state_dict(), ckpt_path)
            print(f"  saved checkpoint: {ckpt_path}")

    # final checkpoint + reward curve
    torch.save(agent.model.state_dict(), os.path.join(checkpoint_dir, f"ppo_step{timesteps_done}_final.pt"))
    _plot_rewards(episode_rewards_log, checkpoint_dir)
    print("Training complete.")


def _summarize_reasons(reasons: list[str]) -> str:
    from collections import Counter
    counts = Counter(reasons)
    return ", ".join(f"{k}:{v}" for k, v in counts.most_common())


def _plot_rewards(rewards: list[float], checkpoint_dir: str):
    if not rewards:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="per-episode reward")
    if len(rewards) >= 20:
        window = 20
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-episode moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training reward over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "reward_curve.png"), dpi=100)
    print(f"Saved reward curve to {checkpoint_dir}/reward_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-test", action="store_true",
                         help="Run a small/fast proxy training run instead of the full one, "
                              "to quickly check whether a change helps before committing to "
                              "a full run. See FAST_TEST_* constants to tune size/length.")
    args = parser.parse_args()
    train(fast_test=args.fast_test)