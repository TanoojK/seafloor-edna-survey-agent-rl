"""
Training script for the eDNA seafloor survey agent.

Runs N_ENVS environments in PARALLEL (separate OS processes, see vec_env.py),
each collecting its own rollout simultaneously, then combines them into one
PPO update. This is a wall-clock speedup only -- it does not change what's
being learned, just how fast experience is collected. Each environment still
gets its own correctly-computed GAE (see ppo.py's update_multi), so this
doesn't sacrifice correctness for speed.

Usage:
    python train.py                  # full run (500k steps/env, 50x50 grid)
    python train.py --fast-test      # small/fast proxy run for quickly checking
                                      # whether a change helps, before committing
                                      # to a full run (see FAST_TEST_* config below)

Checkpoints are saved periodically to ./checkpoints/ -- if training on Colab,
point CHECKPOINT_DIR at your mounted Google Drive so a disconnect doesn't lose
your run.

WINDOWS NOTE: this script MUST be run as `python train.py`, not imported or
exec'd some other way -- the `if __name__ == "__main__":` guard at the bottom
is not optional here. Windows' multiprocessing uses "spawn", which re-imports
this whole file in every worker process; without the guard, each worker would
try to spawn its own set of workers recursively.
"""

import argparse
import os
import time
import numpy as np

# ---------------------------------------------------------------------- #
# Config 
# ---------------------------------------------------------------------- #
N_ENVS = 8                   # parallel environments -- tune down if you hit
                                # CPU core or VRAM limits; 8 is a reasonable
                                # starting point for a modern multi-core CPU
                                # + RTX 3050
TOTAL_TIMESTEPS = 500_000      # per-environment; total env-steps collected
                                # across all workers is N_ENVS times this
ROLLOUT_LENGTH = 1024         # steps PER ENVIRONMENT collected before each
                                # PPO update (so each update uses
                                # ROLLOUT_LENGTH * N_ENVS transitions total)
CHECKPOINT_EVERY = 100_000      # timesteps (per-env) between checkpoint saves
CHECKPOINT_DIR = "./checkpoints"
LOG_EVERY_EPISODES = 100      # print a summary every N completed episodes (across all envs)

GRID_SIZE = 50
N_MAP_CHANNELS = 5             # elevation, eDNA field, vent mask, predator mask, sampled mask
N_SCALARS = 6                  # agent_row, agent_col, energy, canisters, spawn_row, spawn_col

ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END = 0.006       # was 0.003 -- raised slightly; entropy was crashing to
                                # near-zero (0.1-0.6) even before reaching the old floor
LR_START = 3e-4
LR_END = 3e-5                  # 10x decay over the run -- standard PPO practice, and the
                                # volatility in value_loss across every run so far suggests
                                # late-training updates are still as aggressive as early
                                # ones, fighting against convergence
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
FAST_TEST_GRID_SIZE = GRID_SIZE   # NOT shrunk -- see note above
FAST_TEST_CHECKPOINT_DIR = "./checkpoints_fast_test"

# ---- auto-stop: if avg_samples is still ~0 this far into a run, something is
# almost certainly broken (see project history -- this has happened multiple
# times). Stop early instead of burning the rest of the timestep budget and
# only finding out at the very end. ---- #
STUCK_CHECK_AFTER_TIMESTEPS = 420_000    # must be checked AFTER curriculum finishes
                                          # ramping (75% of TOTAL_TIMESTEPS = 375k), or
                                          # it judges "stuck" while the task is still
                                          # actively getting harder, which looks like
                                          # failure even when the policy is learning fine
STUCK_CHECK_SCALE_WITH_FAST_TEST = True  # scale the above proportionally in --fast-test
STUCK_AVG_SAMPLES_THRESHOLD = 0.01       # "still ~0" cutoff


def train(fast_test: bool = False):
    # imports deliberately deferred to here, not module level -- Windows'
    # spawn re-executes this file's top-level code in every worker process.
    # If torch/matplotlib were imported at module level, every worker would
    # load torch's full CUDA DLL stack too, even though workers only need
    # environment.py. With N_ENVS processes doing that simultaneously, it can
    # exhaust Windows' page file. Keeping these imports inside train() means
    # spawn's re-execution of this file stays lightweight in child processes.
    import torch
    import matplotlib.pyplot as plt
    from vec_env import SubprocVecEnv
    from ppo import PPOAgent

    total_timesteps = FAST_TEST_TOTAL_TIMESTEPS if fast_test else TOTAL_TIMESTEPS
    grid_size = FAST_TEST_GRID_SIZE if fast_test else GRID_SIZE
    checkpoint_dir = FAST_TEST_CHECKPOINT_DIR if fast_test else CHECKPOINT_DIR
    stuck_check_after = (
        int(STUCK_CHECK_AFTER_TIMESTEPS * (total_timesteps / TOTAL_TIMESTEPS))
        if (fast_test and STUCK_CHECK_SCALE_WITH_FAST_TEST) else STUCK_CHECK_AFTER_TIMESTEPS
    )

    if fast_test:
        print(f"*** FAST-TEST MODE: {total_timesteps} steps/env (full {grid_size}x{grid_size} grid, "
              f"not shrunk -- see comment above FAST_TEST_GRID_SIZE), "
              f"stuck-check at {stuck_check_after} steps. Not a substitute for a full run. ***")

    print(f"Running {N_ENVS} parallel environments.")
    os.makedirs(checkpoint_dir, exist_ok=True)

    vec_env = SubprocVecEnv(n_envs=N_ENVS, grid_size=grid_size, max_steps=500, gamma=0.99)
    agent = PPOAgent(
        n_map_channels=N_MAP_CHANNELS,
        n_scalars=N_SCALARS,
        n_actions=9,
        grid_size=grid_size,
        gamma=0.99,
        device=torch.device("cpu"),  # model is tiny -- GPU barely helps it, and this
                                      # sidesteps VRAM constraints entirely. The real
                                      # speedup from N_ENVS is CPU-side env stepping.
    )
    agent.init_parallel_buffers(N_ENVS)

    obs, masks = vec_env.reset(curriculum_start=CURRICULUM_START_DIST)
    maps_t = torch.from_numpy(obs["maps"])
    scalars_t = torch.from_numpy(obs["scalars"])
    masks_t = torch.from_numpy(masks)

    episode_rewards = [0.0] * N_ENVS       # running total for each env's CURRENT episode
    episode_rewards_log = []               # completed-episode total rewards, for plotting
    episode_end_reasons = []
    episode_samples_log = []               # samples collected per completed episode

    timesteps_done = 0
    episodes_done = 0
    start_time = time.time()

    metrics_log_path = os.path.join(checkpoint_dir, "metrics_log.csv")
    if not os.path.exists(metrics_log_path):
        with open(metrics_log_path, "w") as f:
            f.write("timesteps,episodes,avg_reward,avg_samples,entropy,value_loss,policy_loss,lr,elapsed_s\n")

    last_dones = np.zeros(N_ENVS, dtype=bool)

    try:
        while timesteps_done < total_timesteps:
            # ---- collect one rollout: ROLLOUT_LENGTH steps, across all N_ENVS
            # environments simultaneously ---- #
            for _ in range(ROLLOUT_LENGTH):
                # curriculum ceiling is computed from GLOBAL progress, same formula as
                # the original single-env version -- passed to every worker each step,
                # and applied by the worker itself whenever ITS OWN episode ends (see
                # vec_env.py's _worker), so each environment independently gets a fresh
                # randomly-sampled difficulty within [start, current ceiling] at its own
                # episode boundaries, keeping easy examples mixed in throughout training.
                progress_for_curriculum = min(timesteps_done / total_timesteps, 1.0)
                curriculum_progress = min(progress_for_curriculum / CURRICULUM_END_FRACTION, 1.0)
                if CURRICULUM_END_DIST is None:
                    curriculum_ceiling = CURRICULUM_START_DIST + curriculum_progress * (
                        grid_size * 0.7 - CURRICULUM_START_DIST
                    )
                else:
                    curriculum_ceiling = CURRICULUM_START_DIST + curriculum_progress * (
                        CURRICULUM_END_DIST - CURRICULUM_START_DIST
                    )

                actions_np, actions_t, log_probs_t, values_t = agent.select_action_batch(maps_t, scalars_t, masks_t)
                next_obs, rewards, terminateds, truncateds, infos, next_masks = vec_env.step(
                    actions_np, curriculum_start=CURRICULUM_START_DIST, curriculum_ceiling=curriculum_ceiling
                )
                dones = terminateds | truncateds

                for i in range(N_ENVS):
                    agent.store_transition_multi(
                        i, maps_t[i], scalars_t[i], actions_t[i], log_probs_t[i], values_t[i],
                        float(rewards[i]), bool(dones[i]), masks_t[i]
                    )
                    episode_rewards[i] += rewards[i]
                    timesteps_done += 1

                    if dones[i]:
                        episode_rewards_log.append(episode_rewards[i])
                        episode_end_reasons.append(infos[i].get("end_reason", "unknown"))
                        episode_samples_log.append(infos[i].get("samples_taken", 0))
                        episodes_done += 1
                        episode_rewards[i] = 0.0

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

                maps_t = torch.from_numpy(next_obs["maps"])
                scalars_t = torch.from_numpy(next_obs["scalars"])
                masks_t = torch.from_numpy(next_masks)
                last_dones = dones

                if timesteps_done >= total_timesteps:
                    break

            # ---- entropy annealing: decays across the FULL run (raw progress),
            # deliberately NOT tied to curriculum_progress -- see prior notes on
            # why they need to be decoupled (entropy hitting its floor exactly
            # when curriculum hits full difficulty left no exploration budget
            # for the hardest final stretch of training). ---- #
            progress = min(timesteps_done / total_timesteps, 1.0)
            agent.entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress

            # LR annealing, same reasoning: decays over the FULL run, independent of curriculum.
            current_lr = LR_START + (LR_END - LR_START) * progress
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = current_lr

            # ---- PPO update using everything collected this rollout, across all N_ENVS ---- #
            update_stats = agent.update_multi(maps_t, scalars_t, last_dones)
            print(f"    [update] entropy={update_stats['entropy']:.4f} "
                  f"value_loss={update_stats['value_loss']:.4f} "
                  f"policy_loss={update_stats['policy_loss']:.4f}")

            # ---- persist metrics ---- #
            recent = episode_rewards_log[-LOG_EVERY_EPISODES:] if episode_rewards_log else [0.0]
            recent_samples = episode_samples_log[-LOG_EVERY_EPISODES:] if episode_samples_log else [0.0]
            avg_samples_now = float(np.mean(recent_samples))
            with open(metrics_log_path, "a") as f:
                f.write(f"{timesteps_done},{episodes_done},{np.mean(recent):.4f},"
                        f"{avg_samples_now:.4f},{update_stats['entropy']:.4f},"
                        f"{update_stats['value_loss']:.4f},{update_stats['policy_loss']:.4f},"
                        f"{current_lr:.6f},{time.time() - start_time:.1f}\n")

            # ---- auto-stop ---- #
            if timesteps_done >= stuck_check_after and avg_samples_now < STUCK_AVG_SAMPLES_THRESHOLD:
                print(f"\n!!! STUCK: avg_samples={avg_samples_now:.4f} after {timesteps_done} steps "
                      f"(threshold={STUCK_AVG_SAMPLES_THRESHOLD}, checked after {stuck_check_after} steps). "
                      f"Stopping early -- something is very likely broken, not just 'needs more training'. "
                      f"Checkpoint saved below for inspection. !!!\n")
                torch.save(agent.model.state_dict(), os.path.join(checkpoint_dir, f"ppo_step{timesteps_done}_STUCK.pt"))
                _plot_rewards(episode_rewards_log, checkpoint_dir)
                return

            # ---- periodic checkpoint ---- #
            if timesteps_done % CHECKPOINT_EVERY < ROLLOUT_LENGTH * N_ENVS:
                ckpt_path = os.path.join(checkpoint_dir, f"ppo_step{timesteps_done}.pt")
                torch.save(agent.model.state_dict(), ckpt_path)
                print(f"  saved checkpoint: {ckpt_path}")

        # final checkpoint + reward curve
        torch.save(agent.model.state_dict(), os.path.join(checkpoint_dir, f"ppo_step{timesteps_done}_final.pt"))
        _plot_rewards(episode_rewards_log, checkpoint_dir)
        print("Training complete.")
    finally:
        vec_env.close()  # always clean up worker processes, even if something above crashes


def _summarize_reasons(reasons: list[str]) -> str:
    from collections import Counter
    counts = Counter(reasons)
    return ", ".join(f"{k}:{v}" for k, v in counts.most_common())


def _plot_rewards(rewards: list[float], checkpoint_dir: str):
    import matplotlib.pyplot as plt
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