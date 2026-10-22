import argparse
import os
import time
import numpy as np

# Config 

N_ENVS = 8                     # parallel environments 
TOTAL_TIMESTEPS = 500_000      # per-environment; total env-steps collected
                                # across all workers is N_ENVS times this
ROLLOUT_LENGTH = 2048          # steps PER ENVIRONMENT collected before each PPo update

CHECKPOINT_EVERY = 100_000      
CHECKPOINT_DIR = "./checkpoints"
LOG_EVERY_EPISODES = 100      # print a summary every N completed episodes (across all envs)

GRID_SIZE = 50
N_MAP_CHANNELS = 6             # elevation, eDNA field, vent mask, predator mask, sampled mask , position mask
N_SCALARS = 6                  # agent_row, agent_col, energy, canisters, spawn_row, spawn_col

ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END = 0.008       
LR_START = 3e-4
LR_END = 3e-5                  

CURRICULUM_START_DIST = 5
CURRICULUM_END_DIST = 5     
CURRICULUM_END_FRACTION = 1  

# Fast-test proxy run config 
FAST_TEST_TOTAL_TIMESTEPS = 100_000
FAST_TEST_GRID_SIZE = GRID_SIZE   # NOT shrunk -- see note above
FAST_TEST_CHECKPOINT_DIR = "./checkpoints_fast_test"

STUCK_CHECK_AFTER_TIMESTEPS = 420_000    
STUCK_CHECK_SCALE_WITH_FAST_TEST = True  # scale the above proportionally in fast-test
STUCK_AVG_SAMPLES_THRESHOLD = 0.01       # "still ~0" cutoff

def train(fast_test: bool = False):

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
    samples_history = []
    baseline_samples = None


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
            device=torch.device("cpu")
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
    total_env_steps = 0
    episodes_done = 0
    next_log_episode = LOG_EVERY_EPISODES
    start_time = time.time()

    metrics_log_path = os.path.join(checkpoint_dir, "metrics_log.csv")
    if not os.path.exists(metrics_log_path):
        with open(metrics_log_path, "w") as f:
            f.write("timesteps,episodes,avg_reward,avg_samples,entropy,value_loss,policy_loss,lr,elapsed_s\n")

    last_dones = np.zeros(N_ENVS, dtype=bool)

    try:
        while timesteps_done < total_timesteps:
            for _ in range(ROLLOUT_LENGTH):
                progress_for_curriculum = min(timesteps_done / total_timesteps, 1.0)
                curriculum_progress = min(progress_for_curriculum / CURRICULUM_END_FRACTION, 1.0)
                curriculum_progress = curriculum_progress ** 2
                if CURRICULUM_END_DIST is None:
                    curriculum_ceiling = CURRICULUM_START_DIST + curriculum_progress * (
                        grid_size * 0.6 - CURRICULUM_START_DIST
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

                    if dones[i]:
                        episode_rewards_log.append(episode_rewards[i])
                        episode_end_reasons.append(infos[i].get("end_reason", "unknown"))
                        episode_samples_log.append(infos[i].get("samples_taken", 0))
                        episodes_done += 1
                        episode_rewards[i] = 0.0

                timesteps_done += 1
                total_env_steps += N_ENVS

                while episodes_done >= next_log_episode:
                    recent = episode_rewards_log[-LOG_EVERY_EPISODES:]
                    recent_reasons = episode_end_reasons[-LOG_EVERY_EPISODES:]
                    recent_samples = episode_samples_log[-LOG_EVERY_EPISODES:]
                    elapsed = time.time() - start_time
                    print(f"[{timesteps_done:>8}/{total_timesteps} per-env, {total_env_steps} combined] "
                          f"episode {episodes_done:>5} | "
                          f"avg_reward(last {LOG_EVERY_EPISODES})={np.mean(recent):>7.2f} | "
                          f"avg_samples={np.mean(recent_samples):.2f} | "
                          f"reasons={_summarize_reasons(recent_reasons)} | "
                          f"elapsed={elapsed:.0f}s")
                    next_log_episode += LOG_EVERY_EPISODES


                maps_t = torch.from_numpy(next_obs["maps"])
                scalars_t = torch.from_numpy(next_obs["scalars"])
                masks_t = torch.from_numpy(next_masks)
                last_dones = dones

                if timesteps_done >= total_timesteps:
                    break


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
            samples_history.append(avg_samples_now)
            if baseline_samples is None and timesteps_done >= stuck_check_after * 0.25:
                baseline_samples = sum(samples_history) / len(samples_history)
            with open(metrics_log_path, "a") as f:
                f.write(f"{timesteps_done},{episodes_done},{np.mean(recent):.4f},"
                        f"{avg_samples_now:.4f},{update_stats['entropy']:.4f},"
                        f"{update_stats['value_loss']:.4f},{update_stats['policy_loss']:.4f},"
                        f"{current_lr:.6f},{time.time() - start_time:.1f}\n")

            # ---- auto-stop if stuck ---- #
            recent_window = samples_history[-10:]
            recent_avg = sum(recent_window) / len(recent_window)
            if (baseline_samples is not None and timesteps_done >= stuck_check_after
                    and recent_avg < max(STUCK_AVG_SAMPLES_THRESHOLD, baseline_samples * 1.5)):
                print(f"\n! STUCK: recent avg_samples={recent_avg:.4f} after {timesteps_done} steps "
                      f"(baseline={baseline_samples:.4f}, "
                      f"needed > {max(STUCK_AVG_SAMPLES_THRESHOLD, baseline_samples * 1.5):.4f}, "
                      f"checked after {stuck_check_after} steps). "
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