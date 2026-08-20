"""
Runs N environment instances in parallel, each in its own OS process, following
the same pattern as Stable-Baselines3's SubprocVecEnv.

Each worker creates and owns exactly one SeafloorEDNAEnv instance for its entire
lifetime -- it's never recreated, just repeatedly reset(). This matters for RNG
behavior: each worker's env self-seeds from OS entropy once at construction
(same as the existing single-environment design), and that RNG state naturally
continues advancing across every reset() within that worker -- independent
across workers since they're separate OS processes with independent entropy.

IMPORTANT (Windows): multiprocessing on Windows uses the "spawn" start method,
not "fork". This means: (1) the calling script MUST guard process creation with
`if __name__ == "__main__":`, or spawn will recursively re-execute the whole
module in every child process; (2) everything passed to Process(args=...) must
be picklable -- we never pass an env instance itself, only plain ints/floats and
multiprocessing.Pipe connection objects (which are spawn-safe), so this is fine.

Auto-reset convention: when a worker's env reaches a terminal/truncated state,
the worker immediately resets internally and returns the FRESH observation
(matching the standard VecEnv convention) -- but the reward/terminated/
truncated/info from the step that actually ended the episode are still
returned accurately, so the caller can correctly log that episode's outcome
before the new one begins.
"""

import multiprocessing as mp
import numpy as np

from environment import SeafloorEDNAEnv


def _worker(remote, grid_size, max_steps, gamma):
    """Runs in a child process. Owns one persistent env instance for its
    entire lifetime. Communicates via a duplex Pipe connection."""
    env = SeafloorEDNAEnv(grid_size=grid_size, max_steps=max_steps, gamma=gamma)
    obs, info = env.reset()

    while True:
        cmd, data = remote.recv()

        if cmd == "step":
            action, curriculum_start, curriculum_ceiling = data
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                # curriculum: sample this env's NEXT episode's difficulty using
                # its own (already-seeded) rng, same distribution the original
                # single-env train.py used at each episode boundary
                env.max_hotspot_dist = float(env._rng.uniform(curriculum_start, curriculum_ceiling))
                obs, reset_info = env.reset()
            mask = env.get_action_mask()
            remote.send((obs, reward, terminated, truncated, info, mask))

        elif cmd == "reset":
            curriculum_start = data
            env.max_hotspot_dist = curriculum_start
            obs, info = env.reset()
            mask = env.get_action_mask()
            remote.send((obs, mask))

        elif cmd == "close":
            remote.close()
            break

        else:
            raise ValueError(f"Unknown command sent to worker: {cmd}")


class SubprocVecEnv:
    """Manages N worker processes, each running one SeafloorEDNAEnv.
    All public methods operate on BATCHES across all N environments at once."""

    def __init__(self, n_envs, grid_size=50, max_steps=500, gamma=0.99):
        self.n_envs = n_envs
        self.grid_size = grid_size

        ctx = mp.get_context("spawn")  # explicit -- matches Windows' only option,
                                        # tested against on Linux too so nothing
                                        # Windows-specific slips through untested
        self.remotes, worker_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for worker_remote in worker_remotes:
            p = ctx.Process(target=_worker, args=(worker_remote, grid_size, max_steps, gamma), daemon=True)
            p.start()
            self.processes.append(p)

    def reset(self, curriculum_start=10.0):
        for remote in self.remotes:
            remote.send(("reset", curriculum_start))
        results = [remote.recv() for remote in self.remotes]
        obs_list, masks = zip(*results)
        return self._stack_obs(obs_list), np.stack(masks)

    def step(self, actions, curriculum_start, curriculum_ceiling):
        """actions: array of length n_envs, one action per environment."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", (int(action), curriculum_start, curriculum_ceiling)))
        results = [remote.recv() for remote in self.remotes]
        obs_list, rewards, terminateds, truncateds, infos, masks = zip(*results)
        return (
            self._stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            list(infos),
            np.stack(masks),
        )

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)

    @staticmethod
    def _stack_obs(obs_list):
        return {
            "maps": np.stack([o["maps"] for o in obs_list]),
            "scalars": np.stack([o["scalars"] for o in obs_list]),
        }