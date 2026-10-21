import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing as mp
import numpy as np

from environment import SeafloorEDNAEnv


def _worker(remote, grid_size, max_steps, gamma):

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

    def __init__(self, n_envs, grid_size=50, max_steps=300, gamma=0.99):
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