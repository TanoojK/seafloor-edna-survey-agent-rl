"""
Rollout buffer: stores a batch of experience (states, actions, rewards, etc.)
collected by running the current policy, then computes advantages via GAE
(Generalized Advantage Estimation) once the batch is complete.

Why GAE instead of raw returns?
Using raw discounted rewards as the training signal is high-variance (noisy).
Using only the value network's estimate is low-variance but biased.
GAE blends the two with a parameter `lambda`, trading off bias vs variance --
lambda=1 is closer to raw returns (high variance), lambda=0 is closer to pure
value-function bootstrapping (low variance, more biased). 0.95 is a common default.
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.maps = []
        self.scalars = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []

    def add(self, maps, scalars, action, log_prob, value, reward, done, action_mask=None):
        self.maps.append(maps)
        self.scalars.append(scalars)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        self.__init__()

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """
        Computes advantages and returns for every step in the buffer.
        `last_value` is the value network's estimate of the state AFTER the final
        stored step (needed to bootstrap the last advantage if the episode didn't
        end naturally, i.e. was cut off by a max-steps limit rather than termination).
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            # if done, there's no "next state" contribution -- mask it out
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def get_tensors(self):
        """Returns CPU tensors -- deliberately NOT moved to GPU here. The full
        buffer (especially with multiple parallel environments) can be far
        larger than any single minibatch actually needs on the GPU at once
        (e.g. 8 envs x 2048 steps = 16,384 samples resident simultaneously,
        vs. the ~64 actually used per forward/backward pass) -- on a memory-
        constrained GPU this alone can cause an out-of-memory error. Moving
        data to the GPU is now the caller's job, done per-minibatch instead."""
        maps = torch.stack(self.maps)
        scalars = torch.stack(self.scalars)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        if self.action_masks and self.action_masks[0] is not None:
            action_masks = torch.stack(self.action_masks)
        else:
            action_masks = None
        return maps, scalars, actions, old_log_probs, action_masks