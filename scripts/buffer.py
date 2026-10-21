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
        maps = torch.stack(self.maps)
        scalars = torch.stack(self.scalars)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        if self.action_masks and self.action_masks[0] is not None:
            action_masks = torch.stack(self.action_masks)
        else:
            action_masks = None
        return maps, scalars, actions, old_log_probs, action_masks