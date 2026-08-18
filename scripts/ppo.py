"""
The PPO algorithm itself.

Core idea (the "proximal" part): when we update the policy using collected experience,
we don't want to change it too drastically in one step -- a single bad batch could
otherwise destroy an already-decent policy. PPO enforces this by computing the ratio
between the new policy's probability of an action and the old policy's probability of
that same action, then CLIPPING that ratio to a small range (e.g. [0.8, 1.2] for
epsilon=0.2). This means: "improve the policy based on this experience, but don't
trust any single batch enough to change behavior by more than ~20% at a time."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import ActorCritic
from buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, n_map_channels, n_scalars, n_actions=9, grid_size=50,
                 lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 value_coef=0.5, entropy_coef=0.03, n_epochs=4, batch_size=64,
                 device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(n_map_channels, n_scalars, n_actions, grid_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def select_action(self, maps: torch.Tensor, scalars: torch.Tensor, action_mask: torch.Tensor = None):
        """
        maps: (C, 50, 50) single observation, scalars: (N,) single observation.
        action_mask: (n_actions,) bool, True=allowed (optional).
        Returns the action to take plus the log_prob/value needed for the buffer.
        """
        maps_b = maps.unsqueeze(0).to(self.device)
        scalars_b = scalars.unsqueeze(0).to(self.device)
        mask_b = action_mask.unsqueeze(0).to(self.device) if action_mask is not None else None
        action, log_prob, value = self.model.act(maps_b, scalars_b, mask_b)
        return action.item(), action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def store_transition(self, maps, scalars, action, log_prob, value, reward, done, action_mask=None):
        self.buffer.add(maps, scalars, action, log_prob, value.item(), reward, done, action_mask)

    @torch.no_grad()
    def _bootstrap_value(self, last_maps, last_scalars, last_done):
        """Value estimate for the state after the final stored transition (0 if episode ended)."""
        if last_done:
            return 0.0
        maps_b = last_maps.unsqueeze(0).to(self.device)
        scalars_b = last_scalars.unsqueeze(0).to(self.device)
        _, value = self.model.forward(maps_b, scalars_b)
        return value.item()

    def update(self, last_maps, last_scalars, last_done):
        """Run PPO update using everything currently in the buffer, then clear it.
        Returns a dict of diagnostic stats (entropy, value_loss, policy_loss) averaged
        over the update -- use these to actually SEE what's happening during training
        rather than inferring it indirectly from episode outcomes."""
        last_value = self._bootstrap_value(last_maps, last_scalars, last_done)
        advantages, returns = self.buffer.compute_gae(last_value, self.gamma, self.lam)

        # normalize advantages -- standard trick, stabilizes training a lot in practice
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        maps, scalars, actions, old_log_probs, action_masks = self.buffer.get_tensors(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)

        n_samples = maps.shape[0]
        indices = torch.arange(n_samples)

        stats = {"entropy": [], "value_loss": [], "policy_loss": []}

        for _ in range(self.n_epochs):
            perm = indices[torch.randperm(n_samples)]
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]

                new_log_probs, values, entropy = self.model.evaluate(
                    maps[idx], scalars[idx], actions[idx],
                    action_masks[idx] if action_masks is not None else None
                )

                # the core PPO ratio + clipping
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss WITH clipping (from the original PPO paper) -- without this,
                # a single large value-prediction error can produce an oversized gradient
                # that (through corrupt shared features, or just noisy critic learning)
                # destabilizes the policy. Clipping bounds how much the value estimate
                # is allowed to move in one update, same spirit as the policy ratio clip.
                values_clipped = old_values[idx] + torch.clamp(
                    values - old_values[idx], -self.clip_eps, self.clip_eps
                )
                value_loss_unclipped = (values - returns[idx]) ** 2
                value_loss_clipped = (values_clipped - returns[idx]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropy.mean()  # negative because we want to MAXIMIZE entropy (encourage exploration)

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                stats["entropy"].append(entropy.mean().item())
                stats["value_loss"].append(value_loss.item())
                stats["policy_loss"].append(policy_loss.item())

        self.buffer.clear()
        return {k: float(np.mean(v)) for k, v in stats.items()}