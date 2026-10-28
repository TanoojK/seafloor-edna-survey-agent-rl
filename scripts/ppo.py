import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import ActorCritic
from buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, n_map_channels, n_scalars, n_actions=9, grid_size=50,
                 lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 value_coef=1.0, entropy_coef=0.03, n_epochs=4, batch_size=256,
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

        last_value = self._bootstrap_value(last_maps, last_scalars, last_done)
        advantages, returns = self.buffer.compute_gae(last_value, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        maps, scalars, actions, old_log_probs, action_masks = self.buffer.get_tensors()
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32)

        stats = self._run_epoch_updates(maps, scalars, actions, old_log_probs, advantages, returns, old_values, action_masks)
        self.buffer.clear()
        return stats

    def _run_epoch_updates(self, maps, scalars, actions, old_log_probs, advantages, returns, old_values, action_masks):

        n_samples = maps.shape[0]
        indices = torch.arange(n_samples)
        stats = {"entropy": [], "value_loss": [], "policy_loss": []}

        for _ in range(self.n_epochs):
            perm = indices[torch.randperm(n_samples)]
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]

                maps_b = maps[idx].to(self.device)
                scalars_b = scalars[idx].to(self.device)
                actions_b = actions[idx].to(self.device)
                old_log_probs_b = old_log_probs[idx].to(self.device)
                advantages_b = advantages[idx].to(self.device)
                returns_b = returns[idx].to(self.device)
                mask_b = action_masks[idx].to(self.device) if action_masks is not None else None

                new_log_probs, values, entropy = self.model.evaluate(maps_b, scalars_b, actions_b, mask_b)

                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((values - returns_b) ** 2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                actor_params = list(self.model.actor_encoder.parameters()) + list(self.model.actor_scalar.parameters()) + \
                    list(self.model.actor_trunk.parameters()) + list(self.model.actor_head.parameters())
                critic_params = list(self.model.critic_encoder.parameters()) + list(self.model.critic_scalar.parameters()) + \
                    list(self.model.critic_trunk.parameters()) + list(self.model.critic_head.parameters())
                nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
                nn.utils.clip_grad_norm_(critic_params, max_norm=0.5)
                self.optimizer.step()

                stats["entropy"].append(entropy.mean().item())
                stats["value_loss"].append(value_loss.item())
                stats["policy_loss"].append(policy_loss.item())

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def init_parallel_buffers(self, n_envs: int):
        """Call once, after construction, if using the N-environment methods below."""
        self.buffers = [RolloutBuffer() for _ in range(n_envs)]

    @torch.no_grad()
    def select_action_batch(self, maps_batch: torch.Tensor, scalars_batch: torch.Tensor,
                             action_masks_batch: torch.Tensor = None):

        maps_b = maps_batch.to(self.device)
        scalars_b = scalars_batch.to(self.device)
        mask_b = action_masks_batch.to(self.device) if action_masks_batch is not None else None
        actions, log_probs, values = self.model.act(maps_b, scalars_b, mask_b)
        return actions.cpu().numpy(), actions.cpu(), log_probs.cpu(), values.cpu()

    def store_transition_multi(self, env_idx, maps, scalars, action, log_prob, value, reward, done, action_mask=None):
        self.buffers[env_idx].add(maps, scalars, action, log_prob, value.item(), reward, done, action_mask)

    @torch.no_grad()
    def _bootstrap_value_batch(self, last_maps_batch: torch.Tensor, last_scalars_batch: torch.Tensor, last_dones):

        maps_b = last_maps_batch.to(self.device)
        scalars_b = last_scalars_batch.to(self.device)
        _, values = self.model.forward(maps_b, scalars_b)
        values = values.cpu().numpy()
        return [0.0 if last_dones[i] else float(values[i]) for i in range(len(last_dones))]

    def update_multi(self, last_maps_batch: torch.Tensor, last_scalars_batch: torch.Tensor, last_dones):

        last_values = self._bootstrap_value_batch(last_maps_batch, last_scalars_batch, last_dones)

        all_advantages, all_returns = [], []
        all_maps, all_scalars, all_actions, all_old_log_probs, all_masks, all_old_values = [], [], [], [], [], []

        for i, buf in enumerate(self.buffers):
            if len(buf.rewards) == 0:
                continue  # this env contributed nothing this rollout (shouldn't normally happen)
            advantages, returns = buf.compute_gae(last_values[i], self.gamma, self.lam)
            all_advantages.append(advantages)
            all_returns.append(returns)

            maps, scalars, actions, old_log_probs, action_masks = buf.get_tensors()
            all_maps.append(maps)
            all_scalars.append(scalars)
            all_actions.append(actions)
            all_old_log_probs.append(old_log_probs)
            if action_masks is not None:
                all_masks.append(action_masks)
            all_old_values.append(torch.tensor(buf.values, dtype=torch.float32))

        advantages = np.concatenate(all_advantages)
        returns = np.concatenate(all_returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        maps = torch.cat(all_maps, dim=0)
        scalars = torch.cat(all_scalars, dim=0)
        actions = torch.cat(all_actions, dim=0)
        old_log_probs = torch.cat(all_old_log_probs, dim=0)
        action_masks = torch.cat(all_masks, dim=0) if all_masks else None
        old_values = torch.cat(all_old_values, dim=0)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        stats = self._run_epoch_updates(maps, scalars, actions, old_log_probs, advantages, returns, old_values, action_masks)

        for buf in self.buffers:
            buf.clear()
        return stats