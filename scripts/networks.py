import torch
import torch.nn as nn
from torch.distributions import Categorical


class CNNEncoder(nn.Module):

    def __init__(self, in_channels: int, grid_size: int = 50, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 50x50 -> 25x25
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # 25x25 -> 13x13
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),           # 13x13 -> 7x7
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size, grid_size)
            flat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        x = self.conv(maps)
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))


class ActorCritic(nn.Module):
    
    def __init__(self, n_map_channels: int, n_scalars: int, n_actions: int = 9,
                 grid_size: int = 50, cnn_out_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.actor_encoder = CNNEncoder(n_map_channels, grid_size, cnn_out_dim)
        self.critic_encoder = CNNEncoder(n_map_channels, grid_size, cnn_out_dim)

        self.actor_scalar = nn.Sequential(nn.Linear(n_scalars, 64), nn.ReLU())
        self.critic_scalar = nn.Sequential(nn.Linear(n_scalars, 64), nn.ReLU())
        combined_dim = cnn_out_dim + 64
        self.actor_trunk = nn.Sequential(nn.Linear(combined_dim, hidden_dim), nn.ReLU())
        self.critic_trunk = nn.Sequential(nn.Linear(combined_dim, hidden_dim), nn.ReLU())

        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor, action_mask: torch.Tensor = None):
        """Returns action distribution and state value estimate.
        action_mask (optional): (B, n_actions) bool, True=allowed. Masked-out
        actions get -inf logits so Categorical assigns them ~0 probability."""
        actor_features = self.actor_trunk(torch.cat([self.actor_encoder(maps), self.actor_scalar(scalars)], dim=-1))
        critic_features = self.critic_trunk(torch.cat([self.critic_encoder(maps), self.critic_scalar(scalars)], dim=-1))

        logits = self.actor_head(actor_features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        value = self.critic_head(critic_features)
        dist = Categorical(logits=logits)
        return dist, value.squeeze(-1)

    def act(self, maps: torch.Tensor, scalars: torch.Tensor, action_mask: torch.Tensor = None):
        """Sample an action for environment interaction. Returns action, log_prob, value."""
        dist, value = self.forward(maps, scalars, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, maps: torch.Tensor, scalars: torch.Tensor, actions: torch.Tensor,
                 action_mask: torch.Tensor = None):

        dist, value = self.forward(maps, scalars, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy