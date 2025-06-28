import torch
import torch.nn as nn
from torch.distributions import Normal

class ValueHead(nn.Module):
    """
    Predicts the expected future cumulative reward (the value) from a given latent state.
    V(h_t) = E[sum of future rewards]
    This is the "Critic" part of the Actor-Critic algorithm.
    """
    def __init__(self, latent_state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Predict a scalar value
        )

    def forward(self, latent_state: torch.Tensor) -> Normal:
        """
        Args:
            latent_state (h_t): A state embedding from the world model.
        
        Returns:
            A Normal distribution over the predicted value, which is common practice
            in some advanced RL algorithms for more stable learning. We can just use
            its mean for a simpler implementation.
        """
        return Normal(self.network(latent_state), 1) 