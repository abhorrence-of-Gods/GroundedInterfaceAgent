import torch
import torch.nn as nn

class RewardHead(nn.Module):
    """
    Predicts the immediate reward from a given latent state.
    r_t = R(h_t)
    """
    def __init__(self, latent_state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Predict a scalar reward
        )

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state (h_t): A state embedding from the world model.
        
        Returns:
            The predicted scalar reward for being in that state.
        """
        return self.network(latent_state) 