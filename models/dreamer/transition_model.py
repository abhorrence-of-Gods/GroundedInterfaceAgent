import torch
import torch.nn as nn

class TransitionModel(nn.Module):
    """
    Predicts the next latent state given the current latent state and a latent action.
    This is the core of the "dream" physics engine.
    h_t+1 = T(h_t, a_t)
    """
    def __init__(self, latent_state_dim: int, latent_action_dim: int, hidden_dim: int = 512):
        super().__init__()
        # A simple MLP for the transition dynamics.
        # More advanced versions could use GRUs or Transformers for sequence memory.
        self.network = nn.Sequential(
            nn.Linear(latent_state_dim + latent_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_state_dim)
        )

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_state (h_t): The current state embedding. Shape: (Batch, latent_state_dim)
            latent_action (a_t): The action embedding. Shape: (Batch, latent_action_dim)
        
        Returns:
            The predicted next latent state (h_t+1). Shape: (Batch, latent_state_dim)
        """
        combined_input = torch.cat([latent_state, latent_action], dim=-1)
        return self.network(combined_input) 