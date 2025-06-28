import torch
import torch.nn as nn

class SpacetimeEncoder(nn.Module):
    """
    Encodes a low-dimensional spacetime parameter vector into the high-dimensional
    unified latent space.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, warp_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            warp_params: A tensor representing spacetime dynamics, e.g., [speed, precision].
                         Shape: (Batch, input_dim)
        
        Returns:
            An embedding in the unified latent space. Shape: (Batch, output_dim)
        """
        # Ensure input dimension matches the first Linear layer expectation
        expected_dim = self.encoder[0].in_features  # type: ignore[arg-type]
        in_dim = warp_params.shape[-1]
        if in_dim > expected_dim:
            # Truncate extra dimensions (e.g., when dataset is padded)
            warp_params = warp_params[..., :expected_dim]
        elif in_dim < expected_dim:
            # Pad with zeros if fewer dims than expected
            pad_size = expected_dim - in_dim
            pad = torch.zeros(*warp_params.shape[:-1], pad_size, device=warp_params.device, dtype=warp_params.dtype)
            warp_params = torch.cat([warp_params, pad], dim=-1)

        return self.encoder(warp_params) 