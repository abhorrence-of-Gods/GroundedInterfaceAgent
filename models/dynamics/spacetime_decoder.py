import torch
import torch.nn as nn

class SpacetimeDecoder(nn.Module):
    """
    Decodes an embedding from the unified latent space back into a low-dimensional
    spacetime parameter vector.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid() # Constrain outputs to a predictable range (e.g., 0 to 1)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: An embedding from the unified latent space.
                       Shape: (Batch, input_dim)
        
        Returns:
            A tensor of predicted spacetime parameters. Shape: (Batch, output_dim)
        """
        return self.decoder(embedding) 