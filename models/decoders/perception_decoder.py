import torch
import torch.nn as nn

class PerceptionDecoder(nn.Module):
    """
    Decodes a latent embedding back into an image.
    This acts as the "imagination" of the agent, allowing it to generate
    a visual representation from a language or action concept.
    
    This is a placeholder implementation using a simple transposed convolutional network.
    """
    def __init__(self, input_dim: int, output_channels: int = 3):
        super().__init__()
        self.input_dim = input_dim
        
        self.decoder = nn.Sequential(
            # Normalize the input embedding for stability
            nn.LayerNorm(input_dim),
            # Start from the embedding and project it to a small spatial volume
            nn.Linear(input_dim, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            
            # Upsample to 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Upsample to 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Upsample to 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Upsample to 224x224
            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Output pixel values between 0 and 1
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Generates an image from a given embedding.

        Args:
            embedding: A latent vector of shape (Batch, input_dim).

        Returns:
            A generated image tensor of shape (Batch, C, H, W).
        """
        return self.decoder(embedding) 