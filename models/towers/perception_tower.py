import torch
import torch.nn as nn
from PIL import Image

class PerceptionTower(nn.Module):
    """
    The Perception Tower serves as the digital eye of the GIA.
    It takes raw sensor data (i.e., screenshots) and encodes it into
    meaningful feature representations (perceptual primitives) of UI elements.
    
    This is a placeholder for a sophisticated model like UI-TARS.
    """
    def __init__(self, model_name: str, is_frozen: bool):
        super().__init__()
        self.model_name = model_name
        self.is_frozen = is_frozen

        # In a real implementation, this would load the UI-TARS vision encoder.
        # As a placeholder, we'll use a simple convolutional network.
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Project the simple CNN features to the expected embedding dimension
        self.projection_layer = nn.Linear(32, 4096)
        
        # Placeholder for output feature dimension
        self.output_dim = 4096

        if self.is_frozen:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

    def forward(self, screenshot_batch: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of screenshots into a set of perceptual primitives.

        Args:
            screenshot_batch: A tensor of screenshots with shape (B, C, H, W).

        Returns:
            A tensor representing the global features for each image in the batch.
            The shape will be (B, feature_dim).
        """
        # This is a highly simplified placeholder. A real implementation would:
        # 1. Use the UI-TARS model to perform object detection/segmentation.
        # 2. Extract feature vectors for each detected UI element per image.
        # For now, we'll just get a single global feature vector for each image.

        # Ensure input is on same device/dtype as the vision encoder parameters
        target_device = next(self.vision_encoder.parameters()).device
        screenshot_batch = screenshot_batch.to(target_device, dtype=torch.float32)

        with torch.no_grad() if self.is_frozen else torch.enable_grad():
            cnn_features = self.vision_encoder(screenshot_batch)
            features = self.projection_layer(cnn_features)
        
        return features 