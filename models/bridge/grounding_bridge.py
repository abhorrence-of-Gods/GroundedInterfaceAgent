import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class GroundingBridge(nn.Module):
    """
    The Grounding Bridge is the central nervous system of the GIA.
    It connects the three towers, enabling communication and grounding.
    """
    def __init__(self, type: str, num_layers: int, hidden_dim: int):
        super().__init__()
        self.type = type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Projection layers to bring all modalities to the same dimension
        self.vision_projection = nn.Linear(4096, hidden_dim)
        self.language_projection = nn.Linear(4096, hidden_dim)

        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, language_features: torch.Tensor, vision_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Projects features to a common space and fuses them. Includes safety clamping.

        Returns:
            A dictionary containing projected features for contrastive loss
            and the fused embedding for the action tower.
        """
        # Ensure dtype compatibility between incoming features and projection layers
        projection_dtype = self.vision_projection.weight.dtype  # both projections share dtype
        if vision_features.dtype != projection_dtype:
            vision_features = vision_features.to(projection_dtype)
        if language_features.dtype != projection_dtype:
            language_features = language_features.to(projection_dtype)

        # Clamp the logit_scale to prevent overflow/instability, a common practice in contrastive learning.
        self.logit_scale.data.clamp_(max=np.log(100))

        # 1. Project features to the common embedding space
        projected_vision = self.vision_projection(vision_features)
        projected_language = self.language_projection(language_features)

        # 2. Fuse the projected features using cross-attention
        lang_q = projected_language.unsqueeze(1)
        vis_kv = projected_vision.unsqueeze(1)
        
        attn_output, _ = self.cross_attention_layers[0](
            query=self.ln_1(lang_q),
            key=self.ln_2(vis_kv),
            value=vis_kv
        )
        fused_embedding = attn_output.squeeze(1)
        
        # 3. Return all necessary tensors as anticipated by your agent/trainer modifications
        return {
            "fused_embedding": fused_embedding,
            "projected_vision": projected_vision,
            "projected_language": projected_language,
        }
        
    def reflex_arc(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        A conceptual placeholder for the reflex arc.
        If a vision feature is highly indicative of a specific action
        (e.g., a very "button-like" element), this would generate a
        direct action proposal.
        """
        # In a real system, this could be a small, fast network that maps
        # vision features directly to action primitives.
        pass 