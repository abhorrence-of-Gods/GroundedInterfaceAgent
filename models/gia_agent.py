import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra
import torch.utils.checkpoint as cp

# Import all components
from .towers.language_tower import LanguageTower
from .towers.perception_tower import PerceptionTower
from .towers.action_tower import ActionTower
from .bridge.grounding_bridge import GroundingBridge
from .decoders.perception_decoder import PerceptionDecoder
from .dynamics.spacetime_encoder import SpacetimeEncoder
from .dynamics.spacetime_decoder import SpacetimeDecoder
from .dreamer.transition_model import TransitionModel
from .dreamer.reward_head import RewardHead
from .dreamer.value_head import ValueHead
from .warp_modules import TimeWarp, SpaceWarp, GoalWarp
from .warp_modules import UncertaintyHead

class GiaAgent(nn.Module):
    """
    The GIA Agent, re-architected for the Dream-to-Action paradigm.
    It combines the world model (towers, bridge) with the dreamer components.
    """
    def __init__(self, model_config: DictConfig):
        super().__init__()
        # World Model Components (Encoders/Decoders)
        self.language_tower: LanguageTower = hydra.utils.instantiate(model_config.language_tower)
        self.perception_tower: PerceptionTower = hydra.utils.instantiate(model_config.perception_tower)
        self.action_tower: ActionTower = hydra.utils.instantiate(model_config.action_tower)
        self.bridge: GroundingBridge = hydra.utils.instantiate(model_config.bridge)
        self.spacetime_encoder: SpacetimeEncoder = hydra.utils.instantiate(model_config.spacetime_encoder)
        self.perception_decoder: PerceptionDecoder = hydra.utils.instantiate(model_config.perception_decoder)
        self.spacetime_decoder: SpacetimeDecoder = hydra.utils.instantiate(model_config.spacetime_decoder)

        # Dreamer Components (Actor, Critic, and World Dynamics)
        self.transition_model: TransitionModel = hydra.utils.instantiate(model_config.transition_model)
        self.reward_head: RewardHead = hydra.utils.instantiate(model_config.reward_head)
        self.value_head: ValueHead = hydra.utils.instantiate(model_config.value_head)

        # Project latent-action (256d) into common 4096-d latent space so that
        # contrastive & generative heads share dimensionality.
        self.action_projection = nn.Linear(self.action_tower.latent_action_dim, 4096)

        # Ensure all critical modules are on the same device as the language tower
        self.device = next(self.language_tower.parameters()).device
        self.bridge.to(self.device)
        self.perception_tower.to(self.device)
        self.action_tower.to(self.device)
        self.spacetime_encoder.to(self.device)
        self.spacetime_decoder.to(self.device)
        self.transition_model.to(self.device)
        self.reward_head.to(self.device)
        self.value_head.to(self.device)
        self.perception_decoder.to(self.device)
        self.action_projection.to(self.device)

        # --- Warp modules (initially identity) ---
        self.time_warp = TimeWarp().to(self.device)
        self.space_warp = SpaceWarp().to(self.device)
        self.uncert_head = UncertaintyHead().to(self.device)
        self.goal_warp = GoalWarp(goal_dim=16).to(self.device)

        # Track default precision of the model (required by mixed-precision trainer)
        self.dtype = next(self.parameters()).dtype

        # Gradient checkpoint flag (memory vs compute)
        self._use_grad_ckpt = bool(model_config.get("use_grad_checkpoint", False)) or bool(getattr(model_config, "use_grad_checkpoint", False))

    def encode_state(self, instruction_text: list[str], screenshot: torch.Tensor, goal_vec: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encodes the raw observation (vision + language) into a unified latent state `h_t`.
        This is the entry point into the "dream".
        """
        if self._use_grad_ckpt:
            instruction_embedding = cp.checkpoint(lambda txt: self.language_tower(txt), instruction_text)
            vision_features = cp.checkpoint(lambda img: self.perception_tower(img), screenshot)
        else:
            instruction_embedding = self.language_tower(instruction_text)
            vision_features = self.perception_tower(screenshot)

        # Uncertainty-based TimeWarp using sigma from vision features
        B = vision_features.shape[0]
        sigma_u = self.uncert_head(vision_features.detach())  # detach to avoid feedback loop in this path
        t_vec = torch.zeros((B, 1), device=self.device, dtype=vision_features.dtype)
        alpha = getattr(self, "alpha_timewarp", 0.0)
        vision_features = vision_features + self.time_warp(t_vec + alpha * sigma_u)

        bridge_outputs = cp.checkpoint(lambda a,b: self.bridge(a,b), instruction_embedding, vision_features) if self._use_grad_ckpt else self.bridge(instruction_embedding, vision_features)
        
        # The fused embedding is our latent state representation `h_t`
        latent_state = bridge_outputs["fused_embedding"]

        if goal_vec is not None:
            goal_vec = goal_vec.to(latent_state.device, dtype=latent_state.dtype)
            latent_state = self.goal_warp(latent_state, goal_vec)

        return latent_state

    def plan_in_dream(self, initial_state: torch.Tensor, horizon: int = 15, goal_vec: torch.Tensor | None = None):
        """
        Imagines a sequence of states, actions, and rewards in the latent space.
        """
        latent_states = []
        latent_actions = []
        current_state = initial_state

        for _ in range(horizon):
            state_for_policy = current_state
            if goal_vec is not None:
                goal_vec = goal_vec.to(state_for_policy.device, dtype=state_for_policy.dtype)
                state_for_policy = self.goal_warp(state_for_policy, goal_vec)
            action_dist = self.action_tower.get_action_dist(state_for_policy)
            latent_action = action_dist.rsample()
            latent_states.append(current_state)
            latent_actions.append(latent_action)
            next_state = self.transition_model(current_state, latent_action)
            if goal_vec is not None:
                goal_vec = goal_vec.to(next_state.device, dtype=next_state.dtype)
                next_state = self.goal_warp(next_state, goal_vec)
            current_state = next_state

        latent_states = torch.stack(latent_states)
        latent_actions = torch.stack(latent_actions)

        # Compute imagined rewards/values
        if goal_vec is not None:
            if latent_states.dim() == 3:
                H,B,D = latent_states.shape
                goal_vec_exp = goal_vec.unsqueeze(0).expand(H, -1, -1) if goal_vec.dim()==2 else goal_vec
                flat_states = latent_states.reshape(H*B, D)
                flat_goal = goal_vec_exp.reshape(H*B, -1)
                warped_flat = self.goal_warp(flat_states, flat_goal)
                _states_for_value = warped_flat.reshape(H,B,D)
            else:
                _states_for_value = self.goal_warp(latent_states, goal_vec)
        else:
            _states_for_value = latent_states

        imagined_rewards = self.reward_head(_states_for_value)
        imagined_values = self.value_head(_states_for_value).mean

        return latent_states, latent_actions, imagined_rewards, imagined_values

    def forward(self, instruction_text: list[str], screenshot: torch.Tensor, target_action: torch.Tensor, target_warp: torch.Tensor, target_goal: torch.Tensor | None = None):
        """
        A grand forward pass that computes all embeddings and generates all cross-modal predictions.
        """
        # 1. ENCODING: Get embeddings for all four input modalities
        if self._use_grad_ckpt:
            vision_features = cp.checkpoint(lambda img: self.perception_tower(img), screenshot)
        else:
            vision_features = self.perception_tower(screenshot)

        # Space warp: use target_action coords
        coords = target_action[:, :2]  # assume normalized coords in [0,1]
        coord_embed = self.space_warp(coords)
        vision_features = vision_features + coord_embed

        if self._use_grad_ckpt:
            instruction_embedding = cp.checkpoint(lambda txt: self.language_tower(txt), instruction_text)
        else:
            instruction_embedding = self.language_tower(instruction_text)
        action_embedding = self.action_tower.encode_action(target_action)
        # Align with 4096-d common space
        action_projected = self.action_projection(action_embedding)
        warp_embedding = self.spacetime_encoder(target_warp)

        # 2. BRIDGING & FUSION: For the primary imitation task (P,L -> A)
        bridge_outputs = cp.checkpoint(lambda a,b: self.bridge(a,b), instruction_embedding, vision_features) if self._use_grad_ckpt else self.bridge(instruction_embedding, vision_features)
        fused_embedding = bridge_outputs["fused_embedding"]
        
        # --- Goal Warp (RealNVP) ---
        if target_goal is not None:
            target_goal = target_goal.to(fused_embedding.device, dtype=fused_embedding.dtype)
            fused_embedding, goal_logdet = self.goal_warp(fused_embedding, target_goal, return_logdet=True)
        else:
            goal_logdet = None

        # 3. IMITATION: Predict the action from the primary fused context
        predicted_action_from_pl = self.action_tower(fused_embedding)

        # 4. GENERATION: The full 12-way generation
        projected_vision = bridge_outputs["projected_vision"]
        projected_language = bridge_outputs["projected_language"]

        # Generate from Perception
        pred_action_from_p = self.action_tower(projected_vision)
        pred_warp_from_p = self.spacetime_decoder(projected_vision)

        # Generate from Language
        pred_action_from_l = self.action_tower(projected_language)
        pred_image_from_l = self.perception_decoder(projected_language)
        pred_warp_from_l = self.spacetime_decoder(projected_language)
        
        # Generate from Action
        pred_image_from_a = self.perception_decoder(action_projected)
        pred_warp_from_a = self.spacetime_decoder(action_projected)

        # Generate from Warp
        pred_image_from_w = self.perception_decoder(warp_embedding)
        pred_action_from_w = self.action_tower(warp_embedding)

        # 5. Return everything for the grand loss function
        return {
            "predicted_action": predicted_action_from_pl,
            "projected_vision": projected_vision,
            "projected_language": projected_language,
            "action_embedding": action_projected,
            "warp_embedding": warp_embedding,
            # Generated images
            "pred_image_from_l": pred_image_from_l,
            "pred_image_from_a": pred_image_from_a,
            "pred_image_from_w": pred_image_from_w,
            # Generated actions
            "pred_action_from_p": pred_action_from_p,
            "pred_action_from_l": pred_action_from_l,
            "pred_action_from_w": pred_action_from_w,
            # Generated warps
            "pred_warp_from_p": pred_warp_from_p,
            "pred_warp_from_l": pred_warp_from_l,
            "pred_warp_from_a": pred_warp_from_a,
            "coord_embed": coord_embed,
            "goal_logdet": goal_logdet,
        }