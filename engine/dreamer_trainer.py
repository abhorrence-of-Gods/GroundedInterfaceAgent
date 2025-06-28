import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import torch.nn as nn
from typing import Any
try:
    from safetensors.torch import save_file as _safe_save
    _HAS_SAFE = True
except ImportError:
    _HAS_SAFE = False
from contextlib import nullcontext
try:
    from torch.amp import autocast  # PyTorch 2.1+
except ImportError:
    from torch.cuda.amp import autocast

from models.gia_agent import GiaAgent
from data.dataloader import create_dataloader
from engine.trainer import calculate_comprehensive_loss

class DreamerTrainer:
    """
    The ultimate trainer for the GIA, implementing the Dream-to-Action paradigm.
    It alternates between two phases:
    1. Learning the World Model from real experience.
    2. Learning the Actor and Critic policies within the "dream" of the world model.
    """
    def __init__(self, cfg: DictConfig, model: GiaAgent):
        self.cfg = cfg
        self.model = model
        self.device = next(self.model.parameters()).device
        
        def _only_params(iterable):
            """Return only nn.Parameter objects (guards against configs sneaking in)."""
            return [p for p in iterable if isinstance(p, nn.Parameter)]

        # ---------------- Parameter grouping ----------------
        # 1) World-Model (encoders + decoders + dynamics)
        self.world_model_params = _only_params(
            list(model.language_tower.parameters()) +
            list(model.perception_tower.parameters()) +
            list(model.bridge.parameters()) +
            list(model.perception_decoder.parameters()) +
            list(model.spacetime_encoder.parameters()) +
            list(model.spacetime_decoder.parameters()) +
            list(model.action_tower.action_encoder.parameters()) +
            list(model.action_tower.action_decoder.parameters()) +
            list(model.transition_model.parameters()) +
                                  list(model.reward_head.parameters())
        )
        
        # 2) Actor-Critic (policy + value); only actor_network participates
        self.actor_critic_params = _only_params(
            list(model.action_tower.actor_network.parameters()) +
                                     list(model.value_head.parameters())
        )

        # Build param groups for LR customization
        lr_groups = cfg.training.get("lr_groups", {})
        param_groups = []
        lora_params = _only_params([p for n, p in model.named_parameters() if "lora" in n and p.requires_grad])
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr_groups.get("lora", 1e-5)})

        # default group for remaining world model params
        other_params = [p for p in self.world_model_params if p not in lora_params]
        param_groups.append({"params": other_params, "lr": lr_groups.get("bridge", cfg.training.optimizer.lr)})

        # Ensure Hydra does not wrap our param groups in DictConfig (which breaks PyTorch optimizers)
        self.wm_optimizer = hydra.utils.instantiate(
            cfg.training.optimizer,
            params=param_groups,
            _convert_="object",  # keep nested lists/dicts as plain Python objects
        )

        self.ac_optimizer = hydra.utils.instantiate(
            cfg.training.optimizer,
            params=self.actor_critic_params,
            _convert_="object",
        )
        
        self.dataloader = create_dataloader(
            batch_size=cfg.training.batch_size,
            dataset_roots=cfg.training.get("dataset_roots"),
            dataset_weights=cfg.training.get("dataset_weights"),
            expected_warp_dim=cfg.training.get("expected_warp_dim"),
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.cfg.get("log_dir", "runs/dreamer"))

        # Checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.ckpt_dir = os.path.join(self.cfg.get("checkpoint_root", "checkpoints_dreamer"), timestamp)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Mixed precision options
        use_bf16 = bool(self.cfg.get("use_bfloat16", False)) and torch.cuda.is_available()
        self._mp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Early stopping parameters
        self._best_val = float("inf")
        self._no_improve_epochs = 0
        self._patience = int(self.cfg.training.get("early_stop_patience", 10))

        # ---- Checkpoint resume ----
        self.start_epoch = 0
        ckpt_path_cfg = self.cfg.get("checkpoint_path")
        if ckpt_path_cfg is not None and os.path.isfile(ckpt_path_cfg):
            ckpt = torch.load(ckpt_path_cfg, map_location=self.device)
            self.model.load_state_dict(ckpt.get("model_state", {}), strict=False)
            self.start_epoch = ckpt.get("epoch", 0)
            self._best_val = ckpt.get("val_loss", float("inf"))
            print(f"[DreamerTrainer] Resumed from checkpoint {ckpt_path_cfg} (epoch {self.start_epoch})")

    def train(self):
        """The main training loop."""
        print("Starting Dreamer training...")
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            print(f"--- Epoch {epoch+1}/{self.cfg.training.num_epochs} ---")
            
            # --- Phase 1: Learn World Model from Real Data ---
            self.model.train()
            # In a real implementation, we would collect new data here.
            # For now, we reuse the dataloader.
            # Determine generative scaling factor
            warm_epochs = self.cfg.training.get("generative_warmup_epochs", 0)
            if warm_epochs > 0 and epoch < warm_epochs:
                gen_scale = (epoch + 1) / warm_epochs
            else:
                gen_scale = 1.0

            accum_steps = int(self.cfg.training.get("grad_accumulation_steps", 1))
            for idx, batch in enumerate(self.dataloader):
                with autocast(device_type='cuda', dtype=self._mp_dtype) if torch.cuda.is_available() else nullcontext():
                    loss = self._learn_world_model(batch, gen_scale)
                    loss = loss / accum_steps

                if (idx % accum_steps) == 0:
                self.wm_optimizer.zero_grad()

                loss.backward()

                if ((idx + 1) % accum_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(self.world_model_params, 1.0)
                self.wm_optimizer.step()

            # --- Phase 2: Learn Actor & Critic in Dreams ---
            self.model.eval()  # freeze world-model during AC updates
            initial_batch = next(iter(self.dataloader)) 

            ac_updates = int(self.cfg.training.get("ac_updates", 5))
            for _ in range(ac_updates):
                with autocast(device_type='cuda', dtype=self._mp_dtype) if torch.cuda.is_available() else nullcontext():
                    ac_loss = self._learn_actor_critic(initial_batch)
            self.ac_optimizer.zero_grad()
                ac_loss.backward()
            self.ac_optimizer.step()

            print(f"Epoch {epoch+1} finished.")

            # --- Validation / Evaluation ---
            if epoch % 1 == 0:  # every epoch
                from engine.evaluator import Evaluator
                evaluator = Evaluator(cfg=self.cfg, model=self.model)
                metrics = evaluator.evaluate()

                # Write to TensorBoard
                global_step = (epoch + 1) * len(self.dataloader)
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, global_step)

                # ---- Early stopping & best checkpoint ----
                current_val = metrics.get("val/total_loss", 0.0)
                if current_val < self._best_val:
                    self._best_val = current_val
                    self._no_improve_epochs = 0
                    best_path_pt = os.path.join(self.ckpt_dir, "best.pt")
                    state: dict[str, Any] = {
                        "model_state": self.model.state_dict(),
                        "epoch": epoch + 1,
                        "val_loss": current_val,
                    }
                    if _HAS_SAFE:
                        best_path = best_path_pt.replace('.pt', '.safetensors')
                        _safe_save(state["model_state"], best_path)
                        torch.save({k:v for k,v in state.items() if k!="model_state"}, best_path_pt)
                    else:
                        best_path = best_path_pt
                        torch.save(state, best_path)
                    print(f"[DreamerTrainer] ✨ New best val_loss={current_val:.4f}. Saved to {best_path}")
                else:
                    self._no_improve_epochs += 1
                    if self._no_improve_epochs >= self._patience:
                        print(f"[DreamerTrainer] Early stopping: no improvement for {self._patience} epochs.")
                        break
        print("Training finished.")
        self.writer.close()


    def _learn_world_model(self, batch: dict, gen_scale: float = 1.0) -> torch.Tensor:
        """
        Learns all the encoding, decoding, transition, and reward models.
        This uses the grand 19-objective loss function.
        """
        # Implement the full 19-objective loss using the helper from engine.trainer.
        print("Phase 1: Learning World Model...")

        # --- Move batch tensors to correct device/dtype ---
        batch_device = {
            "instruction_text": batch["instruction_text"],
            "screenshot": batch["screenshot"].to(self.device, dtype=torch.float32),
            "action": batch["action"].to(self.device, dtype=torch.float32),
            "warp": batch["warp"].to(self.device, dtype=torch.float32),
            "goal": batch.get("goal").to(self.device, dtype=torch.float32) if batch.get("goal") is not None else None,
        }

        # Forward pass through the agent to obtain all modality predictions
        model_outputs = self.model(
            instruction_text=batch_device["instruction_text"],
            screenshot=batch_device["screenshot"],
            target_action=batch_device["action"],
            target_warp=batch_device["warp"],
            target_goal=batch_device["goal"],
        )

        # Scale generative loss weights on the fly (shallow copy)
        loss_weights_scaled = self.cfg.training.loss_weights.copy()
        for k in loss_weights_scaled:
            if k.startswith("generative_loss_"):
                loss_weights_scaled[k] = loss_weights_scaled[k] * gen_scale

        # Compute the comprehensive loss (19 objectives)
        loss, _ = calculate_comprehensive_loss(
            model_outputs=model_outputs,
            batch=batch_device,
            loss_weights=loss_weights_scaled,
            logit_scale=self.model.bridge.logit_scale,
            device=self.device,
        )

        # Uncertainty loss
        latent_state = self.model.encode_state(batch_device["instruction_text"], batch_device["screenshot"], batch_device["goal"])
        sigma_u = self.model.uncert_head(latent_state)
        gamma = 0.99
        with torch.no_grad():
            reward_pred = self.model.reward_head(latent_state)
            value_curr = self.model.value_head(latent_state).mean
            # quick next state prediction using zero action
            latent_zero = torch.zeros((latent_state.shape[0], self.model.action_tower.latent_action_dim), device=self.device)
            next_latent = self.model.transition_model(latent_state, latent_zero)
            value_next = self.model.value_head(next_latent).mean * gamma
            td_error = torch.abs(reward_pred + value_next - value_curr)
        l_uncert = torch.nn.functional.mse_loss(sigma_u, td_error)
        w_uncert = self.cfg.training.loss_weights.get("uncertainty_loss", 0.0)
        total_loss = loss + w_uncert * l_uncert
        return total_loss


    def _learn_actor_critic(self, initial_batch: dict) -> torch.Tensor:
        """
        Learns the actor (ActionTower) and critic (ValueHead) from imagined trajectories.
        """
        print("Phase 2: Learning Actor-Critic in Dream...")
        
        # 1. Get a starting state from real data
        instruction = initial_batch["instruction_text"]
        screenshot = initial_batch["screenshot"].to(self.device)
        with torch.no_grad():
            initial_state = self.model.encode_state(instruction, screenshot, initial_batch.get("goal"))

        # 2. Imagine the future from this state (detached so WM weights frozen)
        states, actions, rewards, values = self.model.plan_in_dream(initial_state.detach(), goal_vec=initial_batch.get("goal").to(self.device) if initial_batch.get("goal") is not None else None)

        # states: (H, B, D)  actions: (H, B, A)  rewards/values: (H, B, 1)

        # 3. λ-returns as advantage target
        lambda_returns = self._compute_lambda_returns(rewards, values)

        # 4. Actor loss with optional entropy regularisation
        H, B, _ = actions.shape
        states_flat = states.view(H * B, -1)
        actions_flat = actions.view(H * B, -1)

        action_dist = self.model.action_tower.get_action_dist(states_flat)
        log_prob = action_dist.log_prob(actions_flat).sum(-1)  # (H*B,)
        log_prob = log_prob.view(H, B)

        advantage = (lambda_returns - values.detach()).squeeze(-1)  # (H,B)
        entropy = action_dist.entropy().sum(-1).view(H, B)  # (H,B)
        beta = float(self.cfg.training.get("entropy_coeff", 0.0))
        actor_loss = (-(log_prob * advantage) - beta * entropy).mean()

        # ---- GoalWarp Jacobian regularisation (optional) ----
        gw_coeff = float(self.cfg.training.loss_weights.get("goalwarp_logdet", 0.0))
        if gw_coeff > 0 and "goal" in initial_batch and initial_batch["goal"] is not None:
            gvec = initial_batch["goal"].to(self.device)
            # compute log|detJ| for each initial state (detached to keep AC grads separate)
            _, gld = self.model.goal_warp(initial_state.detach(), gvec, return_logdet=True)
            actor_loss = actor_loss + gw_coeff * gld.mean()

        # 5. Critic loss (value function)
        value_pred = values.squeeze(-1)  # (H,B)
        critic_loss = torch.nn.functional.mse_loss(value_pred, lambda_returns.squeeze(-1).detach())

        total_ac_loss = actor_loss + critic_loss
        return total_ac_loss

    def _compute_lambda_returns(self, rewards, values, gamma=0.99, lambda_=0.95):
        """Helper to compute lambda returns."""
        # rewards / values : (H, B, 1)
        H = rewards.shape[0]
        device = rewards.device
        returns = torch.zeros_like(rewards, device=device)
        next_value = values[-1]
        for t in reversed(range(H)):
            next_value = rewards[t] + gamma * ((1 - lambda_) * values[t] + lambda_ * next_value)
            returns[t] = next_value
        return returns 