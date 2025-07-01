import torch
import torch.nn.functional as F
import hydra
import inspect
import os
from datetime import datetime
from omegaconf import DictConfig
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
import time
try:
    # torch >= 2.1 推奨 API
    from torch.amp import autocast  # type: ignore
except ImportError:  # fallback for older PyTorch versions
    from torch.cuda.amp import autocast
from models.gia_agent import GiaAgent
from data.dataloader import create_dataloader
from engine.metrics import action_mse, click_accuracy
import torch.nn as nn
from typing import Any
try:
    from safetensors.torch import save_file as _safe_save
    _HAS_SAFE = True
except ImportError:
    _HAS_SAFE = False

def _calculate_contrastive_loss(features_a: torch.Tensor, features_b: torch.Tensor, logit_scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Helper function to compute a symmetric contrastive loss with improved numerical stability."""
    # Normalise with a small epsilon to prevent divide-by-zero when the vector norm is zero.
    features_a = F.normalize(features_a, p=2, dim=1, eps=1e-6)
    features_b = F.normalize(features_b, p=2, dim=1, eps=1e-6)

    scaled_logits = logit_scale * features_a @ features_b.t()
    
    batch_size = features_a.shape[0]
    labels = torch.arange(batch_size, device=device)

    loss = (F.cross_entropy(scaled_logits, labels) +
            F.cross_entropy(scaled_logits.t(), labels)) / 2
    return loss

def calculate_comprehensive_loss(model_outputs: dict, batch: dict, loss_weights: DictConfig, logit_scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Calculates the total, comprehensive loss for a given batch.
    This is a standalone function to avoid any potential issues with class method resolution in complex environments.
    """
    target_action = batch["action"].to(device)
    target_image = batch["screenshot"].to(device)
    target_warp = batch.get("warp")
    if target_warp is not None:
        target_warp = target_warp.to(device)
    w = loss_weights
    
    total_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    # Loss 1: Imitation Learning
    if w.action_imitation_loss > 0:
        imitation_loss = F.mse_loss(model_outputs["predicted_action"], target_action)
        total_loss += w.action_imitation_loss * imitation_loss
        loss_dict["Imitate"] = imitation_loss.item()

    # Loss 2: Three-Way Contrastive Loss
    logit_scale_exp = logit_scale.exp()
    if w.contrastive_loss_pl > 0:
        loss_pl = _calculate_contrastive_loss(model_outputs["projected_vision"], model_outputs["projected_language"], logit_scale_exp, device)
        total_loss += w.contrastive_loss_pl * loss_pl
        loss_dict["C(PL)"] = loss_pl.item()
    if w.contrastive_loss_pa > 0:
        loss_pa = _calculate_contrastive_loss(model_outputs["projected_vision"], model_outputs["action_embedding"], logit_scale_exp, device)
        total_loss += w.contrastive_loss_pa * loss_pa
        loss_dict["C(PA)"] = loss_pa.item()
    if "contrastive_loss_la" in w and w.contrastive_loss_la > 0:
        loss_la = _calculate_contrastive_loss(model_outputs["action_embedding"], model_outputs["projected_language"], logit_scale_exp, device)
        total_loss += w.contrastive_loss_la * loss_la
        loss_dict["C(AL)"] = loss_la.item()
    
    # Loss 3: Six-Way Generative Loss (robust to NaNs)
    def _safe_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Robust MSE that tolerates NaNs and mis-matched feature lengths."""
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)

        # Align feature dimension (last dim) if they differ, e.g. pred dim 2 vs target dim 8.
        if pred.shape != target.shape:
            min_dim = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_dim]
            target = target[..., :min_dim]
        return F.mse_loss(pred, target)

    if w.generative_loss_p_from_l > 0:
        gen_p_from_l = _safe_mse(model_outputs["pred_image_from_l"], target_image)
        total_loss += w.generative_loss_p_from_l * gen_p_from_l
        loss_dict["G(P<-L)"] = gen_p_from_l.item()
    if w.generative_loss_p_from_a > 0:
        gen_p_from_a = _safe_mse(model_outputs["pred_image_from_a"], target_image)
        total_loss += w.generative_loss_p_from_a * gen_p_from_a
        loss_dict["G(P<-A)"] = gen_p_from_a.item()
    if w.generative_loss_p_from_w > 0:
        gen_p_from_w = _safe_mse(model_outputs["pred_image_from_w"], target_image)
        total_loss += w.generative_loss_p_from_w * gen_p_from_w
        loss_dict["G(P<-W)"] = gen_p_from_w.item()

    if w.generative_loss_a_from_p > 0:
        gen_a_from_p = _safe_mse(model_outputs["pred_action_from_p"], target_action)
        total_loss += w.generative_loss_a_from_p * gen_a_from_p
        loss_dict["G(A<-P)"] = gen_a_from_p.item()
    if w.generative_loss_a_from_l > 0:
        gen_a_from_l = _safe_mse(model_outputs["pred_action_from_l"], target_action)
        total_loss += w.generative_loss_a_from_l * gen_a_from_l
        loss_dict["G(A<-L)"] = gen_a_from_l.item()
    if w.generative_loss_a_from_w > 0:
        gen_a_from_w = _safe_mse(model_outputs["pred_action_from_w"], target_action)
        total_loss += w.generative_loss_a_from_w * gen_a_from_w
        loss_dict["G(A<-W)"] = gen_a_from_w.item()

    if w.generative_loss_w_from_p > 0:
        gen_w_from_p = _safe_mse(model_outputs["pred_warp_from_p"], target_warp)
        total_loss += w.generative_loss_w_from_p * gen_w_from_p
        loss_dict["G(W<-P)"] = gen_w_from_p.item()
    if w.generative_loss_w_from_l > 0:
        gen_w_from_l = _safe_mse(model_outputs["pred_warp_from_l"], target_warp)
        total_loss += w.generative_loss_w_from_l * gen_w_from_l
        loss_dict["G(W<-L)"] = gen_w_from_l.item()
    if w.generative_loss_w_from_a > 0:
        gen_w_from_a = _safe_mse(model_outputs["pred_warp_from_a"], target_warp)
        total_loss += w.generative_loss_w_from_a * gen_w_from_a
        loss_dict["G(W<-A)"] = gen_w_from_a.item()
    
    # ----- Hard-negative margin losses -----
    def _hard_neg_loss(a: torch.Tensor, b: torch.Tensor, margin: float = 0.2):
        a_n = F.normalize(a, p=2, dim=1, eps=1e-6)
        b_n = F.normalize(b, p=2, dim=1, eps=1e-6)
        sim = a_n @ b_n.T  # (B,B)
        pos = sim.diag()  # (B,)
        # mask diagonal to -inf to find hardest negative
        sim_neg = sim.clone()
        batch_size = sim.shape[0]
        sim_neg[torch.arange(batch_size), torch.arange(batch_size)] = -1e4
        hardest_neg, _ = sim_neg.max(dim=1)
        loss_hn = F.relu(margin + hardest_neg - pos).mean()
        return loss_hn

    if getattr(w, "contrastive_hn_loss_pl", 0) > 0:
        hn_pl = _hard_neg_loss(model_outputs["projected_vision"], model_outputs["projected_language"])
        total_loss += w.contrastive_hn_loss_pl * hn_pl
        loss_dict["HN(PL)"] = hn_pl.item()
    if getattr(w, "contrastive_hn_loss_pa", 0) > 0:
        hn_pa = _hard_neg_loss(model_outputs["projected_vision"], model_outputs["action_embedding"])
        total_loss += w.contrastive_hn_loss_pa * hn_pa
        loss_dict["HN(PA)"] = hn_pa.item()
    if getattr(w, "contrastive_hn_loss_al", 0) > 0:
        hn_al = _hard_neg_loss(model_outputs["action_embedding"], model_outputs["projected_language"])
        total_loss += w.contrastive_hn_loss_al * hn_al
        loss_dict["HN(AL)"] = hn_al.item()

    # ---- Supervised SpaceWarp loss ----
    if hasattr(w, "spacewarp_loss") and w.spacewarp_loss > 0:
        coord_embed_pred = model_outputs.get("coord_embed")
        if coord_embed_pred is not None:
            sw_loss = _safe_mse(coord_embed_pred, model_outputs["warp_embedding"].detach())
            total_loss += w.spacewarp_loss * sw_loss
            loss_dict["SpaceWarp"] = sw_loss.item()

    # Goal warp logdet regularization
    if hasattr(w, "goalwarp_logdet") and w.goalwarp_logdet > 0 and model_outputs.get("goal_logdet") is not None:
        gld = model_outputs["goal_logdet"]
        # Penalise both expansion (positive) and compression (negative) equally
        goal_loss = gld.abs().mean()
        total_loss += w.goalwarp_logdet * goal_loss
        loss_dict["GoalLogDet"] = goal_loss.item() if 'goal_loss' in locals() else 0

    # Logging of loss breakdown is now handled in DreamerTrainer; suppress
    # redundant console output here to keep stdout concise.

    return total_loss, loss_dict

class Trainer:
    """
    Handles the ultimate training loop for the Quaternity Generative Agent,
    incorporating the full 19-objective loss function.
    """
    def __init__(self, cfg: DictConfig, model: GiaAgent, checkpoint_path: str | None = None):
        self.cfg = cfg
        self.model = model
        self.device = next(self.model.parameters()).device
        
        lr_groups = self.cfg.training.get("lr_groups", {})
        def _only_params(it):
            return [p for p in it if isinstance(p, nn.Parameter)]

        lora_params = _only_params([p for n, p in self.model.named_parameters() if "lora" in n and p.requires_grad])
        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr_groups.get("lora", 1e-5)})
        # remaining params
        other_params = _only_params([p for p in self.model.parameters() if p not in lora_params])
        param_groups.append({"params": other_params})
        # Prevent Hydra from converting our param groups into DictConfig objects (breaks PyTorch optimizers)
        self.optimizer = hydra.utils.instantiate(
            self.cfg.training.optimizer,
            params=param_groups,
            _convert_="object",
        )

        self.scheduler = hydra.utils.instantiate(
            self.cfg.training.scheduler,
            optimizer=self.optimizer,
            _convert_="object",
        )
        
        self.dataloader = create_dataloader(
            batch_size=self.cfg.training.batch_size,
            dataset_roots=self.cfg.training.get("dataset_roots"),
            dataset_weights=self.cfg.training.get("dataset_weights"),
            expected_warp_dim=self.cfg.training.get("expected_warp_dim")
        )
        
        use_bf16 = bool(self.cfg.get("use_bfloat16", False)) and torch.cuda.is_available()
        self._mp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        self._use_mixed_precision = bool(self.cfg.get("use_mixed_precision", False) or use_bf16) and torch.cuda.is_available()

        if self._use_mixed_precision:
            print("[MP] Enabling mixed precision for Bridge and ActionTower …")
            self.model.bridge.half()
            self.model.action_tower.half()
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception:
                self.scaler = None
        else:
            self.scaler = None
        
        # ---- Checkpoint resume ----
        self.start_epoch = 0
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            self.start_epoch = ckpt.get("epoch", 0)
            print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.start_epoch})")

        # Initialize dataset and dataloaders
        self.val_dataloader = create_dataloader(
            batch_size=self.cfg.training.batch_size,
            split="val",
            dataset_roots=self.cfg.training.get("dataset_roots"),
            dataset_weights=self.cfg.training.get("dataset_weights"),
            expected_warp_dim=self.cfg.training.get("expected_warp_dim")
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.cfg.get("log_dir", "runs/default"))

        # ---- Checkpoint directory ----
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.ckpt_dir = os.path.join(self.cfg.get("checkpoint_root", "checkpoints"), timestamp)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Early stopping
        self._best_val = float("inf")
        self._no_improve_epochs = 0
        self._patience = int(self.cfg.training.get("early_stop_patience", 0))

    def _calculate_contrastive_loss(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Helper function to compute a symmetric contrastive loss."""
        features_a = F.normalize(features_a, p=2, dim=1)
        features_b = F.normalize(features_b, p=2, dim=1)

        logit_scale = self.model.bridge.logit_scale.exp()
        logits_per_a = logit_scale * features_a @ features_b.t()
        logits_per_b = logits_per_a.t()

        batch_size = features_a.shape[0]
        labels = torch.arange(batch_size, device=self.device)

        loss = (F.cross_entropy(logits_per_a, labels) + 
                F.cross_entropy(logits_per_b, labels)) / 2
        return loss

    def _calculate_grand_loss(self, model_outputs: dict, batch: dict) -> torch.Tensor:
        """
        Calculates the grand total loss from all 19 objectives.
        """
        w = self.cfg.training.loss_weights
        total_loss = torch.tensor(0.0, device=self.device)

        # --- Ground Truth Data ---
        target_action = batch["action"].to(self.device)
        target_image = batch["screenshot"].to(self.device)
        target_warp = batch["warp"].to(self.device)
        
        # --- Loss 1: Imitation Learning ---
        imitation_loss = F.mse_loss(model_outputs["predicted_action"], target_action)
        total_loss += w.action_imitation_loss * imitation_loss

        # --- Loss 2: 6-Way Contrastive Loss ---
        loss_pl = self._calculate_contrastive_loss(model_outputs["projected_vision"], model_outputs["projected_language"])
        loss_pa = self._calculate_contrastive_loss(model_outputs["projected_vision"], model_outputs["action_embedding"])
        loss_pw = self._calculate_contrastive_loss(model_outputs["projected_vision"], model_outputs["warp_embedding"])
        loss_la = self._calculate_contrastive_loss(model_outputs["projected_language"], model_outputs["action_embedding"])
        loss_lw = self._calculate_contrastive_loss(model_outputs["projected_language"], model_outputs["warp_embedding"])
        loss_aw = self._calculate_contrastive_loss(model_outputs["action_embedding"], model_outputs["warp_embedding"])
        total_loss += (w.contrastive_loss_pl * loss_pl + w.contrastive_loss_pa * loss_pa + 
                       w.contrastive_loss_pw * loss_pw + w.contrastive_loss_la * loss_la + 
                       w.contrastive_loss_lw * loss_lw + w.contrastive_loss_aw * loss_aw)
        
        # --- Loss 3: 12-Way Generative Loss (11 implemented) ---
        gen_p_from_l = _safe_mse(model_outputs["pred_image_from_l"], target_image)
        gen_p_from_a = _safe_mse(model_outputs["pred_image_from_a"], target_image)
        gen_p_from_w = _safe_mse(model_outputs["pred_image_from_w"], target_image)
        gen_a_from_p = _safe_mse(model_outputs["pred_action_from_p"], target_action)
        gen_a_from_l = _safe_mse(model_outputs["pred_action_from_l"], target_action)
        gen_a_from_w = _safe_mse(model_outputs["pred_action_from_w"], target_action)
        gen_w_from_p = _safe_mse(model_outputs["pred_warp_from_p"], target_warp)
        gen_w_from_l = _safe_mse(model_outputs["pred_warp_from_l"], target_warp)
        gen_w_from_a = _safe_mse(model_outputs["pred_warp_from_a"], target_warp)
        total_loss += (w.generative_loss_p_from_l * gen_p_from_l + w.generative_loss_p_from_a * gen_p_from_a +
                       w.generative_loss_p_from_w * gen_p_from_w + w.generative_loss_a_from_p * gen_a_from_p +
                       w.generative_loss_a_from_l * gen_a_from_l + w.generative_loss_a_from_w * gen_a_from_w +
                       w.generative_loss_w_from_p * gen_w_from_p + w.generative_loss_w_from_l * gen_w_from_l +
                       w.generative_loss_w_from_a * gen_w_from_a)

        # --- Optional: Supervised SpaceWarp loss ---
        if hasattr(w, "spacewarp_loss") and w.spacewarp_loss > 0:
            coord_embed_pred = model_outputs.get("coord_embed")
            if coord_embed_pred is not None:
                sw_loss = _safe_mse(coord_embed_pred, model_outputs["warp_embedding"].detach())
                total_loss += w.spacewarp_loss * sw_loss
            else:
                sw_loss = torch.tensor(0.0, device=self.device)

        # Goal warp logdet regularization
        if hasattr(w, "goalwarp_logdet") and w.goalwarp_logdet > 0 and model_outputs.get("goal_logdet") is not None:
            gld = model_outputs["goal_logdet"]
            # Penalise both expansion (positive) and compression (negative) equally
            goal_loss = gld.abs().mean()
            total_loss += w.goalwarp_logdet * goal_loss

        # ---- Throttled console logging ----
        if not hasattr(self, "_log_counter"):
            self._log_counter = 0  # type: ignore
        self._log_counter += 1  # type: ignore
        log_interval = int(self.cfg.training.get("log_interval", 100)) if hasattr(self, "cfg") else 100  # type: ignore
        if self._log_counter % log_interval == 0:
            print(f"Total: {total_loss.item():.4f} | Imit: {imitation_loss.item():.4f} | C(PL): {loss_pl.item():.4f} | G(P<-L): {gen_p_from_l.item():.4f}")

        return total_loss, {
            "Imitation": imitation_loss.item(),
            "SpaceWarp": sw_loss.item() if 'sw_loss' in locals() else 0,
            "GoalLogDet": goal_loss.item() if 'goal_loss' in locals() else 0,
        }
        
    def train(self):
        self.model.train()
        print("Starting Quaternity Generative Architecture training...")
        train_start_time = time.time()
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            print(f"Epoch {epoch+1}/{self.cfg.training.num_epochs}")
            epoch_start_time = time.time()
            
            # Determine generative scaling factor
            warm_epochs = self.cfg.training.get("generative_warmup_epochs", 0)
            if warm_epochs > 0 and epoch < warm_epochs:
                gen_scale = (epoch + 1) / warm_epochs
            else:
                gen_scale = 1.0

            # Create scaled loss_weights copy
            loss_weights_scaled = self.cfg.training.loss_weights.copy()
            for k in loss_weights_scaled:
                if k.startswith("generative_loss_"):
                    loss_weights_scaled[k] = loss_weights_scaled[k] * gen_scale

            accum_steps = int(self.cfg.training.get("grad_accumulation_steps", 1))
            for batch_idx, batch in enumerate(self.dataloader):
                # Move batch to the same device as the model
                screenshot = batch["screenshot"].to(torch.float16 if self._use_mixed_precision else torch.float32)
                action = batch["action"].to(self.device, dtype=torch.float32)
                instruction_text = batch["instruction_text"]
                warp = batch["warp"].to(self.device)

                # Use autocast for automatic mixed-precision handling (only if dtype is half/bfloat16)
                if self._use_mixed_precision:
                    _has_device_arg = 'device_type' in inspect.signature(autocast).parameters
                    _autocast_ctx = autocast(device_type='cuda', dtype=self._mp_dtype) if _has_device_arg else autocast(dtype=self._mp_dtype)
                else:
                    _autocast_ctx = nullcontext()
                with _autocast_ctx:
                    model_outputs = self.model(
                        instruction_text=instruction_text,
                        screenshot=screenshot,
                        target_action=action,
                        target_warp=warp,
                        target_goal=batch.get("goal")
                    )
                    loss_batch = { "action": action, "screenshot": screenshot, "warp": warp }
                    loss, extra = self._calculate_grand_loss(model_outputs, loss_batch)
                
                if not torch.isnan(loss) and loss.requires_grad:
                    # Zero grads at start of accumulation
                    if (batch_idx % accum_steps) == 0:
                        self.optimizer.zero_grad()

                    if self._use_mixed_precision and self.scaler is not None and self.scaler.is_enabled():
                        # --- Mixed-precision path with GradScaler ---
                        self.scaler.scale(loss).backward()

                        # Unscale the gradients before clipping so that the clip threshold is in true scale
                        try:
                            self.scaler.unscale_(self.optimizer)
                        except ValueError:
                            pass  # Already unscaled / not needed

                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.max_grad_norm)

                        # Optimiser step + scaler update with robust fallback
                        _stepped_scaled = False
                        try:
                            self.scaler.step(self.optimizer)
                            _stepped_scaled = True
                        except ValueError:
                            # No scaled grads to unscale; fallback to regular step
                            self.optimizer.step()

                        # Update scaler only if scaled step was executed successfully
                        if _stepped_scaled:
                            self.scaler.update()
                    else:
                        # Standard FP32 path (no mixed precision)
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.max_grad_norm)

                        self.optimizer.step()

                    # Optimiser step when accumulation finished
                    if ((batch_idx + 1) % accum_steps) == 0:
                        # Scheduler step once per optimiser update
                        self.scheduler.step()

                    # ---- TensorBoard logging ----
                    global_step = epoch * len(self.dataloader) + batch_idx
                    if ((batch_idx + 1) % accum_steps) == 0:  # log after real update
                        self.writer.add_scalar("train/total_loss", loss.item()*accum_steps, global_step)
                        for k, v in extra.items():
                            self.writer.add_scalar(f"train/{k}", v, global_step)
            
            print(f"Epoch {epoch+1} finished.")

            # ---- Epoch timing ----
            epoch_dur = time.time() - epoch_start_time
            print(f"[Timing] Epoch {epoch+1} took {epoch_dur:.1f} sec ({epoch_dur/60:.2f} min)")

            # -------- Validation --------
            val_loss = self._validate(epoch)
            print(f"Validation loss: {val_loss:.4f}")

            # TensorBoard logging
            self.writer.add_scalar("val/total_loss", val_loss, epoch)

            # Save checkpoint
            ckpt_path_pt = os.path.join(self.ckpt_dir, f"epoch_{epoch+1}.pt")
            state: dict[str, Any] = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch+1,
            }
            if _HAS_SAFE:
                ckpt_path = ckpt_path_pt.replace('.pt', '.safetensors')
                _safe_save(state["model_state"], ckpt_path)
                torch.save({k: v for k, v in state.items() if k != "model_state"}, ckpt_path_pt)  # save rest
            else:
                ckpt_path = ckpt_path_pt
                torch.save(state, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

            # Early stopping
            if self._patience > 0:
                if val_loss < self._best_val:
                    self._best_val = val_loss
                    self._no_improve_epochs = 0
                else:
                    self._no_improve_epochs += 1
                    if self._no_improve_epochs >= self._patience:
                        print(f"[Trainer] Early stopping: no improvement for {self._patience} epochs.")
                        break

        print("Training finished.")
        total_dur = time.time() - train_start_time
        print(f"[Timing] Total training time: {total_dur/3600:.2f} h ({total_dur/60:.1f} min)")
        self.writer.close()

    def _validate(self, epoch: int) -> float:
        """Runs the validation loop and logs metrics."""
        self.model.eval()
        running_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                screenshot = batch["screenshot"].to(torch.float32)
                action = batch["action"].to(self.device, dtype=torch.float32)
                instruction_text = batch["instruction_text"]
                warp = batch["warp"].to(self.device)

                outputs = self.model(
                    instruction_text=instruction_text,
                    screenshot=screenshot,
                    target_action=action,
                    target_warp=warp,
                    target_goal=batch.get("goal")
                )

                loss, extra = self._calculate_grand_loss(outputs, { "action": action, "screenshot": screenshot, "warp": warp })

                running_loss += loss.item()
                count += 1

                # Metrics
                mse = action_mse(outputs["predicted_action"].cpu(), action.cpu())
                acc = click_accuracy(outputs["predicted_action"].cpu(), action.cpu())

                self.writer.add_scalar("val/action_mse_batch", mse, epoch * len(self.val_dataloader) + count)
                self.writer.add_scalar("val/click_acc_batch", acc, epoch * len(self.val_dataloader) + count)
                for k,v in extra.items():
                    self.writer.add_scalar(f"val/{k}_batch", v, epoch*len(self.val_dataloader)+count)

        avg_loss = running_loss / max(count, 1)
        self.model.train()

        # Log aggregated metrics
        self.writer.add_scalar("val/action_mse", mse, epoch)
        self.writer.add_scalar("val/click_acc", acc, epoch)

        return avg_loss 