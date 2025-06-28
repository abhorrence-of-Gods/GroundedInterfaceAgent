import torch
from omegaconf import DictConfig

from models.gia_agent import GiaAgent
from data.dataloader import create_dataloader
from engine.metrics import action_mse, click_accuracy
from engine.trainer import calculate_comprehensive_loss


class Evaluator:
    """Simple offline evaluator that measures loss, MSE, and click accuracy on a
    held-out validation set (synthetic GUI dataset for now). Laterこのクラスに
    実 GUI タスク評価を追加できるように設計している。"""

    def __init__(self, cfg: DictConfig, model: GiaAgent):
        self.cfg = cfg
        self.model = model.eval()
        self.device = next(self.model.parameters()).device

        # Validation dataloader (non-shuffled, smaller dataset)
        self.val_loader = create_dataloader(
            batch_size=cfg.training.batch_size,
            split="val",
            dataset_roots=cfg.training.get("dataset_roots"),
            dataset_weights=cfg.training.get("dataset_weights"),
            expected_warp_dim=cfg.training.get("expected_warp_dim"),
        )

    @torch.no_grad()
    def evaluate(self) -> dict:
        print("[Evaluator] Running validation loop ...")

        total_loss_sum = 0.0
        mse_sum = 0.0
        acc_sum = 0.0
        count = 0

        for batch in self.val_loader:
            screenshot = batch["screenshot"].to(self.device, dtype=torch.float32)
            action_tgt = batch["action"].to(self.device, dtype=torch.float32)
            warp_tgt = batch["warp"].to(self.device, dtype=torch.float32)
            instruction = batch["instruction_text"]

            outputs = self.model(
                instruction_text=instruction,
                screenshot=screenshot,
                target_action=action_tgt,
                target_warp=warp_tgt,
            )

            loss, _ = calculate_comprehensive_loss(
                model_outputs=outputs,
                batch={
                    "action": action_tgt,
                    "screenshot": screenshot,
                    "warp": warp_tgt,
                },
                loss_weights=self.cfg.training.loss_weights,
                logit_scale=self.model.bridge.logit_scale,
                device=self.device,
            )

            total_loss_sum += loss.item()
            mse_sum += action_mse(outputs["predicted_action"].cpu(), action_tgt.cpu())
            acc_sum += click_accuracy(outputs["predicted_action"].cpu(), action_tgt.cpu())
            count += 1

        metrics = {
            "val/total_loss": total_loss_sum / max(count, 1),
            "val/action_mse": mse_sum / max(count, 1),
            "val/click_acc": acc_sum / max(count, 1),
        }

        print("[Evaluator] Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return metrics 