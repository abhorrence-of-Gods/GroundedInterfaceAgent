import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
import os

from models.gia_agent import GiaAgent
from engine.dreamer_trainer import DreamerTrainer

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Grounded Interface Agent (GIA),
    now orchestrated by the DreamerTrainer.
    """
    print("Initializing GIA Agent (Dreamer Edition)...")
    print(OmegaConf.to_yaml(cfg.model))

    # Initialize the main agent model from the config
    model = GiaAgent(cfg.model)

    if cfg.mode == "train":
        print("Mode: Training with Dreamer")
        trainer = DreamerTrainer(cfg=cfg, model=model)
        trainer.train()

    elif cfg.mode == "evaluate":
        print("Mode: Evaluation (offline)")
        from engine.evaluator import Evaluator
        evaluator = Evaluator(cfg=cfg, model=model)
        evaluator.evaluate()

    elif cfg.mode == "inference":
        print("Mode: Interactive inference")

        # --- Load checkpoint if provided ---
        ckpt_path = cfg.get("checkpoint_path")
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print(f"[main] Loading checkpoint from {ckpt_path} …")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
            if missing:
                print(f"[main] ⚠️  Missing keys while loading checkpoint: {missing}")
            if unexpected:
                print(f"[main] ⚠️  Unexpected keys while loading checkpoint: {unexpected}")
        else:
            print("[main] No valid checkpoint given – using randomly initialised weights.")

        from engine.inference import InferenceEngine

        # Single instruction inference; can be overridden via CLI: +instruction="Click the OK button"
        instruction: str = cfg.get("instruction", "Click the start button.")
        engine = InferenceEngine(cfg=cfg, model=model)
        engine.run(instruction, execute_env=True)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

if __name__ == "__main__":
    main() 