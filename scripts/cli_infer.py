import argparse
import base64
from pathlib import Path
import json

import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

from models.gia_agent import GiaAgent


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)


def main():
    parser = argparse.ArgumentParser("Run single inference with GIA")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt or .safetensors)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--image", required=True, help="Path to screenshot image (PNG/JPG)")
    parser.add_argument("--instruction", required=True, help="Instruction text")
    parser.add_argument("--out", default="prediction.json", help="Where to save output JSON")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = GiaAgent(cfg.model)

    ckpt_path = Path(args.ckpt)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(ckpt_path)
        model.load_state_dict(state, strict=False)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img_tensor = load_image(Path(args.image)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            instruction_text=[args.instruction],
            screenshot=img_tensor,
            target_action=torch.zeros(1, 4, device=device),
            target_warp=torch.zeros(1, cfg.training.expected_warp_dim, device=device),
            target_goal=torch.zeros(1, cfg.training.expected_warp_dim, device=device),
        )
        action = outputs["predicted_action"].squeeze(0).cpu().tolist()

    result = {
        "x": action[0],
        "y": action[1],
        "click": bool(action[2] > 0.5),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("Saved prediction to", args.out)


if __name__ == "__main__":
    main() 