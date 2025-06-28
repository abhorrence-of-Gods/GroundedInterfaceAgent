import torch
import argparse, random, os
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from omegaconf import OmegaConf
import hydra

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gia_agent import GiaAgent
from data.dataloader import create_dataloader


def main():
    parser = argparse.ArgumentParser("Visualize latent space before/after GoalWarp using TSNE")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="warp_tsne.png")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = GiaAgent(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    dl = create_dataloader(batch_size=32, expected_warp_dim=cfg.training.expected_warp_dim)
    pre_embeds, post_embeds = [], []
    with torch.no_grad():
        for batch in dl:
            instr = batch["instruction_text"]
            screenshot = batch["screenshot"]
            goal = batch.get("goal")
            if goal is None:
                goal = torch.zeros(screenshot.size(0), cfg.training.expected_warp_dim)
            bridge = model.bridge(model.language_tower(instr), model.perception_tower(screenshot))
            fused = bridge["fused_embedding"]
            pre_embeds.append(fused)
            warped = model.goal_warp(fused, goal)
            post_embeds.append(warped)
            if len(pre_embeds)*32 >= args.samples:
                break
    pre = torch.cat(pre_embeds, dim=0)[:args.samples].cpu().numpy()
    post = torch.cat(post_embeds, dim=0)[:args.samples].cpu().numpy()

    combined = torch.from_numpy(pre).float()
    tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=0)
    coords_pre = tsne.fit_transform(pre)
    coords_post = tsne.fit_transform(post)

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].scatter(coords_pre[:,0], coords_pre[:,1], s=8, alpha=0.7)
    ax[0].set_title('Before GoalWarp')
    ax[1].scatter(coords_post[:,0], coords_post[:,1], s=8, alpha=0.7, color='orange')
    ax[1].set_title('After GoalWarp')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved TSNE plot to {args.out}")

if __name__ == "__main__":
    main() 