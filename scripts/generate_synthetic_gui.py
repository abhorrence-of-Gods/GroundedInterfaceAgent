import os
import random
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --------------------------- CONFIG ---------------------------
IMAGE_SIZE = (224, 224)
SHAPES = ["circle", "square", "triangle", "drag"]
COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]
ACTIONS = {
    "circle": "click the {color} circle",
    "square": "click the {color} square",
    "triangle": "click the {color} triangle",
    "drag": "drag the {color} bar to the right",
}

OUTPUT_DIR = Path("synthetic_dataset")
# -------------------------------------------------------------


def random_shape() -> Tuple[str, str]:
    shape = random.choice(SHAPES)
    color = random.choice(COLORS)
    return shape, color


def draw_shape(draw: ImageDraw.Draw, shape: str, color: str):
    if shape == "circle":
        x0, y0 = random.randint(30, 130), random.randint(30, 130)
        r = random.randint(20, 40)
        draw.ellipse([x0, y0, x0 + 2 * r, y0 + 2 * r], fill=color)
        center = (x0 + r, y0 + r)
    elif shape == "square":
        x0, y0 = random.randint(30, 130), random.randint(30, 130)
        side = random.randint(40, 60)
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=color)
        center = (x0 + side / 2, y0 + side / 2)
    else:  # triangle
        x0, y0 = random.randint(50, 150), random.randint(50, 150)
        size = random.randint(40, 60)
        points = [(x0, y0), (x0 + size, y0), (x0 + size / 2, y0 - size)]
        draw.polygon(points, fill=color)
        center = (sum(p[0] for p in points) / 3, sum(p[1] for p in points) / 3)
    return center


def generate_dataset(num_samples: int = 1000):
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    meta: List[dict] = []
    for idx in tqdm(range(num_samples), desc="Generating synthetic GUI dataset"):
        shape, color = random_shape()
        img = Image.new("RGB", IMAGE_SIZE, color="white")
        draw = ImageDraw.Draw(img)
        if shape == "drag":
            # draw horizontal bar and treat start point as action
            y = random.randint(90, 130)
            draw.rectangle([20, y, 60, y + 20], fill=color)
            center = (40, y + 10)  # start point
        else:
            center = draw_shape(draw, shape, color)

        # Normalized coordinates and click flag
        x_norm = center[0] / IMAGE_SIZE[0]
        y_norm = center[1] / IMAGE_SIZE[1]
        click_flag = 1.0 if shape != "drag" else 0.0
        action_vec = [x_norm, y_norm, click_flag, 0.0]
        warp_vec = [random.random() for _ in range(16)]

        instruction = ACTIONS[shape].format(color=color)

        # Goal vector (8-dim): [x, y, 4 shape one-hot, 2 reserved]
        shape_onehot = {
            "circle": [1, 0, 0, 0],
            "square": [0, 1, 0, 0],
            "triangle": [0, 0, 1, 0],
            "drag": [0, 0, 0, 1],
        }[shape]
        goal_vec = [x_norm, y_norm] + shape_onehot + [0.0]*10

        img_path = OUTPUT_DIR / "images" / f"{idx:06d}.png"
        img.save(img_path)

        meta.append(
            {
                "image": str(img_path.relative_to(OUTPUT_DIR)),
                "instruction_text": instruction,
                "action": action_vec,
                "warp": warp_vec,
                "goal": goal_vec,
            }
        )

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {num_samples} samples to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Synthetic GUI dataset generator")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    generate_dataset(args.num_samples) 