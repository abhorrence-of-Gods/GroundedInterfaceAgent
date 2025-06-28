import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image, ImageDraw
from torchvision import transforms
import random
import numpy as np
import json
from pathlib import Path
from collections.abc import Sequence

class GuiActionDataset(Dataset):
    """
    A dummy dataset that generates programmatically correlated data for all four
    modalities: vision, language, action, and spacetime dynamics.
    """
    def __init__(self, num_samples=1000, image_size=(224, 224), action_dim=4, warp_dim=16):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        self.warp_dim = warp_dim
        # Ensure every screenshot has the same resolution so that default
        # collate_fn can stack the resulting tensors without raising size
        # mismatch errors. 224×224 keeps computation light while preserving
        # enough detail for most GUI elements. If you need a different
        # resolution, expose it via `create_dataloader` arguments.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ])
        
        self.instruction_map = {
            "click the blue button": {
                "draw": lambda draw: draw.rectangle([50, 50, 100, 100], fill="blue"),
                "action": [75/image_size[0], 75/image_size[1], 1, 0],
                "warp": [0.5, 0.5] + [0.0] * 14  # pad to 16-dim
            },
            "click the red square quickly": {
                "draw": lambda draw: draw.rectangle([120, 120, 170, 170], fill="red"),
                "action": [145/image_size[0], 145/image_size[1], 1, 0],
                "warp": [1.0, 0.2] + [0.0] * 14  # pad to 16-dim
            },
            "click the green circle precisely": {
                "draw": lambda draw: draw.ellipse([80, 150, 130, 200], fill="green"),
                "action": [105/image_size[0], 175/image_size[1], 1, 0],
                "warp": [0.1, 1.0] + [0.0] * 14  # pad to 16-dim
            },
        }
        self.instructions = list(self.instruction_map.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generates a single, fully correlated dummy data point."""
        instruction = self.instructions[idx % len(self.instructions)]
        metadata = self.instruction_map[instruction]
        
        image = Image.new('RGB', self.image_size, color = 'white')
        draw = ImageDraw.Draw(image)
        metadata["draw"](draw)
        
        action = torch.tensor(metadata["action"], dtype=torch.float32)
        warp = torch.tensor(metadata["warp"], dtype=torch.float32)

        shape_onehot = {
            "click the blue button": [1,0,0,0],
            "click the red square quickly": [0,1,0,0],
            "click the green circle precisely": [1,0,0,0],
        }.get(instruction, [0,0,0,0])
        x_norm = metadata["action"][0]
        y_norm = metadata["action"][1]
        goal = torch.tensor([x_norm, y_norm] + shape_onehot + [0.0]*4, dtype=torch.float32)

        return {
            'instruction_text': instruction,
            'screenshot': self.transform(image),
            'action': action,
            'warp': warp,
            'goal': goal,
        }

def _build_dataset(root: str | Path):
    root_path = Path(root)
    if (root_path / "metadata.json").exists():
        return SyntheticGUIDataset(root_path)
    raise FileNotFoundError(f"Unknown dataset format at {root}")

def create_dataloader(batch_size: int, split: str = "train", num_workers: int = 0,
                      dataset_root: str | None = None, dataset_roots: list[str] | None = None,
                      dataset_weights: list[float] | None = None, expected_warp_dim: int | None = None):
    """Creates DataLoader; supports single or multiple dataset roots."""

    # Normalize root(s) input -------------------------------------------------
    # Support three possibilities:
    # 1) legacy single-root argument `dataset_root`
    # 2) list/tuple of roots via `dataset_roots`
    # 3) a single string accidentally passed via Hydra override, e.g.
    #    `training.dataset_roots=/path/to/root` (not wrapped in list)

    if dataset_roots is None and dataset_root is not None:
        dataset_roots = [dataset_root]

    # If the caller passed a plain string, wrap it into a list so the logic
    # below doesn't iterate character-wise and raise `Unknown dataset format at /`.
    if isinstance(dataset_roots, str):
        dataset_roots = [dataset_roots]

    # Ensure weights, if provided, are a plain Python list of floats.
    if dataset_weights is not None:
        if isinstance(dataset_weights, Sequence) and not isinstance(dataset_weights, (str, bytes)):
            # Convert any Hydra ListConfig or other Sequence to list[float]
            dataset_weights = [float(w) for w in list(dataset_weights)]
        else:
            # Scalar weight → single-element list
            dataset_weights = [float(dataset_weights)]

    if dataset_roots:
        datasets_raw = [_build_dataset(r) for r in dataset_roots if r]
        # apply warp padding if needed
        if expected_warp_dim is not None:
            datasets = [WarpPadDataset(ds, expected_warp_dim) for ds in datasets_raw]
        else:
            datasets = datasets_raw
        shuffle = split == "train"
    else:
        # Legacy dummy dataset (single GuiActionDataset)
    if split == "train":
            datasets = [GuiActionDataset(num_samples=1000)]
        shuffle = True
    else:
            datasets = [GuiActionDataset(num_samples=200)]
        shuffle = False

    # -------------------------------------------------------------
    # Combine datasets (single or multiple) with optional weighting
    # -------------------------------------------------------------
    if dataset_weights and len(dataset_weights) == len(datasets):
        dataset = BalancedConcatDataset(datasets, dataset_weights)
    elif len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) 

# ---------------------- New: Synthetic GUI Dataset ----------------------

class SyntheticGUIDataset(Dataset):
    """Loads synthetic GUI dataset generated by scripts/generate_synthetic_gui.py"""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        meta_path = self.root_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {root_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # Resize to the same resolution applied in GuiActionDataset so that
        # all tensors share an identical shape during batching.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        img_path = self.root_dir / item["image"]
        image = Image.open(img_path).convert("RGB")

        screenshot = self.transform(image)
        action = torch.tensor(item["action"], dtype=torch.float32)
        warp = torch.tensor(item["warp"], dtype=torch.float32)

        goal_vec = torch.tensor(item.get("goal", [0.0]*8), dtype=torch.float32)
        if goal_vec.shape[-1] < 8:
            goal_vec = torch.cat([goal_vec, torch.zeros(8 - goal_vec.shape[-1])], dim=-1)
        return {
            "instruction_text": item["instruction_text"],
            "screenshot": screenshot,
            "action": action,
            "warp": warp,
            "goal": goal_vec,
        }

# Balanced concat dataset with sampling weights
class BalancedConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset], weights: list[float]):
        assert len(datasets) == len(weights)
        self.datasets = datasets
        self.weights = [w / sum(weights) for w in weights]

    def __len__(self):
        # define as sum of lengths
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        # ignore idx, sample according to weights
        ds = random.choices(self.datasets, self.weights)[0]
        inner_idx = random.randint(0, len(ds) - 1)
        return ds[inner_idx]

class WarpPadDataset(Dataset):
    """Wraps a dataset and pads/truncates 'warp' tensor to expected_dim."""
    def __init__(self, base: Dataset, expected_dim: int):
        self.base = base
        self.expected_dim = expected_dim

    def __len__(self):
        return len(self.base)

    def _pad(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.shape[-1] == self.expected_dim:
            return vec
        if vec.shape[-1] > self.expected_dim:
            return vec[..., : self.expected_dim]
        pad_size = self.expected_dim - vec.shape[-1]
        pad = torch.zeros(pad_size, dtype=vec.dtype)
        return torch.cat([vec, pad], dim=-1)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["warp"] = self._pad(item["warp"])
        if "goal" in item:
            item["goal"] = self._pad(item["goal"])
        return item 