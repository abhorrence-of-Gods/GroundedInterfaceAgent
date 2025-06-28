import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple

import keywin.input as kw
from PIL import Image
import pyscreenshot as ImageGrab

# Simple recorder that logs left-clicks coordinates and captures screen.

def capture_screen() -> Image.Image:
    return ImageGrab.grab()


def record_session(out_dir: Path, num_samples: int, delay: float = 1.0, warp_dim: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    meta: List[dict] = []

    print("[Recorder] Move your mouse and click; ESC to stop early.")
    samples = 0
    try:
        while samples < num_samples:
            if kw.peek_mouse_down():
                x, y = kw.get_mouse_pos()
                screen = capture_screen()
                img_path = out_dir / "images" / f"{samples:06d}.png"
                screen.save(img_path)

                w, h = screen.size
                action = [x / w, y / h, 1.0, 0.0]
                warp = [0.5, 0.5] + [0.0] * (warp_dim - 2)
                instruction = "click at ({}, {})".format(x, y)

                meta.append({
                    "image": str(img_path.relative_to(out_dir)),
                    "instruction_text": instruction,
                    "action": action,
                    "warp": warp,
                    "goal": [x / w, y / h] + [0.0]*10,
                })
                samples += 1
                print(f"Captured {samples}/{num_samples}")
                time.sleep(delay)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(meta)} samples to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Real GUI recorder")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0, help="Delay after each capture (s)")
    parser.add_argument("--warp_dim", type=int, default=16, help="Desired length of warp vector (default 16)")
    args = parser.parse_args()
    record_session(Path(args.out_dir), args.num_samples, args.delay, warp_dim=args.warp_dim) 