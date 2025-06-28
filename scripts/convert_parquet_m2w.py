import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
from io import BytesIO
import base64
import pyarrow.parquet as pq
from tqdm import tqdm

TAG_VOCAB = [
    "button",
    "a",
    "input",
    "img",
    "select",
    "textarea",
    "checkbox",
    "radio",
    "div",
    "span",
]


def _decode_image(b64: str) -> Image.Image:
    """Decode base64 PNG/JPEG image to PIL Image."""
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def convert_parquet(src_dir: Path, out_dir: Path, warp_dim: int = 16):
    """Convert Multimodal-Mind2Web parquet files to GIA metadata format.

    Args:
        src_dir: Directory containing *.parquet files (train-*.parquet).
        out_dir: Output directory; images/ subdir and metadata.json will be created.
        warp_dim: Length of warp vector (default 16).
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    meta: List[dict] = []
    parquet_files = sorted(src_dir.glob("*.parquet"))
    if not parquet_files:
        print("[Warn] No parquet files under", src_dir)
        return

    for pf in parquet_files:
        print("[Convert]", pf.name)
        pf_obj = pq.ParquetFile(pf)
        # process row groups in streaming fashion to keep memory low
        for rg in range(pf_obj.num_row_groups):
            table = pf_obj.read_row_group(rg)
            df = table.to_pandas()
            for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
                # Get instruction & screenshot
                instr = row.get("instruction") or row.get("confirmed_task", "")
                img: Image.Image | None = None
                # Multimodal-Mind2Web has several variants for the screenshot:
                #   1) raw bytes in `screenshot`
                #   2) relative file path in `screenshot` or `screenshot_path`
                #   3) relative file path in `path` (newer schema)
                scr_val = row.get("screenshot") if "screenshot" in row else None
                if scr_val is None:
                    scr_val = row.get("screenshot_path") or row.get("path")

                # If the value is dict and contains raw bytes -> extract
                if isinstance(scr_val, dict) and "bytes" in scr_val:
                    scr_val = scr_val["bytes"]

                if isinstance(scr_val, (bytes, bytearray)):
                    try:
                        img = Image.open(BytesIO(scr_val)).convert("RGB")
                    except Exception:
                        img = None
                elif isinstance(scr_val, str) and scr_val.strip().lower().endswith((".png", ".jpg", ".jpeg")):
                    path_candidate = (src_dir / scr_val.strip()).resolve()
                    if path_candidate.exists():
                        try:
                            img = Image.open(path_candidate).convert("RGB")
                        except Exception:
                            img = None
                # Fallback again for screenshot_path if not tried above
                if img is None and isinstance(row.get("screenshot_path"), str):
                    candidate = (src_dir / row["screenshot_path"].strip()).resolve()
                    if candidate.exists():
                        try:
                            img = Image.open(candidate).convert("RGB")
                        except Exception:
                            img = None
                # Legacy inline base64 (rare)
                if img is None and isinstance(scr_val, str) and scr_val.startswith("data:image"):
                    try:
                        img = _decode_image(scr_val)
                    except Exception:
                        img = None
                if img is None:
                    continue  # skip sample

                # Mouse coordinates (assumed absolute) and tag hint if available
                x_norm = row.get("mouse_x_norm")
                y_norm = row.get("mouse_y_norm")
                tag: str = ""

                # First fallback: derive from first entry of pos_candidates bounding_box_rect
                def _center_from_rect(rect_str: str):
                    try:
                        x, y, w, h = map(float, rect_str.split(","))
                        return (x + w / 2), (y + h / 2)
                    except Exception:
                        return None, None

                pcs_raw = row.get("pos_candidates", None)
                if (x_norm is None or y_norm is None) and pcs_raw is not None:
                    try:
                        # Ensure list of dicts
                        if isinstance(pcs_raw, str):
                            pcs = ast.literal_eval(pcs_raw)
                        else:
                            pcs = pcs_raw

                        parsed = []
                        for itm in pcs:
                            if isinstance(itm, dict):
                                parsed.append(itm)
                            elif isinstance(itm, str):
                                try:
                                    parsed.append(json.loads(itm))
                                except Exception:
                                    try:
                                        parsed.append(ast.literal_eval(itm))
                                    except Exception:
                                        pass

                        if parsed:
                            cand0 = parsed[0]
                            rect = cand0.get("bounding_box_rect") or cand0.get("bounding_box")
                            # If still None, inspect attributes JSON string
                            if rect is None and isinstance(cand0.get("attributes"), str):
                                try:
                                    attrs = json.loads(cand0["attributes"])
                                    rect = attrs.get("bounding_box_rect") or attrs.get("bounding_box")
                                except Exception:
                                    try:
                                        attrs = ast.literal_eval(cand0["attributes"])
                                        rect = attrs.get("bounding_box_rect") or attrs.get("bounding_box")
                                    except Exception:
                                        rect = None
                            if rect:
                                cx, cy = _center_from_rect(rect)
                                if cx is not None:
                                    x_norm = cx / img.width
                                    y_norm = cy / img.height
                                    tag = str(cand0.get("tag", "")).lower()
                    except Exception:
                        pass

                # Second fallback: bbox encoded in target_action_... columns
                if (x_norm is None or y_norm is None) and row.get("target_action_bbox"):
                    try:
                        import ast
                        bx, by, bw, bh = ast.literal_eval(str(row["target_action_bbox"]))
                        x_norm = (bx + bw / 2) / img.width
                        y_norm = (by + bh / 2) / img.height
                    except Exception:
                        pass

                if x_norm is None or y_norm is None:
                    continue
                action = [x_norm, y_norm, 1.0, 0.0]
                warp = [0.5, 0.5] + [0.0] * (warp_dim - 2)

                # If tag is still empty, attempt to glean from `target_action_reprs` like "[div] ..."
                if not tag and row.get("target_action_reprs"):
                    try:
                        rep = str(row["target_action_reprs"])
                        if rep.startswith("[") and "]" in rep:
                            tag = rep[1 : rep.index("]")].lower()
                    except Exception:
                        pass

                tag_onehot = [1 if tag == t else 0 for t in TAG_VOCAB]
                goal = [x_norm, y_norm] + tag_onehot + [0.0] * (16 - 2 - len(tag_onehot))

                idx = len(meta)
                img_path = out_dir / "images" / f"{idx:06d}.png"
                try:
                    img.save(img_path)
                except Exception:
                    continue  # skip if save fails

                meta.append(
                    {
                        "image": str(img_path.relative_to(out_dir)),
                        "instruction_text": instr if instr else row.get("confirmed_task", ""),
                        "action": action,
                        "warp": warp,
                        "goal": goal,
                    }
                )

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Converted {len(meta)} samples → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Multimodal-Mind2Web Parquet to GIA metadata")
    parser.add_argument("src_dir", help="Directory containing *.parquet files (train-*.parquet)")
    parser.add_argument("out_dir", help="Output directory for metadata + images")
    parser.add_argument("--warp_dim", type=int, default=16)
    args = parser.parse_args()

    convert_parquet(Path(args.src_dir), Path(args.out_dir), warp_dim=args.warp_dim) 