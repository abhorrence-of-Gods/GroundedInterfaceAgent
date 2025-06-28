import json
import argparse
from pathlib import Path
from shutil import copy2
from PIL import Image


def convert(src_dir: Path, out_dir: Path, warp_dim: int = 8):
    """Converts the official ScreenAgent dataset into metadata format.
    It processes each session's JSON files, extracting instructions, actions,
    and associated screenshots to build a comprehensive training set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'images').mkdir(exist_ok=True)
    meta = []

    # Official dataset may have extra nesting (e.g., test/test/<session_id>)
    # Therefore we recurse one level deeper and pick any directory that
    # contains an `images` subfolder.
    for session_dir in src_dir.rglob('*'):
        if not session_dir.is_dir():
            continue
        image_dir = session_dir / 'images'
        if not image_dir.exists():
            continue
        for json_file in session_dir.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            
            img_path_src = image_dir / data.get("saved_image_name", "")
            if not img_path_src.exists():
                continue

            # Use edited response if present
            response = data.get("LLM_response_editer", "")

            actions = data.get("actions", [])
            if not actions:
                # Some JSONs (e.g., Plan prompts) may not contain actions but still
                # carry a useful instruction + screenshot. We keep these with a
                # dummy action.
                actions = [{}]

            # Iterate over *all* actions to maximize sample count --------------
            for act in actions:
                # Unify different schema fields
                action_type = act.get("type") or act.get("action_type")

                # ---------------- Mouse-based actions ----------------
                if action_type in {"click", "double_click", "MouseAction", "drag"}:
                    # If using the new schema ensure valid sub-type
                    if action_type == "MouseAction":
                        # Accept click / double_click / drag / move
                        mouse_subtype = act.get("mouse_action_type")
                        if mouse_subtype not in {"click", "double_click", "drag", "move"}:
                            continue
                        pos = act.get("mouse_position", {})
                        x, y = pos.get("width"), pos.get("height")
                    else:  # legacy fields
                        x, y = act.get("x"), act.get("y")

                    if x is None or y is None:
                        continue

                    w, h = data.get('video_width', 1), data.get('video_height', 1)
                    x_norm, y_norm = x / max(w, 1), y / max(h, 1)
                    action_vec = [x_norm, y_norm, 1.0, 0.0]  # click flag set
                    goal_vec = [x_norm, y_norm] + [0.0] * 14

                # ---------------- Scroll actions ---------------------
                elif action_type == "scroll":
                    # No coordinate, embed direction via flags
                    direction = act.get("direction", "down") if isinstance(act, dict) else "down"
                    scroll_flag = 1.0 if direction == "down" else -1.0  # simple encoding
                    action_vec = [0.5, 0.5, 0.0, scroll_flag]
                    goal_vec = [0.0] * 16

                # ---------------- Keyboard actions -------------------
                elif action_type in {"type", "KeyboardAction", "WaitAction", None}:
                    # Represent as zero-action; keep textual instruction
                    action_vec = [0.0, 0.0, 0.0, 0.0]
                    goal_vec = [0.0] * 16

                else:
                    # Unsupported or high-level plan action – still retain as
                    # a language-only sample with a dummy action.
                    action_vec = [0.0, 0.0, 0.0, 0.0]
                    goal_vec = [0.0] * 16

                # Copy screenshot once per metadata entry to allow parallel dataloading
                idx = len(meta)
                img_path_dst = out_dir / 'images' / f'{idx:06d}.png'
                if not img_path_dst.exists():
                    copy2(img_path_src, img_path_dst)

                warp = [0.5, 0.5] + [0.0] * (warp_dim - 2)
                instruction_text = data.get("task_prompt_en") or data.get("task_prompt") or response
                meta.append({
                    'image': str(img_path_dst.relative_to(out_dir)),
                    'instruction_text': instruction_text,
                    'action': action_vec,
                    'warp': warp,
                    'goal': goal_vec,
                })

    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('Converted', len(meta), 'samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert ScreenAgent dataset')
    parser.add_argument('src_dir', help='ScreenAgent root directory')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--warp_dim', type=int, default=16, help='Desired length of warp vector (default 16)')
    args = parser.parse_args()
    convert(Path(args.src_dir), Path(args.out_dir), warp_dim=args.warp_dim) 