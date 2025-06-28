import json
import argparse
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO


def decode_image(data_uri: str):
    header, encoded = data_uri.split(',', 1)
    return Image.open(Path(BytesIO(base64.b64decode(encoded))))


def convert(src_dir: Path, out_dir: Path, warp_dim: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'images').mkdir(exist_ok=True)
    meta = []
    for json_file in src_dir.glob('**/*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Mind2Web sometimes stores a list of tasks in one file
        tasks = data if isinstance(data, list) else [data]

        for task in tasks:
            instr = task.get('instruction', '')
            steps = task.get('steps', [])
            if not steps:
                continue
            first = steps[0]
            img = None
            if 'screenshot' in first and first['screenshot']:
                try:
                    img = decode_image(first['screenshot'])
                except Exception as e:
                    print(f"Failed to decode inline image in {json_file}: {e}")
                    img = None

            # Fallback: external file reference
            if img is None and 'screenshot_path' in first:
                img_path_src = (json_file.parent / first['screenshot_path']).resolve()
                if img_path_src.exists():
                    try:
                        img = Image.open(img_path_src).convert('RGB')
                    except Exception as e:
                        print(f"Failed to open {img_path_src}: {e}")
                        img = None

            if img is None:
                # skip if still no image
                continue

            idx = len(meta)
            img_path = out_dir / 'images' / f'{idx:06d}.png'
            img.save(img_path)

            x_norm = first['mouse']['x'] / img.width
            y_norm = first['mouse']['y'] / img.height
            action = [x_norm, y_norm, 1.0, 0.0]
            warp = [0.5, 0.5] + [0.0] * (warp_dim - 2)

            tag = first.get('dom', {}).get('tag', '') if isinstance(first, dict) else ''
            tag_vocab = ['button','a','input','img','select','textarea','checkbox','radio','div','span']
            tag_onehot = [1 if tag.lower()==t else 0 for t in tag_vocab]
            goal = [x_norm, y_norm] + tag_onehot + [0.0]*(16-2-len(tag_onehot))

            meta.append({
                'image': str(img_path.relative_to(out_dir)),
                'instruction_text': instr,
                'action': action,
                'warp': warp,
                'goal': goal,
            })
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('Converted', len(meta), 'samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert Mind2Web to metadata format')
    parser.add_argument('src_dir', help='Path to Mind2Web json tasks')
    parser.add_argument('out_dir', help='Output directory for metadata format')
    parser.add_argument('--warp_dim', type=int, default=16, help='Desired length of warp vector (default 16)')
    args = parser.parse_args()
    convert(Path(args.src_dir), Path(args.out_dir), warp_dim=args.warp_dim) 