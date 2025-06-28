import argparse, re, json
from pathlib import Path

def extract_val_loss(stdout_path: Path) -> float | None:
    pattern = re.compile(r"val/total_loss: ([0-9.]+)")
    last = None
    for line in stdout_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            last = float(m.group(1))
    return last

def extract_success(result_dir: Path) -> float | None:
    summary = result_dir / "summary.json"
    if summary.exists():
        try:
            return json.loads(summary.read_text()).get("success_rate")
        except Exception:
            return None
    return None

def main():
    parser = argparse.ArgumentParser("Summarize sweep jobs (val loss & success rate)")
    parser.add_argument("root", help="sweep root directory (outputs/sweeps/xxxx)")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for job_dir in sorted(root.glob("job_*")):
        params = job_dir.name.split("_")
        stdout = job_dir/"stdout.txt"
        val_loss = extract_val_loss(stdout) if stdout.exists() else None
        result_dir = job_dir / "osworld_results"
        succ = extract_success(result_dir) if result_dir.exists() else None
        rows.append({"job": job_dir.name, "val_loss": val_loss, "success_rate": succ})
    rows.sort(key=lambda x: (x['success_rate'] is None, x['success_rate'] if x['success_rate'] is not None else float('inf')))
    # create symlink to best checkpoint if success_rate available
    best = next((r for r in rows if r['success_rate'] is not None), None)
    if best:
        best_dir = root / best['job']
        # find best checkpoint file (prefer safetensors)
        ckpt = None
        safes = list(best_dir.rglob('best.safetensors'))
        if safes:
            ckpt = safes[0]
        else:
            pts = list(best_dir.rglob('best.pt'))
            if pts:
                ckpt = pts[0]
        if ckpt:
            link_path = root / 'best_ckpt'
            try:
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(ckpt.resolve())
                print(f"Best checkpoint symlinked to {link_path}")
            except Exception as e:
                print("[Warn] Unable to create symlink:", e)
    out_path = root/"sweep_summary.json"
    json.dump(rows, out_path.open("w"), indent=2)
    print(f"Saved summary to {out_path}")
    for r in rows:
        print(r)

if __name__ == "__main__":
    main() 