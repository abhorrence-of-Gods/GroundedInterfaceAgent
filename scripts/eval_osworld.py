import argparse, json, subprocess, sys, os, shutil, tempfile, datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser("Run OSWorld benchmark with a trained GIA checkpoint and summarise success rate")
    parser.add_argument("--ckpt", required=True, help="Path to GIA checkpoint (best.pt)")
    parser.add_argument("--vm_path", default=None, help="Path to VM for DesktopEnv (optional)")
    parser.add_argument("--domain", default="all", help="Single domain or 'all'")
    parser.add_argument("--result_dir", default="osworld_results", help="Directory to store results")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--provider", default="docker", help="vmware | docker | virtualbox | none")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    osworld_dir = Path(__file__).parent.parent / "OSWorld"
    (osworld_dir / "logs").mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        str(osworld_dir / "run.py"),
        "--model", "gia",
        "--ckpt", str(Path(args.ckpt).resolve()),
        "--result_dir", str(result_dir.resolve()),
        "--max_steps", str(args.max_steps),
    ]
    if args.vm_path:
        cmd += ["--path_to_vm", args.vm_path]
    if args.domain != "all":
        cmd += ["--domain", args.domain]
    if args.provider:
        cmd += ["--provider", args.provider]

    print("[OSWorld Eval] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=osworld_dir)

    # Aggregate success rate
    success = []
    for file in result_dir.rglob("result.txt"):
        try:
            val = float(file.read_text())
            success.append(val)
        except Exception:
            continue
    total = len(success)
    success_rate = sum(success)/total*100 if total>0 else 0.0
    summary = {
        "checkpoint": args.ckpt,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_tasks": total,
        "success_rate": success_rate,
    }
    summary_path = result_dir / "summary.json"
    json.dump(summary, summary_path.open("w"), indent=2)
    print(f"[OSWorld Eval] Success rate: {success_rate:.2f}% on {total} tasks. Summary saved to {summary_path}")

if __name__ == "__main__":
    main() 