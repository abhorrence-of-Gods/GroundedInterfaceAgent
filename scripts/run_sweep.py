import argparse, itertools, subprocess, os, sys, datetime, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser("Hyper-parameter sweep for GIA")
    parser.add_argument("--gld", nargs="*", type=float, default=[0.0, 0.005, 0.01, 0.02], help="goalwarp_logdet values")
    parser.add_argument("--alpha", nargs="*", type=float, default=[0.3, 0.5, 0.7], help="alpha_timewarp values")
    parser.add_argument("--warp_dim", nargs="*", type=int, default=[16], help="warp_dim variants")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent jobs")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_root = Path("outputs/sweeps") / timestamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.gld, args.alpha, args.warp_dim))
    procs = []
    for idx, (gld, alpha, wdim) in enumerate(combos):
        out_dir = sweep_root / f"job_{idx}_gld{gld}_a{alpha}_w{wdim}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "main.py",
            f"training.loss_weights.goalwarp_logdet={gld}",
            f"training.alpha_timewarp={alpha}",
            f"model.action_tower.warp_output_dim={wdim}",
            f"model.spacetime_encoder.input_dim={wdim}",
            f"model.spacetime_decoder.output_dim={wdim}",
            f"training.expected_warp_dim={wdim}",
            "training.num_epochs="+str(args.epochs),
            f"log_dir={out_dir}",
            f"checkpoint_root={out_dir}"
        ]
        env = os.environ.copy()
        log_file = open(out_dir/"stdout.txt", "w")
        print("Launching", " ".join(cmd))
        procs.append((subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env), log_file))
        while len([p for p,_ in procs if p.poll() is None]) >= args.jobs:
            # Wait for at least one job to finish
            for p, lf in procs:
                if p.poll() is not None:
                    lf.close()
                    procs.remove((p, lf))
                    break
    # Wait all
    for p, lf in procs:
        p.wait()
        lf.close()

    print("[Sweep] All jobs finished. Logs in", sweep_root)

if __name__ == "__main__":
    main() 