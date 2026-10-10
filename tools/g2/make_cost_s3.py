"""configs/g2_compute_cost_s3.json from the one-host latency measurements of
the S3 workflow (research/final/g2/s3_cache/latency_<det>.json): full-frame
calls 512 ... 1344 px and one tile call, normalized by the full-frame 736 px
call on the same host. Also writes the round-robin tile schedule (one tile
per frame, cycling; the scene-blind control of S3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    src = ROOT / "research/final/g2/s3_cache"
    cfg = dict(about="G2 S3 compute axis: mean detector latency per call on one CPU host per detector "
                     "(research/final/g2/s3_cache/latency_<det>.json, 60 timed frames of uav0000086_00000_v), "
                     "divided by the full-frame 736 px call on the same host. tile_cost = one 2x2-grid tile call "
                     "at 736 px.", reference_setting=736, latency_ms={}, normalized_cost={}, tile_cost={}, hosts={})
    for det in ("yolov8", "rtdetr"):
        lat = json.loads((src / f"latency_{det}.json").read_text())
        ref = lat["full"]["736"]["mean_ms"]
        cfg["latency_ms"][det] = dict(full={k: v["mean_ms"] for k, v in lat["full"].items()},
                                      tile={k: v["mean_ms"] for k, v in lat["tile"].items()})
        cfg["normalized_cost"][det] = {k: round(v["mean_ms"] / ref, 4) for k, v in lat["full"].items()}
        cfg["tile_cost"][det] = round(lat["tile"]["736"]["mean_ms"] / ref, 4)
        cfg["hosts"][det] = (src / f"host_{det}.txt").read_text().strip()
    (ROOT / "configs/g2_compute_cost_s3.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(json.dumps(dict(normalized_cost=cfg["normalized_cost"], tile_cost=cfg["tile_cost"], hosts=cfg["hosts"])))


def round_robin(layer="V7f"):
    from tools.g2 import dev
    _, seqs_of = dev._splits()
    for det in ("yolov8", "rtdetr"):
        for seq in seqs_of(dev.SPLIT):
            n = dev.open_cache(det, seq).frames
            f = dev.out_dir() / "schedules" / layer / det / f"{seq}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            old = json.loads(f.read_text()) if f.exists() else {}
            old["RR"] = [[f"R736T{1 << ((t - 1) % 4)}", 1] for t in range(1, n + 1)]
            f.write_text(json.dumps(old))


if __name__ == "__main__":
    main() if sys.argv[1:] == ["cost"] else round_robin(*sys.argv[2:])
