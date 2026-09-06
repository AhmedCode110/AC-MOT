"""Run with %run -i after the FULL17 notebook setup/main run.

Records all five ablations, per-frame tracking, detector outputs and settings.
Existing v10 evaluator is retained and explicitly labelled as legacy/proxy.
Replay videos are rendered only from recorded tracking; rendering needs no YOLO.
"""
import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import traceback
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
assert len(VAL_SEQS) == 17
RECORD_ROOT = Path(globals().get("RECORDED_RUN_DIR", DRIVE_RESULTS / (
    "acmot_full17_recorded_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
RECORD_ROOT.mkdir(parents=True, exist_ok=True)
RECORDED_RUN_DIR = str(RECORD_ROOT)
print("RECORD_ROOT:", RECORD_ROOT, flush=True)

# Freeze implementation and configuration before any additional inference.
original_sources = ["".join(c["source"]) for c in code_cells[:3]]
for i, source in enumerate(original_sources, 1):
    target = RECORD_ROOT / f"original_cell_{i}.py"
    if not target.exists():
        target.write_text(source)
suite_source = Path(__file__).read_text() if "__file__" in globals() else ""
if suite_source:
    (RECORD_ROOT / "record_full17.py").write_text(suite_source)
frozen_systems = [dict(s) for s in ABLATION_SYSTEMS]
assert len(frozen_systems) == 5
from ultralytics.utils.checks import check_yaml
frozen_trackers = {k: Path(check_yaml(v)).read_text() for k, v in TRACKERS.items()}
fingerprint = hashlib.sha256(json.dumps([original_sources, frozen_systems,
    frozen_trackers, suite_source], sort_keys=True).encode()).hexdigest()
configuration = dict(fingerprint=fingerprint, gpu=torch.cuda.get_device_name(0),
    dataset=str(SEQ_DIR.parent), python=platform.python_version(),
    packages={p: importlib.metadata.version(p) for p in
              ["torch", "ultralytics", "motmetrics", "numpy", "pandas", "opencv-python-headless"]},
    systems=frozen_systems, tracker_yamls=frozen_trackers,
    ground_truth_filter=dict(categories=[1,4,5,6,9], score=1, occlusion_lt=2, truncation_lt=2),
    metric_protocol="Original v10: unthresholded IoU assignment; HOTA is a proxy, not official HOTA",
    fps_protocol="GPU synchronized inference/tracking including image read and scene analysis; excludes record serialization",
    replay_scope="Exact recorded predictions and visual replay; detector/model changes require new inference")
config_path = RECORD_ROOT / "configuration.json"
if config_path.exists():
    assert json.loads(config_path.read_text())["fingerprint"] == fingerprint, "Resume configuration differs"
else:
    atomic_json(config_path, configuration)
subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=open(RECORD_ROOT / "requirements-lock.txt", "w"), check=True)
atomic_json(RECORD_ROOT / "status.json", dict(status="validating_dataset"))

manifest = []
for seq in VAL_SEQS:
    frames = sorted(seq.glob("*.jpg"))
    ids = [int(p.stem) for p in frames]
    assert ids == list(range(1, len(frames)+1)), f"Missing frames: {seq.name}"
    ann = ANNOT_DIR / f"{seq.name}.txt"
    assert ann.is_file() and not load_gt(ann).empty, seq.name
    raw_gt = pd.read_csv(ann, header=None)
    assert int(raw_gt.iloc[:,0].max()) <= len(frames), seq.name
    file_hashes = {p.name: sha256(p) for p in frames}
    manifest.append(dict(sequence=seq.name, frames=len(frames), annotation_sha256=sha256(ann),
        annotation_rows=len(raw_gt), filtered_gt_rows=len(load_gt(ann)), frame_sha256=file_hashes))
atomic_json(RECORD_ROOT / "dataset_manifest.json", manifest)

# Instrument the unchanged system logic. One gzip JSONL record per input frame.
_detector_output = []
def capture_detections(predictor):
    global _detector_output
    b = predictor.results[0].boxes
    _detector_output = b.data.detach().cpu().numpy().tolist() if b is not None else []

def record_frame(model, seq, index, frame_path, pred_ids, pred_boxes, state, params, elapsed, result):
    b = result.boxes
    payload = dict(frame=index, filename=frame_path.name,
        ids=pred_ids.tolist(), boxes_xyxy=pred_boxes.tolist(),
        scores=b.conf.detach().cpu().numpy().tolist() if b is not None else [],
        classes=b.cls.detach().cpu().numpy().astype(int).tolist() if b is not None else [],
        detections_before_tracking=_detector_output,
        scene=dict(vars(state)), settings=dict(params, device=DEVICE, half_requested=HALF),
        elapsed_seconds=elapsed,
        actual_model_fp16=bool(getattr(getattr(model.predictor, "model", None), "fp16", False)))
    _record_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")

instrumented = original_sources[2]
assert instrumented.count("    model = YOLO(system['model'])") == 1
instrumented = instrumented.replace("    model = YOLO(system['model'])",
    "    model = YOLO(system['model'])\n    model.add_callback('on_predict_postprocess_end', capture_detections)")
instrumented = instrumented.replace("            t0  = time.perf_counter()",
    "            torch.cuda.synchronize()\n            t0  = time.perf_counter()")
instrumented = instrumented.replace("            times.append(time.perf_counter() - t0)",
    "            torch.cuda.synchronize()\n            times.append(time.perf_counter() - t0)")
needle = "            gt_f     = gt[gt['frame'] == idx]"
assert instrumented.count(needle) == 1
instrumented = instrumented.replace(needle,
    "            record_frame(model, seq, idx, fp, pred_ids, pred_boxes, state, params, times[-1], res[0])\n" + needle)
(RECORD_ROOT / "instrumented_runner.py").write_text(instrumented)
exec(compile(instrumented, str(RECORD_ROOT / "instrumented_runner.py"), "exec"), globals())

completed_rows = []
for system in frozen_systems:
    sysdir = RECORD_ROOT / system["name"]
    sysdir.mkdir(exist_ok=True)
    for seq in VAL_SEQS:
        row_path = sysdir / f"{seq.name}.metrics.json"
        record_path = sysdir / f"{seq.name}.frames.jsonl.gz"
        if row_path.exists() and record_path.exists():
            saved = json.loads(row_path.read_text())
            assert saved["fingerprint"] == fingerprint
            assert saved["record_sha256"] == sha256(record_path)
            completed_rows.append(saved["metrics"])
            continue
        atomic_json(RECORD_ROOT / "status.json", dict(status="running", system=system["name"],
            sequence=seq.name, completed=len(completed_rows), total=85))
        part = sysdir / f"{seq.name}.frames.partial.jsonl.gz"
        try:
            with gzip.open(part, "wt", encoding="utf-8") as _record_stream:
                result_df = run_system(system, [seq], RECORD_ROOT.name)
            assert len(result_df) == 1, f"Missing result: {seq.name}"
            with gzip.open(part, "rt") as f:
                saved_frames = [json.loads(line)["frame"] for line in f]
            count = len(list(seq.glob("*.jpg")))
            assert saved_frames == list(range(1, count+1)), f"Incomplete recordings: {seq.name}"
            part.replace(record_path)
            metric = result_df.iloc[0].to_dict()
            atomic_json(row_path, dict(fingerprint=fingerprint, record_sha256=sha256(record_path), metrics=metric))
            completed_rows.append(metric)
            pd.DataFrame(completed_rows).to_csv(RECORD_ROOT / "per_sequence_checkpoint.csv", index=False)
            print(f"RECORDED {len(completed_rows)}/85: {system['name']} / {seq.name}", flush=True)
        except Exception:
            atomic_json(RECORD_ROOT / "status.json", dict(status="failed", system=system["name"],
                sequence=seq.name, error=traceback.format_exc(), completed=len(completed_rows)))
            raise
        weight = Path(system["model"])
        if weight.is_file() and not (RECORD_ROOT / weight.name).exists():
            shutil.copy2(weight, RECORD_ROOT / weight.name)
            atomic_json(RECORD_ROOT / "weights.json", dict(filename=weight.name, sha256=sha256(weight)))

recorded_df = pd.DataFrame(completed_rows)
assert len(recorded_df) == 85
assert all(len(g) == 17 and set(g.sequence) == set(VAL_SEQS_NAMES) for _, g in recorded_df.groupby("system"))
recorded_summary = summarize(recorded_df)
recorded_df.to_csv(RECORD_ROOT / "per_sequence.csv", index=False)
recorded_summary.to_csv(RECORD_ROOT / "summary.csv", index=False)
atomic_json(RECORD_ROOT / "status.json", dict(status="recording_complete", completed=85, total=85))
print("RECORDING COMPLETE", RECORD_ROOT, flush=True)
print(recorded_summary.to_string(index=False))

# Standalone replay program is copied next to the records by the notebook.
replay_script = Path("/content/render_recorded.py")
if replay_script.exists():
    shutil.copy2(replay_script, RECORD_ROOT / "render_recorded.py")
    subprocess.run([sys.executable, str(replay_script), str(RECORD_ROOT)], check=True)
else:
    print("Recordings saved. Run render_recorded.py with RECORD_ROOT to produce videos.")
