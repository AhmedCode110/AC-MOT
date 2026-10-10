"""
POST-FREEZE recent external system: C-TWiX (Miah et al., "Learning Data
Association for Multi-Object Tracking using Only Coordinates", Pattern
Recognition 160 (2025) 111169; github.com/Guepardow/TWiX @ 3cff9cc).

Both arms replay the authors' RELEASED detections (results.zip) with the
authors' RELEASED TWiX weights (weights.zip) and the official per-dataset
arguments of script/c-twix_<DS>.sh. The loop of src/tracker/c-twix.py::main
is reproduced line for line (first-frame initialisation, STA matching, LTA
matching, kill/update, new tracks, state update, save).

V7f arm: per frame, the frame's released detections (class filter only) go
through the frozen layer; kept candidates with the layer's scores form the
frame's ObsCollection after the host's own score and area filters, with
min_score <- decision.assoc and min_score_new <- decision.birth. Host
contract (V7_RECENT_EXTERNAL_PROTOCOL.md section 5): assoc = min_score,
birth = min_score_new, low = min_score (single stage), match 0.8 (C-TWiX has
no IoU gate; not mapped). C-TWiX reads no images, so no image cue is passed.

  python ctwix_v7.py --dataset MOT17|KITTIMOT|DanceTrack --system BASELINE|V7f --name N
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ACMOT = Path(os.environ.get("ACMOT_ROOT", Path(__file__).resolve().parents[3]))
EXT = Path(os.environ.get("ACMOT_EXT", Path.home() / "acmot_work/acmot_external"))
TWIX = EXT / "TWiX"
# official arguments: script/c-twix_MOT17.sh, c-twix_KT.sh, c-twix_DT.sh
OFFICIAL = {
    "MOT17": dict(subset="val_half", detection="bytetrack_x_mot17", min_score=0.50, min_area=128,
                  method_twix_1="twix_sta_mot17_half", theta_1=0.8,
                  method_twix_2="twix_lta_mot17_half", theta_2=-0.4, max_age=0.8, min_score_new=0.70),
    "KITTIMOT": dict(subset="val", detection="Permatrack", min_score=0.50, min_area=128,
                     method_twix_1="twix_sta_kittimot_half", theta_1=0.4,
                     method_twix_2="twix_lta_kittimot_half", theta_2=-0.6, max_age=0.8, min_score_new=0.50),
    "DanceTrack": dict(subset="val", detection="bytetrack_model", min_score=0.50, min_area=128,
                       method_twix_1="twix_sta_dancetrack", theta_1=-0.4,
                       method_twix_2="twix_lta_dancetrack", theta_2=-0.2, max_age=1.6, min_score_new=0.90),
}


def _cpu_autocast_fp16():
    """Compatibility only: the official code runs `torch.autocast(device_type=DEVICE)`,
    i.e. float16 on the authors' CUDA GPU. On CPU the default autocast dtype is
    bfloat16, which the official `.numpy()` call rejects; float16 is kept, as on GPU."""
    import torch
    orig = torch.autocast

    class Autocast(orig):
        def __init__(self, device_type, dtype=None, *args, **kw):
            if device_type == "cpu" and dtype is None:
                dtype = torch.float16
            super().__init__(device_type, dtype, *args, **kw)
    torch.autocast = Autocast


def load_ctwix():
    _cpu_autocast_fp16()
    os.chdir(TWIX / "src/tracker")           # the official code uses paths relative to src/tracker
    sys.path.insert(0, str(TWIX / "src/tracker"))
    sys.path.insert(0, str(TWIX / "src"))
    spec = importlib.util.spec_from_file_location("ctwix", TWIX / "src/tracker/c-twix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(OFFICIAL))
    ap.add_argument("--system", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=str(EXT / "runs/ctwix"))
    ap.add_argument("--scene", default=None)
    a = ap.parse_args()
    o = OFFICIAL[a.dataset]
    out_dir = Path(a.root) / f"{a.dataset}-{o['subset']}" / a.name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = a.system == "BASELINE"
    if not baseline:
        sys.path.insert(0, str(ACMOT))
        from acmot_v7 import HostContract, V7Layer, spec_from_dict
        cfg = json.loads((ACMOT / "configs/universal_acmot_policy_v7.json").read_text())
        assert a.system == cfg["system"], "post-freeze: only the frozen policy may run"
        spec = spec_from_dict(cfg["spec"])
        host = HostContract(assoc=o["min_score"], birth=o["min_score_new"], low=o["min_score"], match=0.8)
    ct = load_ctwix()
    state = {"layer": None}

    class Gated(ct.Cascade_TWiX):
        """Official tracker; only the per-frame candidate set is supplied by the layer."""

        def load_detections(self):
            if state["layer"] is None:
                return super().load_detections()
            raw = self.scene.load_detections(
                Path(f"../../results/{self.scene.dataset_name}-{self.scene.subset}/Detection/{self.detection}/{self.scene.scene_name}.txt"),
                dict_classe=ct.IDX2COCO)
            for _, d in raw.items():
                d.keep_class(classes_to_keep=self.scene.dict_COI.values())
            self._raw = raw
            self.all_detections = {f: ct.ObsCollection() for f in raw}
            self.gate(self.scene.first_frame)

        def gate(self, frame):
            obs = list(self._raw[frame])
            names = sorted(set(self.scene.dict_COI.values()))
            b = np.array([[x.locator.xmin, x.locator.ymin, x.locator.xmax, x.locator.ymax] for x in obs],
                         np.float64).reshape(-1, 4)
            s = np.array([x.score for x in obs], np.float64)
            dec = state["layer"].step(b, s, None, classes=np.array([names.index(x.classe) for x in obs], int))
            self.min_score = float(dec.assoc)
            self.min_score_new = float(dec.birth)
            coll = ct.ObsCollection()
            for k, sc in zip(dec.keep, dec.scores):
                x = copy.copy(obs[int(k)])
                x.score = float(sc)
                coll.add_observation(x)
            coll.keep_high_score(min_score=self.min_score)
            coll.keep_big_objects(min_area=self.min_area)
            self.all_detections[frame] = coll
            state["audit"].append(dict(seq=self.scene.scene_name, frame=frame,
                                       **{k: v for k, v in dec.log.items() if isinstance(v, (int, float, str))}))

        def observe(self, frame):
            ids, boxes = [], []
            for oid, t in self.alive_tracks.items():
                if len(t.frames) and int(t.frames[-1]) == frame:
                    ids.append(int(oid))
                    boxes.append(list(t.coordinates[-1]))
            state["layer"].observe(np.asarray(boxes, np.float64).reshape(-1, 4), ids)

    state["audit"] = []
    scene = ct.init_scene(a.dataset, o["subset"])
    scenes = scene.list_scenes if a.scene is None else [a.scene]
    for scene_name in scenes:
        scene.load_scene(scene_name)
        state["layer"] = None if baseline else V7Layer(spec, host)
        tracker = Gated(scene=scene, detection=o["detection"], min_score=o["min_score"], min_area=o["min_area"],
                        method_twix_1=o["method_twix_1"], theta_1=o["theta_1"],
                        method_twix_2=o["method_twix_2"], theta_2=o["theta_2"],
                        max_age=o["max_age"], min_score_new=o["min_score_new"])
        if not baseline:
            tracker.observe(scene.first_frame)
        for frame in tracker.scene.list_frames[1:]:
            if not baseline:
                tracker.gate(frame)
            # --- official loop body (src/tracker/c-twix.py::main) ---
            tracker.matching(dict_of_tracks=tracker.alive_tracks,
                             obsColl=[obs for obs in tracker.all_detections.get(frame) if obs.score >= tracker.min_score],
                             frame=frame, theta=tracker.theta_1, model=tracker.model_twix_sta, WP=tracker.SWP)
            tracker.matching(dict_of_tracks=tracker.unmatched_tracks,
                             obsColl=tracker.unmatched_dets,
                             frame=frame, theta=tracker.theta_2, model=tracker.model_twix_lta, WP=tracker.LWP)
            tracker.kill_and_update_tracks()
            tracker.create_new_tracks(frame)
            tracker.update_states()
            # ---------------------------------------------------------
            if not baseline:
                tracker.observe(frame)
        tracker.save(folder=out_dir)
        print("done", scene_name, flush=True)
    if not baseline:
        json.dump(state["audit"], open(out_dir.parent / "audit.json", "w"))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
