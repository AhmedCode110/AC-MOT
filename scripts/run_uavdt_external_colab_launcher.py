#!/usr/bin/env python3
"""One-cell Colab bootstrap for the frozen UAVDT external-generalization run."""
from __future__ import annotations
import shutil, subprocess, sys, urllib.request
from pathlib import Path

BASE_COMMIT="5becc52a569f271ee8b73ce47c67d495de0d64a5"
TE_COMMIT="12c8791b303e0a0b50f753af204249e622d0281a"
RUNNER_COMMIT="17e9319036d403559ca005289750b7a62432fa1f"
REPO=Path("/content/AC-MOT")
TE=Path("/content/TrackEval")
RUNNER=Path("/content/run_uavdt_external_generalization_frozen.py")
URL=f"https://raw.githubusercontent.com/AhmedCode110/AC-MOT/{RUNNER_COMMIT}/scripts/run_uavdt_external_generalization_frozen.py"

print("="*92,flush=True)
print("UAVDT FROZEN EXTERNAL GENERALIZATION — COLAB BOOTSTRAP",flush=True)
print("="*92,flush=True)
print("[1/4] Installing pinned runtime...",flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","ultralytics==8.3.200","numpy==2.2.6","scipy==1.15.3","lap","opencv-python-headless","pandas","matplotlib"],check=True)

print("[2/4] Preparing frozen AC-MOT repo...",flush=True)
if REPO.exists(): shutil.rmtree(REPO)
subprocess.run(["git","clone","-q","https://github.com/AhmedCode110/AC-MOT.git",str(REPO)],check=True)
subprocess.run(["git","-C",str(REPO),"checkout","-q",BASE_COMMIT],check=True)
print("      AC-MOT HEAD:",subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),flush=True)

print("[3/4] Preparing pinned TrackEval...",flush=True)
if TE.exists(): shutil.rmtree(TE)
subprocess.run(["git","clone","-q","https://github.com/JonathonLuiten/TrackEval.git",str(TE)],check=True)
subprocess.run(["git","-C",str(TE),"checkout","-q",TE_COMMIT],check=True)
print("      TrackEval HEAD:",subprocess.check_output(["git","-C",str(TE),"rev-parse","HEAD"],text=True).strip(),flush=True)

print("[4/4] Downloading immutable UAVDT runner...",flush=True)
urllib.request.urlretrieve(URL,RUNNER)
print("      Runner commit:",RUNNER_COMMIT,flush=True)
print("      Runner path  :",RUNNER,flush=True)
print("="*92,flush=True)
print("Launching frozen UAVDT external test...",flush=True)
print("="*92,flush=True)
subprocess.run([sys.executable,str(RUNNER)],check=True)
