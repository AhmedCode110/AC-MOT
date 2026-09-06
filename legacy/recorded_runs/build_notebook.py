"""Build a self-contained T4 recording notebook from the existing v10 source."""
import ast
import base64
import json
import shutil
from datetime import datetime
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[2]
legacy_wrapper = root / "AC_MOT_v10_FULL17_runner_final.ipynb"
wrapper = json.loads(legacy_wrapper.read_text())
source = "".join(next(c for c in wrapper["cells"] if c["cell_type"] == "code")["source"])
assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == "NB_B64" for t in n.targets))
original = json.loads(base64.b64decode(ast.literal_eval(assignment.value)))
sources = ["".join(c["source"]) for c in original["cells"] if c["cell_type"] == "code"][:3]
setup = sources[0]
start = setup.index("VAL_SEQS_NAMES = [")
end = setup.index("by_name", start)
setup = setup[:start] + "VAL_SEQS_NAMES = [s.name for s in all_sequences]\n" + setup[end:]
setup = setup.replace("SETUP + 3 SEQUENCES", "SETUP + 17 SEQUENCES")
setup += "\nimport sys\nassert len(VAL_SEQS)==17\nassert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)\nprint('GPU:',torch.cuda.get_device_name(0))\n"
sources[0] = setup

def code(src):
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=src.splitlines(True))

cells = [dict(cell_type="markdown", metadata={}, source=[
    "# AC-MOT: full 17-sequence recorded experiment and replay\n",
    "T4 required. Five systems A0–A4. A3 is the adopted architecture; A4 is ablation only.\n",
    "Saves per-frame predictions/detections/settings, per-sequence checkpoints, source, weights and hashes.\n",
    "Metrics retain the legacy v10 evaluator: unthresholded IoU assignment and HOTA proxy. They are not official VisDrone benchmark scores.\n",
    "To resume, set RECORDED_RUN_DIR to the previous run directory before executing the recording cell.\n",
    "To render again without inference, run only the replay cell with an existing RECORDED_RUN_DIR.\n"])]
cells += [code(s) for s in sources]
cells += [code("code_cells = " + repr([dict(source=s.splitlines(True)) for s in sources]) + "\n")]
for name in ["record_full17.py", "render_recorded.py", "evaluate_recorded.py"]:
    cells.append(code("%%writefile /content/" + name + "\n" + (here/name).read_text()))
cells.append(code("# Optional resume: RECORDED_RUN_DIR = '/content/drive/MyDrive/VisDrone_Results/acmot_full17_recorded_TIMESTAMP'\n%run -i /content/record_full17.py\n"))
cells.append(code("# Replay only: no YOLO inference. Run this cell on an existing completed recording.\n# import subprocess, sys\n# subprocess.run([sys.executable, '/content/render_recorded.py', RECORDED_RUN_DIR], check=True)\n"))
cells.append(code("# Optional diagnostic with minimum IoU .5, using saved tracks only. Does not replace legacy results.\n# subprocess.run([sys.executable, '/content/evaluate_recorded.py', RECORDED_RUN_DIR, '--iou', '0.5'], check=True)\n"))
notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(
    accelerator="GPU", colab=dict(gpuType="T4", provenance=[]),
    kernelspec=dict(name="python3", display_name="Python 3")), cells=cells)
target = here / "AC_MOT_FULL17_Recorded_Replay_20260905.ipynb"
if target.exists():
    backups = here / "build_backups"
    backups.mkdir(exist_ok=True)
    shutil.copy2(target, backups / (datetime.now().strftime("%Y%m%d_%H%M%S_") + target.name))
target.write_text(json.dumps(notebook, indent=2))
print(target)
