"""Build a new candidate notebook; never overwrite a previous experiment."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
old = json.loads((here.parent/'recorded_runs/AC_MOT_FULL17_Recorded_Replay_20260905.ipynb').read_text())
sources = [''.join(c['source']) for c in old['cells'] if c['cell_type']=='code'][:3]
# Preserve A0–A4 legacy runner as reference, and substitute controller only in A3.
sources[2] = sources[2].replace('        calibrator   = SmartCalibrator(',
    "        Calibrator = StableCalibrator if system.get('candidate_v11') else SmartCalibrator\n        calibrator   = Calibrator(")
assert 'Calibrator = StableCalibrator' in sources[2]
sources[0] = sources[0].replace('HALF=True', 'HALF=False')
sources[0] += '\n# v11 uses explicit FP32 to match recorded actual precision.\n'
sources[1] += '\n' + (here/'controller.py').read_text()
sources[2] += "\nABLATION_SYSTEMS[3]['candidate_v11'] = True\nABLATION_SYSTEMS[3]['name'] = 'A3_v11_Candidate'\n"
def cell(s):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.splitlines(True))
cells = [dict(cell_type='markdown', metadata={}, source=[
    '# AC-MOT v11 candidate — not final results\n',
    'A0–A2 retain reference logic; A3 tests low-confidence recovery and stable resolution. A4 remains legacy ablation.\n',
    'Do not tune on test-dev. Freeze settings on train/validation before a full test-dev run.\n',
    'Legacy recorder metrics are diagnostic only. Report TrackEval outputs from the last cell.\n'])]
cells += [cell(s) for s in sources]
cells += [cell('code_cells = '+repr([dict(source=s.splitlines(True)) for s in sources]))]
for name, path in [('record_full17.py', here.parent/'recorded_runs/record_full17.py'), ('evaluate_v11.py', here/'evaluate.py')]:
    cells.append(cell('%%writefile /content/'+name+'\n'+path.read_text()))
cells.append(cell("# Explicit execution cell: performs new T4 inference for all five systems.\n%run -i /content/record_full17.py\n"))
cells.append(cell("# Can also run separately against the OLD recorded run: no inference needed.\nimport subprocess, sys\nfrom pathlib import Path\nrepo = Path('/content/TrackEval_acmot')\nif not repo.exists():\n    subprocess.run(['git','clone','https://github.com/JonathonLuiten/TrackEval.git',str(repo)],check=True)\nsubprocess.run(['git','-C',str(repo),'checkout','12c8791b303e0a0b50f753af204249e622d0281a'],check=True)\nsubprocess.run([sys.executable,'/content/evaluate_v11.py',RECORDED_RUN_DIR,'--dataset',str(DATASET_ROOT),'--trackeval',str(repo),'--output',str(Path(RECORDED_RUN_DIR)/'trackeval_research_v11')],check=True)\n"))
target = here/'AC_MOT_v11_Candidate_T4.ipynb'
with target.open('x') as f:
    json.dump(dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(accelerator='GPU', colab=dict(gpuType='T4'), kernelspec=dict(name='python3',display_name='Python 3'))), f, indent=2)
print(target)
