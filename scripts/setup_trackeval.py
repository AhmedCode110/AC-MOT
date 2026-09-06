"""Install the exact upstream revision used by the original evaluator."""
import argparse,subprocess
from pathlib import Path
REV='12c8791b303e0a0b50f753af204249e622d0281a'
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
if not a.destination.exists():
 subprocess.run(['git','clone','https://github.com/JonathonLuiten/TrackEval.git',str(a.destination)],check=True)
 subprocess.run(['git','-C',str(a.destination),'checkout','--detach',REV],check=True)
actual=subprocess.check_output(['git','-C',str(a.destination),'rev-parse','HEAD'],text=True).strip()
if actual!=REV:raise RuntimeError('Existing TrackEval checkout has different revision; use a fresh destination')
print(a.destination.resolve())
