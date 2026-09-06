"""Read-only publication audit. Reports secret locations without printing values."""
from pathlib import Path
import re,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1]
patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'AIza[0-9A-Za-z_-]{30,}',r'sk-[A-Za-z0-9_-]{30,}',r'"private_key"\s*:\s*"-----',r'https://[^\s/@]+:[^\s/@]+@github\.com']
files=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
findings=[];large=[]
for name in sorted(set(filter(None,files))):
 p=ROOT/name
 if not p.is_file():continue
 if p.stat().st_size>1024**2:large.append([name,p.stat().st_size])
 try:txt=p.read_text()
 except UnicodeDecodeError:findings.append([name,'binary file']);continue
 for i,line in enumerate(txt.splitlines(),1):
  if any(re.search(pattern,line) for pattern in patterns):findings.append([name,i,'possible credential'])
print(json.dumps(dict(files=sorted(set(filter(None,files))),large_over_1MiB=large,secret_or_binary_findings=findings),indent=2))
if findings or large:sys.exit(1)
