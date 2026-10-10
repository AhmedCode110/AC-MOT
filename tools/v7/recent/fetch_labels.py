"""
Extract annotation members (gt.txt, seqinfo.ini, labels, seqmaps) from the
official dataset archives by HTTP range requests (tools/v7/kitti/remotezip.py),
without downloading the images. Every member is CRC-checked by zipfile.

  python fetch_labels.py <url> <dest_dir> <regex on member names> [<strip prefix>]
"""
import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kitti"))
from remotezip import RangeFile  # noqa: E402

url, dest, pat = sys.argv[1], Path(sys.argv[2]), re.compile(sys.argv[3])
strip = sys.argv[4] if len(sys.argv) > 4 else ""
z = zipfile.ZipFile(RangeFile(url, block=4 << 20))
n = 0
for i in z.infolist():
    if i.is_dir() or not pat.search(i.filename):
        continue
    name = i.filename[len(strip):] if strip and i.filename.startswith(strip) else i.filename
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(z.read(i))
    n += 1
print(f"extracted {n} members from {url}")
