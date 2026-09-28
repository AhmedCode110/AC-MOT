"""Read members of an official remote ZIP archive through HTTP range requests
(no full download, no local copy of the images). The bytes are the original
archive bytes; every member is CRC-checked by zipfile on read."""
from __future__ import annotations

import io
import os

import requests


class RangeFile(io.RawIOBase):
    def __init__(self, url, block=16 << 20):
        self.url, self.block = url, block
        self.s = requests.Session()
        self.s.verify = os.environ.get("REQUESTS_CA_BUNDLE", True)
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.buf_start, self.buf = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start):
        end = min(self.size, start + self.block) - 1
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == end - start + 1:
                    self.buf_start, self.buf = start, r.content
                    return
            except requests.RequestException:
                pass
        raise IOError(f"range {start}-{end} failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            if not (self.buf_start <= self.pos < self.buf_start + len(self.buf)):
                self._fetch(self.pos)
            i = self.pos - self.buf_start
            chunk = self.buf[i:i + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)
