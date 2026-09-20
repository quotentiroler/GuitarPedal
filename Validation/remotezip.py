#!/usr/bin/env python3
#
# Read one file out of a zip on a web server, without fetching the zip.
#
# The datasets here are per-device archives of a gigabyte or two and what
# is wanted is twenty seconds of one recording.  The member comes down in
# a single range request; letting zipfile drive the reads issues one per
# megabyte and Zenodo stalls partway through.
#
import io
import json
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

RETRIES = 5
TIMEOUT = 180


def fetch(url, lo, hi):
    """One range request, retried: a truncated response is not rare."""
    want = hi - lo + 1
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"Range": "bytes=%d-%d" % (lo, hi)})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data, rng = r.read(), r.headers.get("Content-Range", "")
            if len(data) == want:
                return data, rng
            last = "short: %d of %d bytes" % (len(data), want)
        except Exception as exc:
            last = exc
            if attempt == RETRIES - 1:
                raise
        time.sleep(1.0 + attempt)
    raise OSError("%s: %s" % (url, last))


class _Remote(io.RawIOBase):
    """Enough of a file for zipfile to find the central directory."""

    def __init__(self, url):
        self.url, self.pos, self.pulled = url, 0, 0
        _, rng = fetch(url, 0, 0)
        self.size = int(rng.rsplit("/", 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = (off, self.pos + off, self.size + off)[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        data, _ = fetch(self.url, self.pos, self.pos + n - 1)
        self.pulled += len(data)
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def index(url, cache=None):
    """{name: [offset, compressed size, method, size]}, read once."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            got = json.load(f)
        if isinstance(got, dict):
            return got
    z = zipfile.ZipFile(io.BufferedReader(_Remote(url), 1 << 20))
    out = {i.filename: [i.header_offset, i.compress_size, i.compress_type,
                        i.file_size]
           for i in z.infolist() if not i.filename.endswith("/")}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return out


def member(url, entry, out, nbytes=None):
    """One member into 'out', stopping after 'nbytes' of it.

    Returns how much came over the wire.  A deflated member has to be
    read from its start, so a prefix costs a prefix and no more.
    """
    off, csize, method, size = entry
    want = size if nbytes is None else min(nbytes, size)

    #
    # Audio barely deflates, so asking for a little over what is wanted
    # covers it; a member that inflates short is simply asked for whole.
    #
    grab = min(csize + 4096, int(want * 1.2) + 8192)
    blob, _ = fetch(url, off, off + grab - 1)
    body = _inflate(blob, method, want)
    if len(body) < want and grab < csize + 4096:
        blob, _ = fetch(url, off, off + csize + 4096 - 1)
        body = _inflate(blob, method, want)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".part", "wb") as f:
        f.write(body[:want])
    os.replace(out + ".part", out)
    return len(blob)


def _inflate(blob, method, want):
    """Past the local header, as much of the member as 'want' asks for."""
    n, m = struct.unpack("<HH", blob[26:30])
    data = blob[30 + n + m:]
    if not method:
        return data[:want]
    return zlib.decompressobj(-15).decompress(data, want)


if __name__ == "__main__":
    url, name = sys.argv[1], sys.argv[2]
    ix = index(url)
    if name == "-":
        for k in sorted(ix):
            print("%12d  %s" % (ix[k][3], k))
    else:
        got = member(url, ix[name], sys.argv[3],
                     int(sys.argv[4]) if len(sys.argv) > 4 else None)
        print("%s: %.2f MB over the wire" % (sys.argv[3], got / 1e6))
