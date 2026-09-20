#!/usr/bin/env python3
#
# Twenty-seven overdrive pedals, each fed the same file by the same robot.
#
# tonetwist.py has one pedal per family, so it can say how far the model
# is from a pedal but not how far two pedals are from each other.  The
# shared input here means any two of these null directly.
#
# CC BY-NC 4.0: fetched rather than vendored, and only numbers committed.
#
import csv
import os
import sys

import remotezip

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("PODSET_CACHE", os.path.join(HERE, ".cache", "podset"))

RECORD = "15389653"
CITE = ("Parametric Overdrive pedal dataset (pOD-set), "
        "https://zenodo.org/records/15389653, CC BY-NC 4.0")

RATE = 48000
WIDTH = 3

#
# 'a' is the passage, 'n' the noise floor, and s0/s1/s2 sweeps at -6,
# -12 and -24 dBFS.
#
KINDS = ("a", "n", "s0", "s1", "s2")
GAINS = tuple("g%d" % i for i in range(6))
TONES = tuple("t%d" % i for i in range(6))


def url_of(archive):
    return "https://zenodo.org/records/%s/files/%s" % (RECORD, archive)


def pedals():
    """{name: (brand, model)}, from the dataset's own metadata."""
    path = os.path.join(CACHE, "metadata.csv")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(CACHE, exist_ok=True)
        with urllib.request.urlopen(url_of("0_dataset_metadata.csv"),
                                    timeout=120) as r:
            body = r.read()
        with open(path, "wb") as f:
            f.write(body)
    with open(path, newline="") as f:
        return {row["datasetname"]: (row["brand"], row["model"])
                for row in csv.DictReader(f)}


def _pull(archive, name, out, seconds):
    if os.path.exists(out):
        return out
    ix = remotezip.index(url_of(archive),
                         os.path.join(CACHE, "%s.index.json" % archive))
    if name not in ix:
        raise SystemExit("podset: %s holds no %s" % (archive, name))
    want = None if seconds is None else 256 + int(seconds * RATE) * WIDTH
    got = remotezip.member(url_of(archive), ix[name], out, want)
    print("  fetched %s (%.1f MB over the wire)"
          % (os.path.basename(out), got / 1e6), file=sys.stderr)
    return out


def dry(kind="a", seconds=None):
    """The recording every pedal in the set was fed."""
    return _pull("0_input.zip", "input/%s_0-input.wav" % kind,
                 os.path.join(CACHE, "input.%s.wav" % kind), seconds)


def take(pedal, gain="g2", tone="t2", kind="a", seconds=None):
    """What one pedal did to it, at one gain and tone position."""
    name = "%s/%s_%s_%s_%s.wav" % (pedal, kind, pedal, gain, tone)
    out = os.path.join(CACHE, "%s.%s_%s.%s.wav" % (pedal, gain, tone, kind))
    return _pull("%s.zip" % pedal, name, out, seconds)


if __name__ == "__main__":
    for name, (brand, model) in sorted(pedals().items()):
        print("%-16s %s %s" % (name, brand, model))
    print("\n%s" % CITE)
