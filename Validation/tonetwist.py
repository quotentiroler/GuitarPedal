#!/usr/bin/env python3
#
# Recordings of real pedals, from people who own the pedal.
#
# The bench has one unit of one clone, and a claim about a circuit wants
# more than that.  capture-rat.py needs the pedal in the room; this does
# not.
#
# The audio is CC BY-NC 4.0, so it is fetched rather than vendored and
# the cache is ignored.  Only the numbers are committed.
#
# Costs about twelve megabytes rather than the three gigabytes the
# archive weighs: Zenodo honours range requests, so the index is read
# first, then one member, and only as far as the seconds wanted.
#
import os
import sys

import remotezip

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("TONETWIST_CACHE", os.path.join(HERE, ".cache", "tonetwist"))

CITE = ("ToneTwist AFx Dataset, Marco Comunita, Centre for Digital Music, "
        "Queen Mary University of London. CC BY-NC 4.0. "
        "https://github.com/mcomunita/tonetwist-afx-dataset")

DRY = dict(
    record="10455730",
    archive="DRY-with-markers.zip",
    # Sync impulses, at 0.5 s and 1.5 s.
    markers=(24000, 72000),
)

DEVICES = {
    "rodent": dict(
        record="10796378",
        archive="HarleyBenton-Rodent.zip",
        pedal="Harley Benton Rodent",
        note="a RAT clone; the dataset's own README says 'similar to Proco Rat'",
        controls=("Volume", "Filter", "Distortion", "Mode"),
    ),
}

# 'nam' is the only source with no pre-processing; the others get a
# random gain every five seconds.
SOURCES = ("nam", "idmt-gtr2", "idmt-gtr4-sg", "prvt-gtr", "yt-bass")
DEFAULT_SOURCE = "nam"

RATE = 48000
WIDTH = 4


def url_of(record, archive):
    return "https://zenodo.org/records/%s/files/%s" % (record, archive)


def _index(record, archive):
    return remotezip.index(url_of(record, archive),
                           os.path.join(CACHE, "%s.index.json" % archive))


def members(record, archive):
    """Every path in the archive, read once and remembered."""
    return sorted(_index(record, archive))


def pull(record, archive, member, out, seconds=None):
    """One member into 'out', stopping once 'seconds' of audio is there."""
    if os.path.exists(out):
        return out
    want = None if seconds is None else 256 + int(seconds * RATE) * WIDTH
    got = remotezip.member(url_of(record, archive),
                           _index(record, archive)[member], out, want)
    print("  fetched %s (%.1f MB over the wire)"
          % (os.path.basename(out), got / 1e6), file=sys.stderr)
    return out


def _pick(names, want, source):
    """The one member under a directory naming every control in 'want'."""
    hits = []
    for n in names:
        parts = n.split("/")
        if len(parts) < 3 or ".%s." % source not in parts[-1]:
            continue
        got = _controls(parts[-2])
        if got and all(got.get(k) == v for k, v in want.items()):
            hits.append(n)
    if len(hits) != 1:
        have = sorted({p.split("/")[-2] for p in names if "/" in p})
        raise SystemExit(
            "tonetwist: %d members match %s with source %r; the archive has %s"
            % (len(hits), want, source, " ".join(have)))
    return hits[0]


def _controls(directory):
    """'V100_F050_D030_MNormal' as {'V': '100', 'F': '050', ...}.

    Empty for anything else: parsed loosely, 'trainval' in the same
    archive reads as a control 't' set to 'rainval'.
    """
    fields = directory.split("_")
    if len(fields) < 2:
        return {}
    out = {}
    for field in fields:
        if len(field) < 2 or not field[0].isupper() or not field[1:].isalnum():
            return {}
        out[field[0]] = field[1:]
    return out


def dry(seconds=None, source=DEFAULT_SOURCE):
    """The clean recording every device in the dataset was fed."""
    names = members(DRY["record"], DRY["archive"])
    hit = [n for n in names if n.endswith("/%s.input.wav" % source)]
    if len(hit) != 1:
        raise SystemExit("tonetwist: no dry input named %r" % source)
    out = os.path.join(CACHE, "dry", "%s.wav" % source)
    return pull(DRY["record"], DRY["archive"], hit[0], out, seconds)


def take(device, setting, seconds=None, source=DEFAULT_SOURCE):
    """What the pedal did to it, at one setting."""
    d = DEVICES[device]
    names = members(d["record"], d["archive"])
    member = _pick(names, setting, source)
    tag = "_".join("%s%s" % kv for kv in sorted(setting.items()))
    out = os.path.join(CACHE, device, "%s.%s.wav" % (tag, source))
    return pull(d["record"], d["archive"], member, out, seconds)


def settings(device, source=DEFAULT_SOURCE):
    """Every setting the device was recorded at, as control dicts."""
    names = members(DEVICES[device]["record"], DEVICES[device]["archive"])
    seen = {}
    for n in names:
        parts = n.split("/")
        if len(parts) < 3 or ".%s." % source not in parts[-1]:
            continue
        got = _controls(parts[-2])
        if got:
            seen[parts[-2]] = got
    return [seen[k] for k in sorted(seen)]


if __name__ == "__main__":
    for name, d in sorted(DEVICES.items()):
        print("%s - %s\n  %s" % (name, d["pedal"], d["note"]))
        for s in settings(name):
            print("    %s" % " ".join("%s=%s" % kv for kv in sorted(s.items())))
    print("\n%s" % CITE)
