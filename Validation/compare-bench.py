#!/usr/bin/env python3
#
# The model against a capture of the pedal it is a model of.
#
# analyse-rat-bench.py fits the clamp from a capture; nothing scored the
# result.  Both sides of the loop are in one frame, so there is no input
# level to search for - only the output gain is fitted.
#
# --knobs says what the Helios was set to: the capture records that as
# the sentence somebody typed.
#
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio
import bench as B
import loop
import parity
import targets as T

FS = audio.RATE
GUARD = 2400


def knobs_from(text, t):
    """'Distortion=0.45,Mode=0' onto the target's defaults."""
    out = dict(t["knobs"])
    for part in (p.strip() for p in text.split(",") if p.strip()):
        if "=" not in part:
            raise SystemExit("compare-bench: %r is not knob=value" % part)
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in out:
            raise SystemExit("compare-bench: %s has no %r; it has %s"
                             % (t["short"], name, " ".join(sorted(out))))
        out[name] = float(value)
    return out


def score(sent, back, t, knobs, tonal, guard=GUARD):
    """(null, shape, gain, delay) for the model against one capture."""
    sent = np.asarray(sent, dtype=np.float64)
    back = np.asarray(back, dtype=np.float64)
    n = min(len(sent), len(back))
    sent, back = sent[:n], back[:n]

    y, _, info = B.run(T.pot_args(t, knobs), sent.astype(np.float32),
                       warmup=B.settle())
    if info.get("clipped"):
        raise SystemExit("compare-bench: the bench clipped on this capture")
    y = np.asarray(y[:n], dtype=np.float64)

    # Tiny on purpose: a clipped burst correlates poorly and a wide
    # search locks onto a neighbouring cycle.
    lag = audio.lag_samples(y, back, 8)
    s = np.roll(y, int(round(lag)))
    frac, g, s = audio.fit_delay(s, back, guard)
    c = slice(guard, n - guard)

    # Not on a tone.  A burst carries ten harmonics and a filter has a
    # degree of freedom per harmonic, so coherence explains anything.
    span = (n - 2 * guard) // 3
    nper = 1 << max(8, int(np.floor(np.log2(max(span, 256)))))
    shape = (float("nan") if tonal or span < 512 else
             audio.coherence_db(s[c], back[c], nper=min(nper, 8192)))
    return audio.null_db(g * s[c], back[c]), shape, g, lag + frac


def measure(path, knobs_text, uncolour=False, only=None):
    with open(path) as f:
        cap = json.load(f)
    t = T.target("rat")
    knobs = knobs_from(knobs_text, t)

    print("capture: %s" % cap.get("setting", "(no setting recorded)"))
    print("knobs:   %s" % ", ".join("%s=%g" % kv for kv in sorted(knobs.items())))
    if cap.get("firmware"):
        print("board:   %s" % cap["firmware"])
    print()

    waves = cap.get("waves") or {}
    if not waves:
        raise SystemExit("compare-bench: %s holds no waveforms" % path)

    out = {}
    print("%-22s %8s %8s %8s %7s" % ("waveform", "null", "shape", "gain", "lag"))
    for key in sorted(waves):
        if only and key not in only:
            continue
        w = waves[key]
        back = w["back"]
        if uncolour:
            back = loop.uncolour(back)
        tonal = w.get("hz") is not None
        null, shape, g, lag = score(w["sent"], back, t, knobs, tonal)
        out[key] = dict(null=round(null, 2))
        if shape == shape:
            out[key]["shape"] = round(shape, 2)
        print("%-22s %7.1f  %8s  %7.4f  %+6.2f"
              % (key, null, "-" if shape != shape else "%.1f" % shape,
                 g, lag), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", help="a capture-rat.py json")
    ap.add_argument("--knobs", default="",
                    help="what the pedal was set to, as Distortion=0.45,...; "
                         "the capture only records the sentence you typed")
    ap.add_argument("--only", default="", help="one waveform key, or several")
    ap.add_argument("--uncolour", action="store_true",
                    help="undo the board's 9.28 Hz input corner first.  Safe "
                         "on bursts; see loop.py before using it on playing")
    ap.add_argument("--name", default="rat-bench",
                    help="which baseline to hold this against")
    ap.add_argument("--bless", action="store_true",
                    help="write what was measured as the new baseline")
    args = ap.parse_args()

    B.refuse_if_stale()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    got = measure(args.capture, args.knobs, args.uncolour, only)

    was = parity.load(args.name)
    if was is None or args.bless:
        if was is not None:
            got = dict(was, **got)
        print("\ncompare-bench: wrote %s" % parity.save(args.name, got))
        return 0

    worse, better, new = parity.check(was, got)
    print()
    for line in better:
        print("  better   %s" % line)
    for line in new:
        print("  new      %s  (--bless to record it)" % line)
    for line in worse:
        print("  WORSE    %s" % line)
    if worse:
        print("\ncompare-bench: %d metric(s) got worse" % len(worse))
        return 1
    if better:
        print("\ncompare-bench: ok, and %d improved - --bless to hold them"
              % len(better))
        return 0
    print("\ncompare-bench: ok, %d waveforms unchanged" % len(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
