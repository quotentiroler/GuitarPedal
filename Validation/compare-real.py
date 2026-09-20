#!/usr/bin/env python3
#
# The model against a recording of a real pedal - see tonetwist.py.
#
# Everything else here compares the model against something we also
# wrote.  This is the only thing in the tree that has been near a pedal.
#
# The recording's input and output levels are not in the file.  The
# output one is fitted; the input one decides how hard the clamp is
# driven, so it is searched and reported.
#
# 'null' is the residual after gain and delay.  'shape' is what survives
# fitting a filter too: a wrong component value is a filter away from
# being right and a wrong clipping curve is not.  --notch measures the
# same null between two settings of the real pedal, which is what a
# decibel is worth here.
#
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio
import bench as B
import parity
import targets as T
import tonetwist

FS = audio.RATE

# Past the second sync impulse.
START = tonetwist.DRY["markers"][1] + FS // 2

SECONDS = 20.0

# The delay ramp is circular; its wrap is not measured.
GUARD = FS // 10

LEVELS = np.arange(-24.0, 19.0, 3.0)


def material(source, seconds):
    """The dry passage and however many takes of it, all the same length."""
    n = int(seconds * FS)
    x = audio.wav(tonetwist.dry(seconds=(START + n) / FS, source=source))
    return x[START:START + n]


def take(device, controls, source, seconds):
    n = int(seconds * FS)
    y = audio.wav(tonetwist.take(device, controls, seconds=(START + n) / FS,
                                 source=source))
    return y[START:START + n]


def model(t, knobs, x, gain_db):
    """The audio core on the passage at one drive; None if the bench clipped.

    The search walks past the ceiling on purpose, and where the ceiling
    is depends on the clamp, which is one of the things being compared.
    """
    y, _, info = B.run(T.pot_args(t, knobs), x * 10.0 ** (gain_db / 20.0),
                       warmup=B.settle())
    if info.get("clipped"):
        return None
    return np.asarray(y[:len(x)], dtype=np.float64)


def against(y, real, lag=None):
    """(null, shape, output gain, delay) for one model run.

    A correlation peak is good to a tenth of a sample and a tenth of a
    sample is a -42 dB floor, so the fraction is fitted rather than read.
    """
    if lag is None:
        lag = audio.lag_samples(y, real, 4000)
    s = np.roll(y, int(round(lag)))
    frac, g, s = audio.fit_delay(s, real, GUARD)
    c = slice(GUARD, len(real) - GUARD)
    null = audio.null_db(g * s[c], real[c])
    return null, audio.coherence_db(s[c], real[c]), g, lag + frac


def drive(t, knobs, x, real, levels=LEVELS):
    """The input level that fits best, and what it fits to."""
    runs, lag, clipped = [], None, 0
    for db in levels:
        y = model(t, knobs, x, float(db))
        if y is None:
            clipped += 1
            continue
        if lag is None:
            lag = audio.lag_samples(y, real, 4000)
        runs.append((float(db), np.roll(y, int(round(lag)))))
    if not runs:
        raise SystemExit("compare-real: every level clipped the bench; the "
                         "search window does not fit this setting")

    # Fitted once and held: a fraction of a sample is 45 degrees at
    # 10 kHz, so refitting per level lets the fit pick the winner.
    frac, _, _ = audio.fit_delay(runs[len(runs) // 2][1], real, GUARD)

    c = slice(GUARD, len(real) - GUARD)
    best = None
    for db, s in runs:
        s = audio.delay(s, frac)
        g = float(np.dot(s[c], real[c]) / max(np.dot(s[c], s[c]), 1e-30))
        null = audio.null_db(g * s[c], real[c])
        if best is None or null < best[1]:
            best = (db, null, g, s)
    if best[0] in (float(levels[0]), float(levels[-1])):
        print("     (best at the edge of the search - widen LEVELS)")
    shape = audio.coherence_db(best[3][c], real[c])
    return (best[0], best[1], shape, best[2], lag + frac), clipped


def measure(name="rat", source=tonetwist.DEFAULT_SOURCE, seconds=SECONDS,
            only=None):
    t = T.target(name)
    ref = t.get("reference")
    if not ref:
        raise SystemExit("compare-real: target %r names no reference" % name)

    x = material(source, seconds)
    out = {}
    for controls, knobs in T.reference_settings(t):
        tag = " ".join("%s=%s" % kv for kv in sorted(controls.items()))
        if only and tag not in only:
            continue
        print("  %s" % tag, flush=True)
        real = take(ref["device"], controls, source, seconds)
        (db, null, shape, g, lag), clipped = drive(t, knobs, x, real)
        out[tag] = dict(null=round(null, 2), shape=round(shape, 2))
        print("     in %+5.1f dB  out x%.4f  lag %+.2f  null %6.1f dB  "
              "shape %6.1f dB%s"
              % (db, g, lag, null, shape,
                 "   (%d clipped)" % clipped if clipped else ""))
    return out


def notch(name="rat", source=tonetwist.DEFAULT_SOURCE, seconds=SECONDS):
    """What one position of a knob costs, between two real recordings.

    Only pairs differing in exactly one control, so the answer is a knob.
    """
    t = T.target(name)
    device = t["reference"]["device"]
    got = T.reference_settings(t)
    rows = []
    for i, (ca, _) in enumerate(got):
        for cb, _ in got[i + 1:]:
            moved = [k for k in ca if ca[k] != cb[k]]
            if len(moved) != 1:
                continue
            a = take(device, ca, source, seconds)
            b = take(device, cb, source, seconds)
            null, shape, _, _ = against(a, b)
            rows.append((moved[0], "%s->%s" % (ca[moved[0]], cb[moved[0]]),
                         null, shape))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", nargs="?", default="rat")
    ap.add_argument("--source", default=tonetwist.DEFAULT_SOURCE,
                    choices=tonetwist.SOURCES,
                    help="which dry passage; the default is the one the "
                         "dataset does not gain-modulate")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--only", default="", help="one setting tag, or several")
    ap.add_argument("--notch", action="store_true",
                    help="what one knob position costs, for scale")
    ap.add_argument("--bless", action="store_true",
                    help="write what was measured as the new baseline")
    args = ap.parse_args()

    B.refuse_if_stale()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    if args.notch:
        print("one knob position, between two recordings of the pedal:")
        for control, moved, null, shape in notch(args.target, args.source,
                                                 args.seconds):
            print("  %-10s %-14s null %6.1f dB   shape %6.1f dB"
                  % (control, moved, null, shape))
        return 0

    device = T.target(args.target)["reference"]["device"]
    print("%s against %s" % (args.target, tonetwist.DEVICES[device]["pedal"]))
    got = measure(args.target, args.source, args.seconds, only)
    print("\n%s" % tonetwist.CITE)

    name = "%s-real" % args.target
    was = parity.load(name)
    if was is None or args.bless:
        if was is not None:
            got = dict(was, **got)
        print("\ncompare-real: wrote %s" % parity.save(name, got))
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
        print("\ncompare-real: %d metric(s) got worse" % len(worse))
        return 1
    if better:
        print("\ncompare-real: ok, and %d improved - --bless to hold them"
              % len(better))
        return 0
    print("\ncompare-real: ok, %d settings unchanged" % len(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
