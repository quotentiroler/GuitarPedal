#!/usr/bin/env python3
#
# How far apart are two real analog pedals?
#
# compare-real.py says how far the model is from one pedal, and cannot
# say whether that is far.  A model sits at the midpoint of two pedals
# and no closer to both, so what two of them cost each other is the
# ceiling on what matching one can mean.
#
# Matched knob positions are not matched drive - the Blues Driver reads
# -29 dB against clean at gain 2 and -9 at gain 5 - so each pedal is
# taken at whichever setting is nearest a stated drive, and that drive
# is printed beside the answer.
#
import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio
import parity
import podset

FS = audio.RATE
GUARD = FS // 10
SECONDS = 19.0

# Where the clamp is clearly working, in coherence against the clean
# input: -30 dB is a filter, -6 dB is hard clipping.
DRIVE = -9.0

SETTINGS = ("g2_t2", "g5_t2")
PEDALS = ("tubedreamer", "plumes", "zendrive", "dumkudo", "janray",
          "ktr", "bluesdriver", "ocd", "redllama")


def load(path, seconds):
    return np.asarray(audio.wav(path)[:int(seconds * FS)], dtype=np.float64)


def compare(a, b):
    """(null, shape) for b against a, gain and delay fitted away."""
    lag = audio.lag_samples(a, b, 4000)
    s = np.roll(a, int(round(lag)))
    _, g, s = audio.fit_delay(s, b, GUARD)
    c = slice(GUARD, len(b) - GUARD)
    return audio.null_db(g * s[c], b[c]), audio.coherence_db(s[c], b[c])


def at_drive(pedals, settings, seconds, target):
    """Each pedal at whichever setting is closest to 'target' drive."""
    dry = load(podset.dry(seconds=seconds + 1), seconds)
    chosen = {}
    for p in pedals:
        best = None
        for s in settings:
            gain, tone = s.split("_")
            y = load(podset.take(p, gain, tone, seconds=seconds + 1), seconds)
            _, shape = compare(dry, y)
            if best is None or abs(shape - target) < abs(best[0] - target):
                best = (shape, s, y)
        chosen[p] = best[2]
        print("  %-14s %-8s %6.1f dB against clean"
              % (p, best[1], best[0]), flush=True)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pedals", default=",".join(PEDALS))
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--drive", type=float, default=DRIVE,
                    help="the drive to compare at, in dB against clean")
    ap.add_argument("--bless", action="store_true",
                    help="write what was measured as the new baseline")
    args = ap.parse_args()

    pedals = [p.strip() for p in args.pedals.split(",") if p.strip()]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    known = podset.pedals()
    for p in pedals:
        if p not in known:
            raise SystemExit("compare-pedals: no pedal %r; have %s"
                             % (p, " ".join(sorted(known))))

    print("each pedal at the setting nearest %.0f dB of drive:" % args.drive)
    sig = at_drive(pedals, settings, args.seconds, args.drive)

    print("\npedal against pedal\n")
    print("%-13s %-13s %8s %8s" % ("a", "b", "null", "shape"))
    got = {}
    for a, b in itertools.combinations(sorted(sig), 2):
        null, shape = compare(sig[a], sig[b])
        got["%s %s" % (a, b)] = dict(null=round(null, 2),
                                     shape=round(shape, 2))
        print("%-13s %-13s %7.1f  %7.1f" % (a, b, null, shape), flush=True)

    nulls = [v["null"] for v in got.values()]
    shapes = [v["shape"] for v in got.values()]
    print("\n%d pairs" % len(got))
    print("  null   closest %.1f  median %.1f  furthest %.1f"
          % (min(nulls), float(np.median(nulls)), max(nulls)))
    print("  shape  closest %.1f  median %.1f  furthest %.1f"
          % (min(shapes), float(np.median(shapes)), max(shapes)))
    print("\n%s" % podset.CITE)

    was = parity.load("pedals")
    if was is None or args.bless:
        if was is not None:
            got = dict(was, **got)
        print("\ncompare-pedals: wrote %s" % parity.save("pedals", got))
        return 0

    worse, better, new = parity.check(was, got)
    for line in better + worse:
        print("  moved    %s" % line)
    for line in new:
        print("  new      %s  (--bless to record it)" % line)
    if worse or better:
        print("\ncompare-pedals: %d moved - the recordings did not, so this "
              "is us" % (len(worse) + len(better)))
        return 1
    print("\ncompare-pedals: ok, %d pairs unchanged" % len(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
