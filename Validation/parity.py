#!/usr/bin/env python3
#
# Does the model still agree with the deck as well as it did?
#
# compare-spice.py measures the disagreement and prints it.  Nothing
# holds it anywhere: a model change that makes the duty cycle worse
# reads the same as one that makes it better, and the numbers in
# Effects/rat.h are a comment rather than a claim anything checks.
#
# So the measurements go in a file and this compares against it.  A
# metric that got worse is a failure, a metric that got better is a
# baseline to rewrite with --bless.
#
# WHAT A METRIC HAS TO SAY ABOUT ITSELF
#
# A number is only a measurement where the thing producing it is valid,
# so each one carries the conditions it was taken under and refuses
# rather than reports when they do not hold.  alias_db is the example
# that earned this: it subtracts the harmonic grid, so a probe frequency
# dividing the sample rate makes every folded image land back on the
# grid and the answer is a confident zero.
#
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

import targets

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "parity", "%s.json")

# Slack for a metric to be called unchanged.  ngspice and the bench are
# both deterministic, so this absorbs float noise and nothing else.
TOL_DB = 0.05


def compare_spice():
    """compare-spice.py, whose name is not an identifier."""
    spec = importlib.util.spec_from_file_location(
        "compare_spice", os.path.join(HERE, "compare-spice.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_spice"] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics(cs, t, knobs):
    """What the model and the deck disagree by, at one setting."""
    lad, (hs, hm), al, lin = cs.one(t, knobs)
    lad, lin = np.asarray(lad), np.asarray(lin)
    out = {
        "small_signal_worst": float(np.max(np.abs(lin[:, 2] - lin[:, 1]))),
        "ladder_worst": float(np.max(np.abs(lad[:, 2] - lad[:, 1]))),
    }
    for i, h in enumerate(("h2", "h3", "h4", "h5")):
        out[h] = float(abs(hm[i] - hs[i]))

    #
    # The alias reading is only reported where it can see: compare-spice
    # probes at F0 and the grid it subtracts hides everything when the
    # sample rate is a multiple of it.
    #
    if abs(cs.FS / cs.F0 - round(cs.FS / cs.F0)) > 1e-9:
        out["alias_delta"] = float(al[1] - al[0])
    return out


def name_of(setting):
    return ",".join("%s=%g" % kv for kv in sorted(setting.items())) or "default"


def measure(target, only=None):
    cs = compare_spice()
    t = targets.target(target)
    out = {}
    for setting in t["settings"]:
        name = name_of(setting)
        if only and name not in only:
            continue
        knobs = dict(t["knobs"], **setting)
        print("  measuring %s" % name, flush=True)
        out[name] = metrics(cs, t, knobs)
    return out


def load(target):
    path = BASELINE % target
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["settings"]


def save(target, got):
    path = BASELINE % target
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        json.dump({"target": target, "settings": got}, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def check(was, got):
    """Every metric, against what it used to be.  Worse is a failure."""
    worse, better, new = [], [], []
    for name, now in sorted(got.items()):
        before = was.get(name)
        if before is None:
            new.append(name)
            continue
        for metric, value in sorted(now.items()):
            old = before.get(metric)
            if old is None:
                new.append("%s %s" % (name, metric))
            elif value > old + TOL_DB:
                worse.append("%-22s %-18s %.3f -> %.3f" % (name, metric, old, value))
            elif value < old - TOL_DB:
                better.append("%-22s %-18s %.3f -> %.3f" % (name, metric, old, value))
    return worse, better, new


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", nargs="?", default="rat")
    ap.add_argument("--only", default="", help="one setting name, or several")
    ap.add_argument("--bless", action="store_true",
                    help="write what was measured as the new baseline")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    got = measure(args.target, only)

    was = load(args.target)
    if was is None or args.bless:
        if was is not None:
            got = dict(was, **got)
        print("parity: wrote %s" % save(args.target, got))
        return 0

    worse, better, new = check(was, got)
    print()
    for line in better:
        print("  better   %s" % line)
    for line in new:
        print("  new      %s  (--bless to record it)" % line)
    for line in worse:
        print("  WORSE    %s" % line)

    if worse:
        print("\nparity: %d metric(s) got worse" % len(worse))
        return 1
    if better:
        print("\nparity: ok, and %d improved - --bless to hold the new ones"
              % len(better))
        return 0
    print("\nparity: ok, %d settings unchanged" % len(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
