#!/usr/bin/env python3
#
# Can the bench name the component that is wrong?
#
# The breadboard bench measures a physical build and diffs it against the
# netlist it claims to be.  Saying "you disagree above 2 kHz" is easy and
# is not the useful part; saying "C2 is about three times nominal" is.
#
# This tests that claim without a breadboard.  A fault is INJECTED into
# the netlist - one component scaled by a known factor - and ngspice's
# answer for the faulty circuit is handed to the localiser as though it
# had come off a real build.  The localiser never sees which component
# moved, and has to name it.  So the test is a closed loop with a known
# answer, deterministic, and needs no hardware and no breadboard.
#
# It needs ngspice, which is the one thing here that is not python.
#
import os
import sys

try:
    import numpy as np
except ImportError:
    print("test-faults: SKIPPED - no numpy")
    sys.exit(0)

import faults
import ngspice
import targets

T = targets.target("rat")
NETLIST, NODE = T["netlist"], T["node"]
# Log-spaced across the band the pots offer.  25 rather than 9 because
# the residual is an RMS over these points, so noise averages down with
# more of them, and ac_src() runs one sweep and interpolates - the extra
# points are free.  Measured: top-1 under 0.05 dB RMS noise goes 90% -> 97%.
FREQS = list(np.geomspace(40.0, 12000.0, 25))

# One decade either side of nominal is the range a wrong part lands in:
# a decade-off resistor, a cap in the wrong decade, a part fitted backwards.
FACTORS = (0.1, 0.22, 0.47, 2.2, 4.7, 10.0)

TRIALS = (("R2", 4.7), ("C2", 4.7), ("R4", 0.22), ("C1", 0.22))

FAILED = []


def record(name, ok, detail=""):
    print("  %-46s %s%s" % (name, "ok" if ok else "FAIL",
                            "  " + detail if detail else ""))
    if not ok:
        FAILED.append(name)


def parse_values():
    print("parse_value")
    for text, want in (("100k", 1e5), ("4.7k", 4700.0), ("1n", 1e-9),
                       ("220n", 220e-9), ("10u", 1e-5), ("1meg", 1e6),
                       ("100p", 1e-10), ("47", 47.0), ("2.2u", 2.2e-6)):
        got = faults.parse_value(text)
        record("%-8s -> %g" % (text, want),
               abs(got - want) <= abs(want) * 1e-9, "got %g" % got)


def component_table():
    print("components")
    src = ngspice.netlist(NETLIST)
    c = faults.components(src)
    record("finds every R/C with a numeric value", len(c) == 23,
           "found %d" % len(c))
    record("includes the letter-suffixed ones",
           {"RPD", "Rld", "Rvol"} <= set(c))
    record("excludes the macromodel's expressions",
           not ({"Rp", "Cp"} & set(c)))
    record("R1 is 1 meg", abs(c["R1"]["value"] - 1e6) < 1.0)
    record("C4 is 100p", abs(c["C4"]["value"] - 100e-12) < 1e-15)
    record("R1 nodes are (vb, a)", c["R1"]["nodes"] == ("vb", "a"),
           str(c.get("R1", {}).get("nodes")))


def perturbation():
    print("perturb")
    src = ngspice.netlist(NETLIST)
    out = faults.perturb(src, "R2", 10.0)
    a, b = src.splitlines(), out.splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    record("changes exactly one line", len(diff) == 1, "changed %d" % len(diff))
    record("line count unchanged", len(a) == len(b))
    record("the changed one is R2",
           bool(diff) and b[diff[0]].split()[0] == "R2",
           b[diff[0]] if diff else "")
    record("R2 is now 10k", abs(faults.components(out)["R2"]["value"] - 1e4) < 1.0)


def localisation():
    print("localise  (inject a fault, then find it)")
    src = ngspice.netlist(NETLIST)
    params = T["spice"](T["knobs"])
    hits = 0
    for ref, factor in TRIALS:
        faulty = faults.perturb(src, ref, factor)
        measured = ngspice.ac_src(faulty, NODE, FREQS, params=params)
        ranked = faults.localise(src, NODE, FREQS, measured,
                                 refs=[r for r, _ in TRIALS],
                                 factors=FACTORS, params=params)
        top_ref, top_factor = ranked[0][1], ranked[0][2]
        ok = top_ref == ref
        hits += ok
        record("%s x%.2f -> %s x%.2f" % (ref, factor, top_ref, top_factor),
               ok, "" if ok else "named the wrong part")
    record("top-1 over every trial", hits == len(TRIALS),
           "%d/%d" % (hits, len(TRIALS)))


def realism():
    """A real leg is not a SPICE sweep: it has a gain offset and noise.

    The offsets are asserted, because a leg that cannot survive one is
    useless.  The noise is CHARACTERISED and only loosely asserted: what
    matters is the budget a bench has to hit, and that is a number to
    report rather than a threshold to pass.
    """
    print("realism  (what a physical measurement does to the ranking)")
    src = ngspice.netlist(NETLIST)
    params = T["spice"](T["knobs"])
    refs = sorted(faults.components(src))
    cands = faults.candidates(src, NODE, FREQS, refs, FACTORS, params)
    truth = {r: faults.candidates(src, NODE, FREQS, [r], [f], params)[1][2]
             for r, f in TRIALS}

    ref, factor = TRIALS[0]
    for offset in (6.0, -13.7):
        top = faults.match(cands, truth[ref] + offset)[0]
        record("survives a %+.1f dB gain offset" % offset, top[1] == ref,
               "named %s x%.2f" % (top[1], top[2]))

    rng = np.random.default_rng(20260913)
    print("    top-1 against %d candidates, %d draws per point:" % (
          len(cands), 20 * len(TRIALS)))
    rate = {}
    for sigma in (0.02, 0.05, 0.10, 0.20):
        hits = 0
        for r, _f in TRIALS:
            for _ in range(20):
                m = truth[r] + rng.normal(0.0, sigma, len(FREQS)) + 6.0
                hits += faults.match(cands, m)[0][1] == r
        rate[sigma] = 100.0 * hits / (20 * len(TRIALS))
        print("      %.2f dB RMS -> %3.0f%%" % (sigma, rate[sigma]))

    record("clean measurement is unambiguous", rate[0.02] == 100.0,
           "%.0f%%" % rate[0.02])
    record("0.05 dB RMS is a workable bench budget", rate[0.05] >= 90.0,
           "%.0f%%" % rate[0.05])


def main():
    for stage in (parse_values, component_table, perturbation,
                  localisation, realism):
        stage()
        print()
    if FAILED:
        print("test-faults: %d FAILED" % len(FAILED))
        for f in FAILED:
            print("    %s" % f)
        return 1
    print("test-faults: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
