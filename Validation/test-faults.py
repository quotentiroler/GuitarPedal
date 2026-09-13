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
FREQS = [40.0, 80.0, 160.0, 320.0, 640.0, 1250.0, 2500.0, 5000.0, 10000.0]

# One decade either side of nominal is the range a wrong part lands in:
# a decade-off resistor, a cap in the wrong decade, a part fitted backwards.
FACTORS = (0.1, 0.22, 0.47, 2.2, 4.7, 10.0)

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
    trials = (("R2", 4.7), ("C2", 4.7), ("R4", 0.22), ("C1", 0.22))
    for ref, factor in trials:
        faulty = faults.perturb(src, ref, factor)
        measured = ngspice.ac_src(faulty, NODE, FREQS, params=params)
        ranked = faults.localise(src, NODE, FREQS, measured,
                                 refs=[r for r, _ in trials],
                                 factors=FACTORS, params=params)
        top_ref, top_factor = ranked[0][1], ranked[0][2]
        ok = top_ref == ref
        hits += ok
        record("%s x%.2f -> %s x%.2f" % (ref, factor, top_ref, top_factor),
               ok, "" if ok else "named the wrong part")
    record("top-1 over every trial", hits == len(trials),
           "%d/%d" % (hits, len(trials)))


def main():
    for stage in (parse_values, component_table, perturbation, localisation):
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
