#
# Which component is wrong?
#
# The bench can already say that a build and its netlist disagree.  That
# is the easy half and not the useful one: a curve that sags above 2 kHz
# is a symptom, and the thing somebody with a soldering iron needs is a
# reference designator.
#
# So the question is turned around.  Rather than trying to invert the
# measurement, every single-component fault that could explain it is
# simulated and the one that lands closest wins.  That is slow and
# completely robust: it needs no derivative, no assumption that the
# circuit is linear in its parts, and it cannot be fooled by a topology
# it was not told about, because every candidate IS the topology.
#
# One component at a time is the honest limit.  Two parts wrong at once
# is a bigger search and a much weaker claim, and a board with two faults
# usually announces itself anyway.
#
import re

import numpy as np

import ngspice

SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
      "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}

_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
_VALUE = re.compile(r"^(%s)\s*([a-zA-Z]*)$" % _NUM)
_DEVICE = re.compile(r"^([RCL]\w*)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)\s*$", re.M)


def parse_value(text):
    """'4.7k' -> 4700.0, '100p' -> 1e-10, '1meg' -> 1e6.

    SPICE reads the suffix greedily and ignores whatever trails it, so
    '100pF' is '100p' and '1kohm' is '1k'.  'meg' is checked before 'm'
    because 'm' is milli here and taking it first would be a billion-fold
    error that still looks like a number.
    """
    m = _VALUE.match(text.strip())
    if not m:
        raise ValueError("not a plain value: %r" % text)
    num, suffix = float(m.group(1)), m.group(2).lower()
    if not suffix:
        return num
    if suffix.startswith("meg"):
        return num * SI["meg"]
    return num * SI.get(suffix[0], 1.0)


def format_value(v):
    """A float back into something the deck will read the same way."""
    return "%.6g" % v


def components(src):
    """Every R/C/L with a plain numeric value, as {ref: {nodes, value}}.

    A device whose value is an expression - the op-amp macromodel's
    '{a0/gm}' and friends - is deliberately not a candidate: it is a
    modelling parameter rather than a part somebody can fit backwards.
    """
    out = {}
    for m in _DEVICE.finditer(src):
        try:
            value = parse_value(m.group(7))
        except ValueError:
            continue
        out[m.group(1)] = {"nodes": (m.group(3), m.group(5)), "value": value,
                           "raw": m.group(7)}
    return out


def perturb(src, ref, factor):
    """The same deck with one component scaled, and nothing else touched."""
    seen = []

    def swap(m):
        if m.group(1) != ref:
            return m.group(0)
        seen.append(ref)
        return "".join(m.group(1, 2, 3, 4, 5, 6)) + \
            format_value(parse_value(m.group(7)) * factor)

    out = _DEVICE.sub(swap, src)
    if len(seen) != 1:
        raise ValueError("%s matched %d devices, wanted 1" % (ref, len(seen)))
    return out


def residual(a, b):
    """How far apart two dB curves are, in dB RMS."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def localise(src, node, freqs, measured, refs=None, factors=(), params=None):
    """Rank single-component faults by how well each explains 'measured'.

    Returns [(residual_db, ref, factor)], closest first.  The nominal
    circuit is included as the (ref=None, factor=1.0) candidate, so a
    board that is simply correct says so rather than being made to pick a
    scapegoat.
    """
    if refs is None:
        refs = sorted(components(src))

    out = [(residual(ngspice.ac_src(src, node, freqs, params=params), measured),
            None, 1.0)]
    for ref in refs:
        for f in factors:
            cand = ngspice.ac_src(perturb(src, ref, f), node, freqs,
                                  params=params)
            out.append((residual(cand, measured), ref, f))
    out.sort(key=lambda r: r[0])
    return out
