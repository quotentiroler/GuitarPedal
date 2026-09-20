#!/usr/bin/env python3
#
# The arithmetic behind comparing a model against a recording.
#
# No network and no bench.  The one that matters is coherence_db()
# telling a filter from a nonlinearity: that is compare-real.py's whole
# claim, and a measure that absorbed clipping would agree with anything.
#
import os
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio
import targets as T
import tonetwist

FS = audio.RATE
fails = []


def check(name, ok, detail=""):
    print("%-56s %s%s" % (name, "ok" if ok else "FAIL",
                          "" if ok else "   " + detail))
    if not ok:
        fails.append(name)


def write_wav(path, x, tag, ch=1, bits=32):
    x = np.asarray(x)
    if tag == audio.WAV_FLOAT:
        body = x.astype("<f4").tobytes()
    elif bits == 24:
        v = np.round(x * (2 ** 23 - 1)).astype(np.int32)
        b = np.empty((len(v), 3), dtype=np.uint8)
        b[:, 0], b[:, 1], b[:, 2] = v & 255, (v >> 8) & 255, (v >> 16) & 255
        body = b.tobytes()
    else:
        body = (x * (2 ** 31 - 1)).astype("<i4").tobytes()
    align = ch * bits // 8
    fmt = struct.pack("<HHIIHH", tag, ch, FS, FS * align, align, bits)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", len(body)) + body)


def noise(n, seed=1):
    return np.random.default_rng(seed).standard_normal(n) * 0.2


def band_limited(n, seed=1, top=20000.0):
    """Noise that stops where audio stops.

    Nothing that reaches Nyquist can be shifted by a fraction of a
    sample and back exactly, and no recording here does.
    """
    x = noise(n, seed)
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(n, 1.0 / FS) > top] = 0.0
    return np.fft.irfft(X, n)


# The recordings are float wav, which the wave module refuses.
with tempfile.TemporaryDirectory() as d:
    want = noise(4096)
    p = os.path.join(d, "f32.wav")
    write_wav(p, want, audio.WAV_FLOAT)
    got = audio.wav(p)
    check("float wav reads back the samples written",
          got.shape == want.shape and np.max(np.abs(got - want)) < 1e-6,
          "worst %.2e" % np.max(np.abs(got - want.astype("<f4"))))

    p = os.path.join(d, "i32.wav")
    write_wav(p, want, audio.WAV_PCM)
    got = audio.wav(p)
    check("32-bit pcm still reads back",
          np.max(np.abs(got - want)) < 1e-6)

    # pOD-set is 24-bit, which numpy has no dtype for.
    p = os.path.join(d, "i24.wav")
    write_wav(p, want, audio.WAV_PCM, bits=24)
    got = audio.wav(p)
    check("24-bit pcm reads back, sign and all",
          len(got) == len(want) and np.max(np.abs(got - want)) < 2e-7
          and got.min() < -0.1, "worst %.2e" % np.max(np.abs(got - want)))

    # A part-fetched member has a whole-file header and a short data
    # chunk; refusing it would mean fetching gigabytes for 20 seconds.
    p = os.path.join(d, "cut.wav")
    write_wav(p, want, audio.WAV_FLOAT)
    whole = os.path.getsize(p)
    with open(p, "r+b") as f:
        f.truncate(whole - 2048)
    got = audio.wav(p)
    check("a truncated download reads as far as it got",
          len(got) == len(want) - 512,
          "got %d of %d" % (len(got), len(want)))

    p = os.path.join(d, "stereo.wav")
    write_wav(p, np.repeat(want[:, None], 2, axis=1).ravel(),
              audio.WAV_FLOAT, ch=2)
    got = audio.wav(p)
    check("stereo comes back two dimensional", got.shape == (len(want), 2))

# Signed alignment.
x = band_limited(8192)
for shift in (3.7, -2.5, 0.0, 11.0, -17.25):
    y = audio.delay(x, shift)
    got = audio.lag_samples(x, y, 64)
    check("lag_samples finds %+.2f samples to a tenth" % shift,
          abs(got - shift) < 0.15, "got %+.3f" % got)

got = audio.delay_samples(x, audio.delay(x, -2.5), 64)
check("delay_samples cannot express an early signal at all",
      got >= 0.0 and abs(got + 2.5) > 1.0, "got %+.3f" % got)

# A tenth of a sample is a -42 dB residual, so the peak is where the
# search starts and not where it stops.
for shift in (0.37, -0.5, 0.0):
    y = audio.delay(x, shift)
    frac, g, _ = audio.fit_delay(y, x, 512)
    check("fit_delay finds %+.2f samples to a hundredth" % shift,
          abs(frac + shift) < 0.01 and abs(g - 1.0) < 1e-3,
          "got %+.4f, gain %.4f" % (-frac, g))

y = audio.delay(x, 4.3)
back = audio.delay(y, -4.3)
c = slice(512, len(x) - 512)
check("delay() is its own inverse", audio.null_db(back[c], x[c]) < -110.0,
      "%.1f dB" % audio.null_db(back[c], x[c]))

# The claim it rests on: a filter is absorbed, clipping is not, and the
# two differ from x by about the same amount.
n = 1 << 17
x = noise(n, seed=7)
filtered = np.convolve(x, np.array([0.5, 0.3, -0.2, 0.1]), mode="same")
clipped = np.tanh(x * 6.0) / 6.0

lin = audio.coherence_db(x, filtered)
non = audio.coherence_db(x, clipped)
check("a linear filter is absorbed", lin < -60.0, "%.1f dB" % lin)
check("clipping is not absorbed", non > -25.0, "%.1f dB" % non)
check("and the two are told apart by a wide margin", non - lin > 30.0,
      "%.1f dB apart" % (non - lin))

# A plain null cannot tell them apart, which is why both are reported.
g = float(np.dot(filtered, x) / np.dot(x, x))
check("a plain null calls the filtered one a mismatch too",
      audio.null_db(x * g, filtered) > -25.0,
      "%.1f dB" % audio.null_db(x * g, filtered))

# Directory names, and the settings left out.
got = tonetwist._controls("V100_F050_D030_MNormal")
check("a setting directory parses",
      got == {"V": "100", "F": "050", "D": "030", "M": "Normal"}, str(got))
check("something that is not one is refused",
      tonetwist._controls("trainval") == {})

ref = T.target("rat")["reference"]
check("Normal is the silicon pair", ref["knobs"](got)["Mode"] == 0)
check("Turbo is the LED pair",
      ref["knobs"](dict(got, M="Turbo"))["Mode"] == 2)
try:
    ref["knobs"](dict(got, M="Solo"))
    check("Solo has no counterpart and is refused", False, "it was accepted")
except KeyError:
    check("Solo has no counterpart and is refused", True)

check("a knob position becomes a fraction of travel",
      ref["knobs"](got)["Distortion"] == 0.30
      and ref["knobs"](got)["Filter"] == 0.50)

check("volume is not read off the recording",
      ref["knobs"](dict(got, V="100"))["Volume"]
      == ref["knobs"](dict(got, V="050"))["Volume"])

print()
if fails:
    print("test-real: %d failed" % len(fails))
    for f in fails:
        print("  %s" % f)
    sys.exit(1)
print("test-real: ok")
