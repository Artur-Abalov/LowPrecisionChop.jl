import csv
import struct
import numpy as np
from pychop.faultchop import FaultChop

# rmode: MATLAB-numbered 1-4 are deterministic and pychop-comparable.
# Mode 7 (RoundNearestTiesAway) has no pychop/MATLAB equivalent -- rows
# for it are NOT generated here (no external oracle exists); it is
# BigFloat-oracle-tested separately in Julia (kernel_bigfloat_oracle.jl).
# Modes 5,6 (stochastic) are excluded from this file for the same
# cross-language-RNG reason documented in test/data/README.md.
PRESETS = [
    ("fp8-e4m3", "fp8-e4m3", 4, 7),
    ("fp8-e5m2", "fp8-e5m2", 3, 15),
    ("fp16", "h", 11, 15),
    ("bfloat16", "b", 8, 127),
    ("fp32", "s", 24, 127),
]

RMODES = [1, 2, 3, 4]

def representative_inputs(t, emax, rng):
    emin = 1 - emax
    xmin = 2.0 ** emin
    xmax = (2 - 2.0 ** (1 - t)) * 2.0 ** emax
    xmins = xmin * 2.0 ** (1 - t)
    vals = [
        0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 3.14159265358979, -3.14159265358979,
        xmin, -xmin, xmin / 2, -xmin / 2, xmin / 4, -xmin / 4,
        xmins, -xmins, xmins * 8, -xmins * 8,
        xmax, -xmax, xmax * 0.999, xmax * 1.5, -xmax * 1.5,
        1e-300, -1e-300, 1e300, -1e300,
    ]
    for _ in range(40):
        exp = rng.integers(emin - 10, emax + 5)
        mant = 1.0 + rng.random()
        sign = 1.0 if rng.random() < 0.5 else -1.0
        vals.append(sign * mant * (2.0 ** float(exp)))
    return vals

def main():
    rng = np.random.default_rng(20260901)
    rows = []
    for label, prec, t, emax in PRESETS:
        for subnormal in (True, False):
            for rmode in RMODES:
                c = FaultChop(prec=prec, rmode=rmode, subnormal=subnormal, explim=1)
                for x in representative_inputs(t, emax, rng):
                    y = float(c(np.array([x], dtype=np.float64))[0])
                    rows.append([label, t, emax, rmode, int(subnormal), repr(x), repr(y)])

    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chop_golden.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["format_label", "t", "emax", "round_mode", "subnormal", "input", "expected"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows")

if __name__ == "__main__":
    main()
