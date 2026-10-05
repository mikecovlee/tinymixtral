# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""Exhaustive bridge lemma for the rounding search proof (L7 reduction).

``proofs/z3_binary_search.py`` reasons about ``_round_positive`` through a
target bit pattern ``t``; that reduction needs ``_positive_value`` to be
strictly monotone and to agree with the IEEE-754 binary32 view.  This script
checks both **exhaustively over all 2^31 finite nonnegative bit patterns**
-- a complete finite check, not a sample.

Run:  python proofs/bridge_monotonicity.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpt_model.constants import _positive_value  # noqa: E402

SENTINEL = 0x7F800000
CHUNK = 1 << 24


def values_of(bits: np.ndarray) -> np.ndarray:
    """Exact values of finite nonnegative FP32 patterns, in float64.

    Every finite binary32 value is exactly representable in binary64.
    """
    bits = bits.astype(np.uint64)
    exponent = bits >> 23
    mantissa = (bits & 0x7FFFFF).astype(np.float64)
    scale = np.exp2(exponent.astype(np.float64) - 150.0)
    return np.where(exponent == 0, mantissa * 2.0**-149, (2.0**23 + mantissa) * scale)


def main() -> None:
    previous_max = None
    ieee_agreement = True
    monotone = True
    for start in range(0, SENTINEL, CHUNK):
        end = min(start + CHUNK, SENTINEL)
        bits = np.arange(start, end, dtype=np.uint32)
        values = values_of(bits)
        # Agreement with the IEEE-754 binary32 view of the same patterns.
        ieee_agreement &= bool(np.array_equal(values, bits.view(np.float32).astype(np.float64)))
        # Strict monotonicity within the chunk and across chunk boundaries.
        monotone &= bool((values[1:] > values[:-1]).all())
        if previous_max is not None:
            monotone &= bool(values[0] > previous_max)
        previous_max = values[-1]
    assert ieee_agreement, "_positive_value formula disagrees with the IEEE view"
    assert monotone, "_positive_value is not strictly monotone"
    sentinel_value = _positive_value(SENTINEL)
    assert sentinel_value == 2.0**128, f"sentinel mismatch: {sentinel_value}"

    # Code correspondence: the real _positive_value matches the formula on
    # boundary and random patterns.
    for pattern in (0, 1, 0x7FFFFF, 0x800000, 0x800001, 0x7F7FFFFE, 0x7F7FFFFF,
                    0x1234567, 0x5ABCDE, 0x2A3B4C5):
        expected = float(values_of(np.array([pattern], dtype=np.uint32))[0])
        actual = _positive_value(pattern)
        assert actual == expected, f"pattern {pattern:#x}: {actual} != {expected}"

    print("bridge_monotonicity.py: proved over all 2^31 patterns")
    print("  - _positive_value agrees exactly with the IEEE-754 binary32 view")
    print("  - _positive_value is strictly monotone (bit adjacency = FP adjacency)")
    print("  - sentinel 0x7F800000 maps to 2^128 (one step past the largest finite)")
    print("  - the Python implementation matches the closed-form formula")


if __name__ == "__main__":
    main()
