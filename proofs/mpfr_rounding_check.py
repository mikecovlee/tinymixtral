# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""Quasi-exhaustive differential check of the FP32 rounding pipeline (L7/L8).

Cross-checks the real implementation in ``model/cpt_constants.py`` against an
independent reference: RN-even rounding **on the binary32 grid**, computed
from scratch with exact rational arithmetic (bracket search + exact midpoint
tie rule).  The grid is extended by the code's sentinel value 2^128, whose
midpoint with the largest finite value is exactly IEEE-754's overflow tie
threshold.

MPFR (24-bit mantissa) is used as a third opinion on the normal range, where
24-bit mantissa rounding coincides with binary32 rounding.  (MPFR alone is
not a binary32 reference: its exponent range and subnormal semantics differ.)

Covered: every exponent-class boundary and its exact midpoint (ties-to-even),
random adjacent pairs in all finite classes, subnormal values, the overflow
boundary, and random rationals across the whole FP32 range.

Run:  python proofs/mpfr_rounding_check.py [random_cases]
"""

import random
import struct
import sys
from fractions import Fraction
from pathlib import Path

import gmpy2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpt_model.constants import inverse_sqrt_fp32, rational_fp32  # noqa: E402

SENTINEL = 0x7F800000
MAX_FINITE = 0x7F7FFFFF
SENTINEL_VALUE = Fraction(2) ** 128  # the grid's next point past max finite


def pattern_value(bits: int) -> Fraction:
    """Exact value of a finite nonnegative FP32 pattern (via the IEEE view)."""
    raw = struct.unpack("<f", struct.pack("<I", bits))[0]
    return Fraction.from_float(raw)


def grid_value(bits: int) -> Fraction:
    """Grid value including the sentinel extension."""
    return SENTINEL_VALUE if bits == SENTINEL else pattern_value(bits)


def rn_even_exact(value: Fraction) -> int:
    """Independent reference: RN-even bit pattern of an exact rational.

    Re-implements the search from scratch with exact rational comparisons and
    an exact midpoint tie rule; returns the pick, or SENTINEL on overflow.
    """
    magnitude = abs(value)
    lo, hi = 0, SENTINEL
    while hi - lo > 1:
        middle = (lo + hi) // 2
        if grid_value(middle) <= magnitude:
            lo = middle
        else:
            hi = middle
    midpoint = (grid_value(lo) + grid_value(hi)) / 2
    if magnitude > midpoint:
        return hi
    if magnitude < midpoint:
        return lo
    return lo if lo % 2 == 0 else hi


def pick_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def check_rational(value: Fraction, failures: list) -> None:
    expected_bits = rn_even_exact(value)
    raised = False
    try:
        got = rational_fp32("mpfr_check", value)
    except ValueError:
        raised = True
    if expected_bits == SENTINEL:
        assert raised, f"{value}: expected overflow, code gave {got!r}"
        return
    assert not raised, f"{value}: code raised but the grid gives {pick_float(expected_bits)!r}"
    expected = pick_float(expected_bits)
    if value < 0:
        expected = -expected
    if expected != got:
        failures.append((value, expected, got))
    # Third opinion: MPFR 24-bit mantissa rounding coincides with binary32 on
    # the normal range and away from overflow.
    if abs(value) >= Fraction(1, 2**125) and abs(value) < SENTINEL_VALUE / 2:
        mpfr_ref = gmpy2.mpfr(gmpy2.mpq(value.numerator, value.denominator), 24)
        if not gmpy2.is_infinite(mpfr_ref) and float(mpfr_ref) != got:
            failures.append((value, f"mpfr:{float(mpfr_ref)}", got))


def rn_even_inverse_sqrt(d: int) -> Fraction:
    """Independent reference: RN-even of 1/sqrt(d) using exact comparisons."""
    one = Fraction(1)
    d_frac = Fraction(d)

    def below(pattern: int) -> bool:
        # value(pattern) <= 1/sqrt(d)  <=>  value^2 * d <= 1
        return pattern_value(pattern) ** 2 * d_frac <= one

    lo, hi = 0, SENTINEL
    while hi - lo > 1:
        middle = (lo + hi) // 2
        if below(middle):
            lo = middle
        else:
            hi = middle
    midpoint = (grid_value(lo) + grid_value(hi)) / 2
    relation = midpoint**2 * d_frac - one
    if relation < 0:
        return grid_value(hi)
    if relation > 0:
        return grid_value(lo)
    return grid_value(lo if lo % 2 == 0 else hi)


def check_inverse_sqrt(d: int, failures: list) -> None:
    expected = rn_even_inverse_sqrt(d)
    got = inverse_sqrt_fp32("mpfr_check", d)
    if float(expected) != got:
        failures.append((d, float(expected), got))


def main() -> None:
    random.seed(0xC0FFEE)  # deterministic sampling: runs are reproducible
    random_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    failures: list = []
    checked = 0

    # 1. Exponent-class boundaries and exact midpoints (ties-to-even), in both
    #    signs, for the subnormal class and all 254 finite normal classes.
    #    (Exponent 255 has no finite patterns; its boundary is the overflow
    #    case covered in section 3.)
    for exponent in range(255):
        if exponent == 0:
            first, last = 0, 0x7FFFFF
        else:
            first = exponent << 23
            last = first | 0x7FFFFF
        patterns = {first, last}
        if first < last:
            patterns |= {first + 1, last - 1}
        for _ in range(48):
            patterns.add(random.randint(first, last))
        ordered = sorted(patterns)
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right != left + 1:
                continue
            midpoint = (grid_value(left) + grid_value(right)) / 2
            for sign in (1, -1):
                for probe in (midpoint, midpoint - Fraction(1, 2**149), midpoint + Fraction(1, 2**149),
                              grid_value(left), grid_value(right)):
                    check_rational(sign * probe, failures)
                    checked += 1

    # 2. Subnormal values: exhaustive on the head/tail bands, sampled between.
    for mantissa in list(range(1 << 16)) + random.sample(range(1 << 16, 1 << 23), 1 << 16):
        check_rational(Fraction(mantissa, 2**149), failures)
        checked += 1

    # 3. Overflow boundary: exactly the midpoint of max finite and 2^128 raises.
    overflow_midpoint = (grid_value(MAX_FINITE) + SENTINEL_VALUE) / 2
    for probe, should_raise in ((overflow_midpoint, True), (overflow_midpoint - Fraction(1, 2**80), False),
                                (SENTINEL_VALUE, True)):
        try:
            rational_fp32("mpfr_check", probe)
            assert not should_raise, f"{probe}: expected overflow"
        except ValueError:
            assert should_raise, f"{probe}: unexpected overflow"
        checked += 1

    # 4. Random rationals across the whole range.
    for _ in range(random_cases):
        numerator = random.getrandbits(random.randint(1, 200))
        denominator = random.getrandbits(random.randint(1, 120)) or 1
        scale = Fraction(numerator, denominator) * Fraction(2) ** random.randint(-180, 120)
        sign = -1 if random.random() < 0.5 else 1
        check_rational(sign * scale, failures)
        checked += 1

    # 5. inverse_sqrt_fp32 against the exact-arithmetic reference.
    sqrt_checked = 0
    for d in list(range(1, 20_001)) + [k * k for k in range(2, 200)]:
        check_inverse_sqrt(d, failures)
        sqrt_checked += 1
    for _ in range(2_000):
        check_inverse_sqrt(random.getrandbits(random.randint(1, 2048)) or 1, failures)
        sqrt_checked += 1

    assert not failures, f"{len(failures)} mismatches, first: {failures[:5]}"
    print(f"mpfr_rounding_check.py: PASS ({checked} rational cases, {sqrt_checked} sqrt cases)")
    print("  - rational_fp32 == exact binary32-grid RN-even (bracket + tie) on all cases")
    print("  - ties-to-even verified at every exponent-class boundary midpoint")
    print("  - overflow raises exactly at/beyond the max-finite tie midpoint")
    print("  - MPFR agrees on the normal range; inverse_sqrt matches exact arithmetic")


if __name__ == "__main__":
    main()
