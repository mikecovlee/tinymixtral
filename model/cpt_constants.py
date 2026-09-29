"""Exact CPT constant arithmetic and correctly rounded IEEE-754 binary32 values.

Only initialization uses rational arithmetic. Runtime tensors remain FP32.
No eval, decimal approximation of radicals, or binary64 double rounding is used.
"""
from fractions import Fraction
from numbers import Real
import struct
from typing import Callable


def exact_scalar(name: str, value: object) -> Fraction:
    """Parse a rational or a decimal spelling as an exact source constant.

    Use strings such as "19/20" in JSON for unambiguous mathematical intent.
    Python floats are interpreted by their shortest decimal spelling; already
    rounded inputs cannot recover the user's original mathematical expression.
    """
    if isinstance(value, bool) or not isinstance(value, (str, Real)):
        raise ValueError(f"{name} must be a finite rational or decimal scalar")
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite rational or decimal scalar") from error


def _positive_value(bits: int) -> Fraction:
    # The sentinel above the largest finite value makes overflow rounding exact.
    if bits == 0x7F800000:
        return Fraction(2**128)
    exponent, mantissa = bits >> 23, bits & 0x7FFFFF
    if exponent == 0:
        return Fraction(mantissa, 2**149)
    significand = (1 << 23) + mantissa
    shift = exponent - 150
    return Fraction(significand * 2**shift) if shift >= 0 else Fraction(significand, 2**(-shift))


def _round_positive(name: str, compare: Callable[[Fraction], int]) -> float:
    """Round a nonnegative exact value using an exact comparison oracle.

    compare(q) gives the sign of (q - exact_value). Binary search brackets the
    value with consecutive binary32 numbers, then compares their exact midpoint.
    Bit parity implements ties-to-even, including subnormal/exponent boundaries.
    """
    low, high = 0, 0x7F800000
    while high - low > 1:
        middle = (low + high) // 2
        if compare(_positive_value(middle)) <= 0:
            low = middle
        else:
            high = middle
    midpoint = (_positive_value(low) + _positive_value(high)) / 2
    relation = compare(midpoint)
    bits = low if relation > 0 else high if relation < 0 else (low if low % 2 == 0 else high)
    if bits == 0x7F800000:
        raise ValueError(f"{name} must be finite in FP32")
    return struct.unpack('!f', struct.pack('!I', bits))[0]


def rational_fp32(name: str, value: Fraction) -> float:
    """One correctly rounded conversion from an exact fraction to binary32."""
    magnitude = abs(value)
    def compare(candidate: Fraction) -> int:
        return (candidate > magnitude) - (candidate < magnitude)
    result = _round_positive(name, compare)
    return -result if value < 0 else result


def inverse_sqrt_fp32(name: str, dimension: int) -> float:
    """Correctly round 1/sqrt(dimension) by exact squared comparisons."""
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    def compare(candidate: Fraction) -> int:
        square = candidate * candidate * dimension
        return (square > 1) - (square < 1)
    return _round_positive(name, compare)
