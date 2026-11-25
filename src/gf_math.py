"""GF(2^8) arithmetic, multiplicative inverse table, and affine S-box construction."""

from __future__ import annotations

import numpy as np

from config import (
    AES_CIRCULANT_FIRST_ROW,
    DEFAULT_AFFINE_CONSTANT,
    GF_SIZE,
    IRREDUCIBLE_POLY,
)

_BIT_INDICES = np.arange(8, dtype=np.uint8)
_BIT_WEIGHTS = (1 << _BIT_INDICES).astype(np.uint16)


def gf_add(a: int, b: int) -> int:
    """Add two elements in GF(2^8) using bitwise XOR."""
    return (a ^ b) & 0xFF


def gf_mul(a: int, b: int, poly: int = IRREDUCIBLE_POLY) -> int:
    """Multiply two elements in GF(2^8) modulo an irreducible polynomial."""
    a &= 0xFF
    b &= 0xFF
    reduction_mask = poly & 0xFF
    result = 0

    for _ in range(8):
        if b & 1:
            result ^= a
        high_bit = a & 0x80
        a = (a << 1) & 0xFF
        if high_bit:
            a ^= reduction_mask
        b >>= 1

    return result


def gf_pow(base: int, exp: int, poly: int = IRREDUCIBLE_POLY) -> int:
    """Compute base^exp in GF(2^8) via binary square-and-multiply."""
    result = 1
    current = base & 0xFF
    power = exp

    while power > 0:
        if power & 1:
            result = gf_mul(result, current, poly)
        current = gf_mul(current, current, poly)
        power >>= 1

    return result


def gf_inv(a: int, poly: int = IRREDUCIBLE_POLY) -> int:
    """Return the multiplicative inverse of `a` in GF(2^8), with inv(0) = 0."""
    a &= 0xFF
    if a == 0:
        return 0
    # By Fermat's Little Theorem in GF(2^8), a^(2^8 - 2) = a^254 = a^-1.
    return gf_pow(a, 254, poly)


def generate_inverse_table(poly: int = IRREDUCIBLE_POLY) -> np.ndarray:
    """Generate the 256-element GF(2^8) multiplicative inverse lookup table."""
    return np.array([gf_inv(x, poly) for x in range(GF_SIZE)], dtype=np.uint8)


def is_invertible_gf2(matrix: np.ndarray) -> bool:
    """Check whether an 8x8 binary matrix has full rank (8) over GF(2)."""
    mat = np.asarray(matrix, dtype=np.uint8)
    if mat.shape != (8, 8):
        return False

    work = (mat & 1).copy()
    row = 0
    for col in range(8):
        pivot = -1
        for r in range(row, 8):
            if work[r, col]:
                pivot = r
                break
        if pivot == -1:
            return False
        if pivot != row:
            work[[row, pivot]] = work[[pivot, row]]
        for r in range(8):
            if r != row and work[r, col]:
                work[r] ^= work[row]
        row += 1

    return True


def circulant_matrix(first_row_mask: int) -> np.ndarray:
    """Build an 8x8 circulant binary matrix from an 8-bit LSB-first row mask."""
    row0 = np.array([(first_row_mask >> j) & 1 for j in range(8)], dtype=np.uint8)
    return np.array([np.roll(row0, i) for i in range(8)], dtype=np.uint8)


def get_aes_affine_matrix() -> np.ndarray:
    """Return the standard 8x8 AES affine transformation matrix."""
    return circulant_matrix(AES_CIRCULANT_FIRST_ROW)


def affine_transform_byte(
    x: int,
    matrix: np.ndarray,
    constant: int = DEFAULT_AFFINE_CONSTANT,
) -> int:
    """Apply B(x) = (K * x + C) mod 2 to a single 8-bit integer."""
    mat = np.asarray(matrix, dtype=np.uint8) & 1
    if mat.shape != (8, 8):
        raise ValueError(f"Affine matrix must have shape (8, 8), got {mat.shape}")

    x_bits = np.array([(x >> j) & 1 for j in range(8)], dtype=np.uint8)
    c_bits = np.array([(constant >> j) & 1 for j in range(8)], dtype=np.uint8)
    y_bits = (mat @ x_bits + c_bits) & 1
    return int(np.sum(y_bits * _BIT_WEIGHTS))


def build_sbox(
    matrix: np.ndarray,
    constant: int = DEFAULT_AFFINE_CONSTANT,
    poly: int = IRREDUCIBLE_POLY,
    inv_table: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a 256-element 8-bit S-box via B(X) = (K * X^-1 + C) mod 2."""
    mat = np.asarray(matrix, dtype=np.uint8) & 1
    if mat.shape != (8, 8):
        raise ValueError(f"Affine matrix must have shape (8, 8), got {mat.shape}")

    if inv_table is None:
        inv_table = generate_inverse_table(poly)

    inv_bits = ((inv_table[:, None] >> _BIT_INDICES) & 1).astype(np.uint8)
    c_bits = ((constant >> _BIT_INDICES) & 1).astype(np.uint8)
    y_bits = (inv_bits @ mat.T + c_bits) & 1
    return (y_bits * _BIT_WEIGHTS).sum(axis=1).astype(np.uint8)


def build_aes_sbox() -> np.ndarray:
    """Construct the standard FIPS-197 AES S-box."""
    return build_sbox(get_aes_affine_matrix(), DEFAULT_AFFINE_CONSTANT, IRREDUCIBLE_POLY)


def build_inverse_sbox(sbox: np.ndarray) -> np.ndarray:
    """Derive the inverse S-box such that inv_sbox[sbox[x]] == x for all x."""
    arr = np.asarray(sbox)
    if arr.shape != (GF_SIZE,):
        raise ValueError(f"S-box must have length {GF_SIZE}, got {arr.shape}")
    if np.unique(arr).size != GF_SIZE or int(arr.min()) < 0 or int(arr.max()) >= GF_SIZE:
        raise ValueError("S-box must be a bijection over 0..255 to invert.")

    inv_sbox = np.empty(GF_SIZE, dtype=np.uint8)
    inv_sbox[arr.astype(np.int64)] = np.arange(GF_SIZE, dtype=np.uint8)
    return inv_sbox


def to_sbox_grid(sbox: np.ndarray) -> list[list[int]]:
    """Reshape a 256-element S-box into a 16x16 integer matrix."""
    arr = np.asarray(sbox, dtype=np.uint8).reshape(16, 16)
    return [[int(val) for val in row] for row in arr]


def to_sbox_hex_grid(sbox: np.ndarray) -> list[list[str]]:
    """Reshape a 256-element S-box into a 16x16 uppercase hex string matrix."""
    arr = np.asarray(sbox, dtype=np.uint8).reshape(16, 16)
    return [[f"{int(val):02X}" for val in row] for row in arr]


def count_fixed_points(sbox: np.ndarray) -> tuple[int, int]:
    """Count fixed points (S(x) == x) and opposite fixed points (S(x) == x ^ 0xFF)."""
    arr = np.asarray(sbox, dtype=np.uint8)
    xs = np.arange(GF_SIZE, dtype=np.uint8)
    fixed = int(np.sum(arr == xs))
    opposite_fixed = int(np.sum(arr == (xs ^ 0xFF)))
    return fixed, opposite_fixed


def find_zero_fixed_point_constant(
    matrix: np.ndarray,
    poly: int = IRREDUCIBLE_POLY,
    preferred: int = DEFAULT_AFFINE_CONSTANT,
    inv_table: np.ndarray | None = None,
) -> int:
    """Find an 8-bit constant C that avoids fixed and opposite-fixed points."""
    if inv_table is None:
        inv_table = generate_inverse_table(poly)

    base_sbox = build_sbox(matrix, constant=0, poly=poly, inv_table=inv_table)
    xs = np.arange(GF_SIZE, dtype=np.uint8)

    def _total_fixed(c: int) -> int:
        shifted = base_sbox ^ np.uint8(c)
        return int(np.sum(shifted == xs) + np.sum(shifted == (xs ^ 0xFF)))

    if _total_fixed(preferred) == 0:
        return preferred

    best_c = preferred
    best_count = _total_fixed(preferred)
    for c in range(GF_SIZE):
        cnt = _total_fixed(c)
        if cnt == 0:
            return c
        if cnt < best_count:
            best_count = cnt
            best_c = c

    return best_c
