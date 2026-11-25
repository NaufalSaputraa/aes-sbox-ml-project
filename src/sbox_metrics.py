"""Hard-constraint validators and cryptographic strength evaluators for 8-bit S-boxes."""

from __future__ import annotations

from typing import Any

import numpy as np

from config import GF_SIZE
from src.gf_math import count_fixed_points

_XS = np.arange(GF_SIZE, dtype=np.int32)
_BIT_INDICES = np.arange(8, dtype=np.uint8)
_PAIR_INDICES: list[tuple[int, int]] = [
    (i, j) for i in range(8) for j in range(i + 1, 8)
]
_POPCOUNT_PARITY = np.array(
    [bin(val).count("1") & 1 for val in range(GF_SIZE)],
    dtype=np.int32,
)
_POPCOUNT = np.array([bin(val).count("1") for val in range(GF_SIZE)], dtype=np.int32)
_B_MASKS = np.arange(1, GF_SIZE, dtype=np.uint8)


def _as_sbox_array(sbox: np.ndarray | list[int]) -> np.ndarray:
    """Validate shape and convert input S-box to a 1D uint8 array of length 256."""
    arr = np.asarray(sbox)
    if arr.shape != (GF_SIZE,):
        raise ValueError(f"Expected 1D S-box of length {GF_SIZE}, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        raise ValueError("S-box values must be integers in 0..255.")
    if int(arr.min()) < 0 or int(arr.max()) >= GF_SIZE:
        raise ValueError("S-box values must be in range 0..255.")
    return arr.astype(np.uint8)


def is_bijective(sbox: np.ndarray | list[int]) -> bool:
    """Return True if every integer 0..255 appears exactly once in the S-box."""
    arr = np.asarray(sbox)
    if arr.shape != (GF_SIZE,) or np.issubdtype(arr.dtype, np.floating):
        return False
    if int(arr.min()) < 0 or int(arr.max()) >= GF_SIZE:
        return False
    return int(np.unique(arr).size) == GF_SIZE


def check_balance(sbox: np.ndarray | list[int]) -> tuple[bool, list[int]]:
    """Verify 128:128 bit balance for every output bit position 0..7."""
    arr = _as_sbox_array(sbox)
    bits = (arr[:, None] >> _BIT_INDICES) & 1
    ones_counts = [int(c) for c in bits.sum(axis=0)]
    balanced = all(c == 128 for c in ones_counts)
    return balanced, ones_counts


def validate_sbox(sbox: np.ndarray | list[int]) -> dict[str, Any]:
    """Validate hard cryptographic constraints (bijective and 128:128 bit balance)."""
    bijective = is_bijective(sbox)
    balanced, ones_counts = check_balance(sbox)
    return {
        "bijective": bijective,
        "balanced": balanced,
        "bit_ones_count": ones_counts,
        "valid": bool(bijective and balanced),
    }


def _fwht(values: np.ndarray) -> np.ndarray:
    """Compute Fast Walsh-Hadamard Transform along the last axis (size 256)."""
    h = np.asarray(values, dtype=np.int32).copy()
    orig_shape = h.shape
    step = 1
    while step < GF_SIZE:
        h = h.reshape(*orig_shape[:-1], -1, 2 * step)
        u = h[..., :step].copy()
        v = h[..., step:].copy()
        h[..., :step] = u + v
        h[..., step:] = u - v
        h = h.reshape(orig_shape)
        step *= 2
    return h


def compute_nonlinearity(sbox: np.ndarray | list[int]) -> dict[str, Any]:
    """Compute Nonlinearity (NL) across all 8 coordinate Boolean functions."""
    arr = _as_sbox_array(sbox)
    bits = ((arr[:, None] >> _BIT_INDICES) & 1).astype(np.int32)
    polar = 1 - 2 * bits.T  # shape: (8, 256)
    wht = _fwht(polar)
    nl_per_bit = [int(v) for v in (128 - np.max(np.abs(wht), axis=1) // 2)]
    return {
        "nl": int(min(nl_per_bit)),
        "nl_avg": float(sum(nl_per_bit) / len(nl_per_bit)),
        "nl_per_bit": nl_per_bit,
    }


def compute_sac(sbox: np.ndarray | list[int]) -> dict[str, Any]:
    """Compute Strict Avalanche Criterion (SAC) matrix and summary statistics."""
    arr = _as_sbox_array(sbox)
    bits = ((arr[:, None] >> _BIT_INDICES) & 1).astype(np.int32)  # (256, 8)
    sac_matrix = np.zeros((8, 8), dtype=np.float64)

    for in_bit in range(8):
        flipped_bits = bits[_XS ^ (1 << in_bit)]
        sac_matrix[in_bit] = np.mean(bits ^ flipped_bits, axis=0)

    return {
        "sac": float(np.mean(sac_matrix)),
        "sac_min": float(np.min(sac_matrix)),
        "sac_max": float(np.max(sac_matrix)),
        "sac_std": float(np.std(sac_matrix)),
        "sac_max_dev": float(np.max(np.abs(sac_matrix - 0.5))),
        "sac_matrix": [[float(val) for val in row] for row in sac_matrix],
    }


def compute_bic_nl(sbox: np.ndarray | list[int]) -> dict[str, Any]:
    """Compute Bit Independence Criterion - Nonlinearity over all 28 bit pairs."""
    arr = _as_sbox_array(sbox)
    bits = ((arr[:, None] >> _BIT_INDICES) & 1).astype(np.int32)
    pairs = np.stack([bits[:, i] ^ bits[:, j] for i, j in _PAIR_INDICES], axis=0)
    wht = _fwht(1 - 2 * pairs)
    nl_vals = 128 - np.max(np.abs(wht), axis=1) // 2
    return {
        "bic_nl": int(np.min(nl_vals)),
        "bic_nl_avg": float(np.mean(nl_vals)),
        "bic_nl_max": int(np.max(nl_vals)),
    }


def compute_bic_sac(sbox: np.ndarray | list[int]) -> dict[str, Any]:
    """Compute Bit Independence Criterion - SAC over all 28 bit pairs."""
    arr = _as_sbox_array(sbox)
    bits = ((arr[:, None] >> _BIT_INDICES) & 1).astype(np.int32)
    pairs = np.stack([bits[:, i] ^ bits[:, j] for i, j in _PAIR_INDICES], axis=0)  # (28, 256)

    pair_sac_sum = np.zeros(len(_PAIR_INDICES), dtype=np.float64)
    for in_bit in range(8):
        flipped_pairs = pairs[:, _XS ^ (1 << in_bit)]
        pair_sac_sum += np.mean(pairs ^ flipped_pairs, axis=1)

    pair_sac = pair_sac_sum / 8.0
    return {
        "bic_sac": float(np.mean(pair_sac)),
        "bic_sac_min": float(np.min(pair_sac)),
        "bic_sac_max": float(np.max(pair_sac)),
        "bic_sac_std": float(np.std(pair_sac)),
    }


def compute_lap(sbox: np.ndarray | list[int]) -> float:
    """Compute Linear Approximation Probability (LAP = max bias / 256)."""
    arr = _as_sbox_array(sbox)
    masked = np.bitwise_and(arr[None, :], _B_MASKS[:, None])  # (255, 256)
    dot_bs = _POPCOUNT_PARITY[masked]
    lat = _fwht(1 - 2 * dot_bs)
    max_wht = int(np.max(np.abs(lat[:, 1:])))
    # WHT(a, b) = 2 * (#{x: <a,x> == <b,S(x)>} - 128)
    return float(max_wht / 512.0)


def compute_differential_uniformity(sbox: np.ndarray | list[int]) -> int:
    """Compute Differential Uniformity (maximum non-zero entry of the DDT)."""
    arr = _as_sbox_array(sbox)
    max_ddt = 0
    for dx in range(1, GF_SIZE):
        dy = arr ^ arr[_XS ^ dx]
        cnt = int(np.bincount(dy, minlength=GF_SIZE).max())
        if cnt > max_ddt:
            max_ddt = cnt
    return max_ddt


def compute_dap(sbox: np.ndarray | list[int]) -> float:
    """Compute Differential Approximation Probability (DAP = DU / 256)."""
    return float(compute_differential_uniformity(sbox) / float(GF_SIZE))


def compute_algebraic_degree(sbox: np.ndarray | list[int]) -> dict[str, Any]:
    """Compute algebraic degree of all 8 coordinate Boolean functions via Fast Mobius Transform."""
    arr = _as_sbox_array(sbox)
    bits = ((arr[:, None] >> _BIT_INDICES) & 1).astype(np.uint8).T  # (8, 256)
    anf = bits.copy()
    for i in range(8):
        step = 1 << i
        anf = anf.reshape(8, -1, 2 * step)
        anf[:, :, step:] ^= anf[:, :, :step]
        anf = anf.reshape(8, GF_SIZE)

    degrees: list[int] = []
    for b in range(8):
        active = np.flatnonzero(anf[b])
        degrees.append(int(_POPCOUNT[active].max()) if active.size > 0 else 0)

    return {
        "algebraic_degree_min": int(min(degrees)),
        "algebraic_degree_max": int(max(degrees)),
        "algebraic_degrees": degrees,
    }


def compute_cycle_structure(sbox: np.ndarray | list[int]) -> dict[str, Any]:
    """Compute disjoint cycle lengths of the permutation x -> S(x)."""
    arr = _as_sbox_array(sbox)
    if not is_bijective(arr):
        return {"cycles": [], "num_cycles": 0, "min_cycle": 0, "max_cycle": 0}

    visited = [False] * GF_SIZE
    cycles: list[int] = []
    for start in range(GF_SIZE):
        if not visited[start]:
            length = 0
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cur = int(arr[cur])
                length += 1
            cycles.append(length)

    cycles.sort(reverse=True)
    return {
        "cycles": cycles,
        "num_cycles": len(cycles),
        "min_cycle": min(cycles),
        "max_cycle": max(cycles),
    }


def compute_composite_scores(
    nl: int,
    sac: float,
    bic_nl: int,
    bic_sac: float,
) -> tuple[float, float]:
    """Compute composite deviation score S and Strength Value (SV)."""
    s_score = (abs(sac - 0.5) + abs(bic_sac - 0.5)) / 2.0
    sv_score = (120.0 - nl) + abs(0.5 - sac) + (120.0 - bic_nl) + abs(0.5 - bic_sac)
    return float(s_score), float(sv_score)


def evaluate_fast_sac_metrics(sbox: np.ndarray) -> tuple[float, float, float, float]:
    """Fast SAC, BIC-SAC, S-score, and SAC-std calculation for optimization loops."""
    bits = ((sbox[:, None] >> _BIT_INDICES) & 1).astype(np.int32)
    pairs = np.stack([bits[:, i] ^ bits[:, j] for i, j in _PAIR_INDICES], axis=0)

    sac_matrix = np.empty((8, 8), dtype=np.float64)
    bic_sum = 0
    for in_bit in range(8):
        idx = _XS ^ (1 << in_bit)
        sac_matrix[in_bit] = np.mean(bits ^ bits[idx], axis=0)
        bic_sum += int(np.sum(pairs ^ pairs[:, idx]))

    sac = float(np.mean(sac_matrix))
    sac_std = float(np.std(sac_matrix))
    bic_sac = float(bic_sum / (8 * len(_PAIR_INDICES) * GF_SIZE))
    s_score = (abs(sac - 0.5) + abs(bic_sac - 0.5)) / 2.0
    return sac, bic_sac, s_score, sac_std


def evaluate_sbox(
    sbox: np.ndarray | list[int],
    include_details: bool = False,
) -> dict[str, Any]:
    """Run full constraint validation and cryptographic metric evaluation."""
    arr = _as_sbox_array(sbox)
    validity = validate_sbox(arr)
    nl_info = compute_nonlinearity(arr)
    sac_info = compute_sac(arr)
    bic_nl_info = compute_bic_nl(arr)
    bic_sac_info = compute_bic_sac(arr)
    lap = compute_lap(arr)
    du = compute_differential_uniformity(arr)
    dap = float(du / float(GF_SIZE))
    deg_info = compute_algebraic_degree(arr)
    cycle_info = compute_cycle_structure(arr)
    s_score, sv_score = compute_composite_scores(
        nl_info["nl"],
        sac_info["sac"],
        bic_nl_info["bic_nl"],
        bic_sac_info["bic_sac"],
    )
    fixed_points, opposite_fixed_points = count_fixed_points(arr)

    result: dict[str, Any] = {
        "valid": validity["valid"],
        "bijective": validity["bijective"],
        "balanced": validity["balanced"],
        "bit_ones_count": validity["bit_ones_count"],
        "nl": nl_info["nl"],
        "nl_avg": nl_info["nl_avg"],
        "sac": sac_info["sac"],
        "sac_std": sac_info["sac_std"],
        "sac_min": sac_info["sac_min"],
        "sac_max": sac_info["sac_max"],
        "sac_max_dev": sac_info["sac_max_dev"],
        "bic_nl": bic_nl_info["bic_nl"],
        "bic_nl_avg": bic_nl_info["bic_nl_avg"],
        "bic_sac": bic_sac_info["bic_sac"],
        "bic_sac_min": bic_sac_info["bic_sac_min"],
        "bic_sac_max": bic_sac_info["bic_sac_max"],
        "bic_sac_std": bic_sac_info["bic_sac_std"],
        "lap": lap,
        "du": du,
        "dap": dap,
        "algebraic_degree_min": deg_info["algebraic_degree_min"],
        "algebraic_degree_max": deg_info["algebraic_degree_max"],
        "cycles": cycle_info["cycles"],
        "num_cycles": cycle_info["num_cycles"],
        "min_cycle": cycle_info["min_cycle"],
        "max_cycle": cycle_info["max_cycle"],
        "s_score": s_score,
        "sv_score": sv_score,
        "fixed_points": fixed_points,
        "opposite_fixed_points": opposite_fixed_points,
    }
    if include_details:
        result["nl_per_bit"] = nl_info["nl_per_bit"]
        result["sac_matrix"] = sac_info["sac_matrix"]
        result["algebraic_degrees"] = deg_info["algebraic_degrees"]
    return result
