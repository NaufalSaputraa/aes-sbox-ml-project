"""CLI runner for Fase 1 validation and Fase 2 Hybrid GA-SA S-box optimization."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_AFFINE_CONSTANT,
    GA_GENERATIONS,
    GA_POPULATION_SIZE,
    IRREDUCIBLE_POLY,
    RANDOM_SEED,
    SA_STEPS,
    SBOX44_CIRCULANT_FIRST_ROW,
    SBOX_RESULTS_PATH,
    TARGET_METRICS,
)
from src.gf_math import (
    build_aes_sbox,
    build_inverse_sbox,
    build_sbox,
    circulant_matrix,
    get_aes_affine_matrix,
    to_sbox_grid,
    to_sbox_hex_grid,
)
from src.sbox_metrics import evaluate_sbox
from src.sbox_optimizer import HybridGASASboxOptimizer


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to native Python JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(item) for item in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _serialize_sbox_entry(
    matrix: np.ndarray,
    constant: int,
    sbox: np.ndarray,
    inv_sbox: np.ndarray,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured JSON dictionary for a single S-box configuration."""
    return {
        "affine_matrix": _to_json_safe(matrix),
        "affine_constant": int(constant),
        "affine_constant_hex": f"0x{constant:02X}",
        "sbox_flat": [int(v) for v in sbox],
        "sbox_16x16": to_sbox_grid(sbox),
        "sbox_hex_16x16": to_sbox_hex_grid(sbox),
        "inverse_sbox_flat": [int(v) for v in inv_sbox],
        "inverse_sbox_16x16": to_sbox_grid(inv_sbox),
        "metrics": _to_json_safe(metrics),
    }


def _verify_aes_baseline(metrics: dict[str, Any]) -> None:
    """Assert that computed standard AES metrics match cryptographic reference values."""
    assert metrics["valid"], "Standard AES S-box failed bijective/balance check."
    assert metrics["nl"] == 112, f"Expected AES NL=112, got {metrics['nl']}"
    assert abs(metrics["sac"] - 0.5048828125) < 1e-9, f"Unexpected AES SAC={metrics['sac']}"
    assert metrics["bic_nl"] == 112, f"Expected AES BIC-NL=112, got {metrics['bic_nl']}"
    assert abs(metrics["bic_sac"] - 0.5046037946428571) < 1e-7, (
        f"Unexpected AES BIC-SAC={metrics['bic_sac']}"
    )
    assert abs(metrics["lap"] - 0.0625) < 1e-9, f"Unexpected AES LAP={metrics['lap']}"
    assert abs(metrics["dap"] - 0.015625) < 1e-9, f"Unexpected AES DAP={metrics['dap']}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reproducible S-box optimization."""
    parser = argparse.ArgumentParser(
        description="Run AES S-box baseline validation and Hybrid GA-SA optimization."
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Locked random seed")
    parser.add_argument(
        "--population",
        type=int,
        default=GA_POPULATION_SIZE,
        help="GA population size",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=GA_GENERATIONS,
        help="Number of GA generations",
    )
    parser.add_argument(
        "--sa-steps",
        type=int,
        default=SA_STEPS,
        help="Number of Simulated Annealing refinement steps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SBOX_RESULTS_PATH,
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Fase 1 baseline verification and Fase 2 S-box optimization."""
    args = parse_args()

    # 1. Standard AES S-box baseline
    aes_matrix = get_aes_affine_matrix()
    aes_sbox = build_aes_sbox()
    aes_inv_sbox = build_inverse_sbox(aes_sbox)
    aes_metrics = evaluate_sbox(aes_sbox, include_details=True)
    _verify_aes_baseline(aes_metrics)

    # 2. Benchmark S-box44 (Alamsyah et al., 2025)
    sbox44_matrix = circulant_matrix(SBOX44_CIRCULANT_FIRST_ROW)
    sbox44 = build_sbox(sbox44_matrix, constant=DEFAULT_AFFINE_CONSTANT, poly=IRREDUCIBLE_POLY)
    sbox44_inv = build_inverse_sbox(sbox44)
    sbox44_metrics = evaluate_sbox(sbox44, include_details=True)

    # 3. Hybrid GA-SA optimization
    optimizer = HybridGASASboxOptimizer(
        poly=IRREDUCIBLE_POLY,
        seed=args.seed,
        population_size=args.population,
        generations=args.generations,
        sa_steps=args.sa_steps,
    )
    opt_result = optimizer.optimize()
    custom_metrics = opt_result.metrics

    # Verify lossless inverse round-trip
    round_trip_ok = bool(
        np.array_equal(opt_result.inverse_sbox[opt_result.sbox], np.arange(256, dtype=np.uint8))
    )
    assert round_trip_ok, "Custom S-box inverse round-trip verification failed."

    # 4. Print comparison summary
    comparison_rows = [
        {
            "Metric": "NL",
            "Target": ">= 112",
            "AES Standard": f"{aes_metrics['nl']}",
            "S-box44 (Ref)": f"{sbox44_metrics['nl']}",
            "Custom GA-SA": f"{custom_metrics['nl']}",
        },
        {
            "Metric": "SAC",
            "Target": "~0.50000",
            "AES Standard": f"{aes_metrics['sac']:.5f}",
            "S-box44 (Ref)": f"{sbox44_metrics['sac']:.5f}",
            "Custom GA-SA": f"{custom_metrics['sac']:.5f}",
        },
        {
            "Metric": "BIC-NL",
            "Target": ">= 112",
            "AES Standard": f"{aes_metrics['bic_nl']}",
            "S-box44 (Ref)": f"{sbox44_metrics['bic_nl']}",
            "Custom GA-SA": f"{custom_metrics['bic_nl']}",
        },
        {
            "Metric": "BIC-SAC",
            "Target": "~0.50000",
            "AES Standard": f"{aes_metrics['bic_sac']:.5f}",
            "S-box44 (Ref)": f"{sbox44_metrics['bic_sac']:.5f}",
            "Custom GA-SA": f"{custom_metrics['bic_sac']:.5f}",
        },
        {
            "Metric": "LAP",
            "Target": "<= 0.0625",
            "AES Standard": f"{aes_metrics['lap']:.4f}",
            "S-box44 (Ref)": f"{sbox44_metrics['lap']:.4f}",
            "Custom GA-SA": f"{custom_metrics['lap']:.4f}",
        },
        {
            "Metric": "DAP",
            "Target": "<= 0.015625",
            "AES Standard": f"{aes_metrics['dap']:.6f}",
            "S-box44 (Ref)": f"{sbox44_metrics['dap']:.6f}",
            "Custom GA-SA": f"{custom_metrics['dap']:.6f}",
        },
        {
            "Metric": "S Score",
            "Target": "<= 0.0040",
            "AES Standard": f"{aes_metrics['s_score']:.6f}",
            "S-box44 (Ref)": f"{sbox44_metrics['s_score']:.6f}",
            "Custom GA-SA": f"{custom_metrics['s_score']:.6f}",
        },
        {
            "Metric": "SV Score",
            "Target": "~16.0000",
            "AES Standard": f"{aes_metrics['sv_score']:.5f}",
            "S-box44 (Ref)": f"{sbox44_metrics['sv_score']:.5f}",
            "Custom GA-SA": f"{custom_metrics['sv_score']:.5f}",
        },
    ]
    df = pd.DataFrame(comparison_rows)

    print("=" * 78)
    print("AES S-Box Construction & Optimization Summary (Fase 1 & Fase 2)")
    print("=" * 78)
    print(f"Algorithm   : {opt_result.algorithm}")
    print(f"Random Seed : {opt_result.seed}")
    print(f"Runtime     : {opt_result.runtime_seconds:.3f} s")
    print(
        f"Constraints : bijective={custom_metrics['bijective']}, "
        f"balanced={custom_metrics['balanced']} (128:128 per bit)"
    )
    print(
        f"Fixed Points: {custom_metrics['fixed_points']} | "
        f"Opposite Fixed Points: {custom_metrics['opposite_fixed_points']} | "
        f"Affine C: 0x{opt_result.affine_constant:02X}"
    )
    print("-" * 78)
    print(df.to_string(index=False))
    print("=" * 78)

    # 5. Persist results to JSON
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "irreducible_poly": f"0x{IRREDUCIBLE_POLY:03X}",
            "seed": int(opt_result.seed),
            "algorithm": opt_result.algorithm,
            "runtime_seconds": float(opt_result.runtime_seconds),
            "hyperparameters": _to_json_safe(opt_result.hyperparameters),
            "target_metrics": _to_json_safe(TARGET_METRICS),
            "inverse_round_trip_verified": round_trip_ok,
        },
        "aes_standard": _serialize_sbox_entry(
            aes_matrix, DEFAULT_AFFINE_CONSTANT, aes_sbox, aes_inv_sbox, aes_metrics
        ),
        "benchmark_sbox44": _serialize_sbox_entry(
            sbox44_matrix, DEFAULT_AFFINE_CONSTANT, sbox44, sbox44_inv, sbox44_metrics
        ),
        "custom_sbox": _serialize_sbox_entry(
            opt_result.affine_matrix,
            opt_result.affine_constant,
            opt_result.sbox,
            opt_result.inverse_sbox,
            custom_metrics,
        ),
        "convergence_history": _to_json_safe(opt_result.history),
        "comparison_summary": comparison_rows,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
