"""CLI runner for Dual-Track S-box optimization (Track A Non-Algebraic + Track B Hybrid GA-SA)."""

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
from src.sbox_optimizer import (
    HybridGASASboxOptimizer,
    NonAlgebraicWHSOptimizer,
    run_random_search_benchmark,
)


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
    matrix: np.ndarray | None,
    constant: int | None,
    sbox: np.ndarray,
    inv_sbox: np.ndarray,
    metrics: dict[str, Any],
    construction_type: str = "algebraic_gf28",
) -> dict[str, Any]:
    """Build a structured JSON dictionary for a single S-box configuration."""
    return {
        "construction_type": construction_type,
        "affine_matrix": _to_json_safe(matrix) if matrix is not None else None,
        "affine_constant": int(constant) if constant is not None else None,
        "affine_constant_hex": f"0x{constant:02X}" if constant is not None else "N/A (Non-Aljabar)",
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
    assert metrics["du"] == 4, f"Expected AES DU=4, got {metrics['du']}"
    assert metrics["algebraic_degree_min"] == 7, "Expected AES algebraic degree=7"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reproducible Dual-Track S-box optimization."""
    parser = argparse.ArgumentParser(
        description="Run Dual-Track S-box optimization (Non-Algebraic WHS-SA & Hybrid GA-SA)."
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Locked random seed")
    parser.add_argument("--population", type=int, default=GA_POPULATION_SIZE, help="GA population size")
    parser.add_argument("--generations", type=int, default=GA_GENERATIONS, help="GA generations")
    parser.add_argument("--sa-steps", type=int, default=SA_STEPS, help="GA-SA local SA steps")
    parser.add_argument(
        "--non-alg-steps",
        type=int,
        default=12000,
        help="Non-Algebraic WHS-SA optimization steps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SBOX_RESULTS_PATH,
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Dual-Track optimization, Random Search benchmark, and save JSON."""
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

    # 3. Track B: Hybrid GA-SA optimization over GL(8, 2)
    optimizer = HybridGASASboxOptimizer(
        poly=IRREDUCIBLE_POLY,
        seed=args.seed,
        population_size=args.population,
        generations=args.generations,
        sa_steps=args.sa_steps,
    )
    opt_result = optimizer.optimize()
    custom_metrics = opt_result.metrics

    # 4. Track A: Pure Non-Algebraic WHS Simulated Annealing (zero GF(2^8), zero affine matrix)
    non_alg_opt = NonAlgebraicWHSOptimizer(seed=args.seed, steps=args.non_alg_steps)
    non_alg_res = non_alg_opt.optimize()
    rand_perm_inv = build_inverse_sbox(non_alg_res.initial_sbox)
    rand_perm_m = non_alg_res.initial_metrics
    non_alg_m = non_alg_res.metrics

    # 5. Fair-budget Random Search Benchmark in GL(8, 2)
    fair_budget = args.population * args.generations + args.sa_steps
    rand_bench = run_random_search_benchmark(
        poly=IRREDUCIBLE_POLY,
        seed=args.seed,
        budget=fair_budget,
    )

    round_trip_ok = bool(
        np.array_equal(opt_result.inverse_sbox[opt_result.sbox], np.arange(256, dtype=np.uint8))
        and np.array_equal(non_alg_res.inverse_sbox[non_alg_res.sbox], np.arange(256, dtype=np.uint8))
    )
    assert round_trip_ok, "Inverse S-box round-trip verification failed."

    # 6. Print 5-way comparison summary
    comparison_rows = [
        {
            "Metric": "NL (min)",
            "Target": ">= 112 (Alg.)",
            "AES Standard": f"{aes_metrics['nl']}",
            "S-box44 (Ref)": f"{sbox44_metrics['nl']}",
            "Custom GA-SA": f"{custom_metrics['nl']}",
            "Random Perm": f"{rand_perm_m['nl']}",
            "Non-Alg WHS-SA": f"{non_alg_m['nl']}",
        },
        {
            "Metric": "SAC (mean)",
            "Target": "~0.50000",
            "AES Standard": f"{aes_metrics['sac']:.5f}",
            "S-box44 (Ref)": f"{sbox44_metrics['sac']:.5f}",
            "Custom GA-SA": f"{custom_metrics['sac']:.5f}",
            "Random Perm": f"{rand_perm_m['sac']:.5f}",
            "Non-Alg WHS-SA": f"{non_alg_m['sac']:.5f}",
        },
        {
            "Metric": "BIC-NL",
            "Target": ">= 112 (Alg.)",
            "AES Standard": f"{aes_metrics['bic_nl']}",
            "S-box44 (Ref)": f"{sbox44_metrics['bic_nl']}",
            "Custom GA-SA": f"{custom_metrics['bic_nl']}",
            "Random Perm": f"{rand_perm_m['bic_nl']}",
            "Non-Alg WHS-SA": f"{non_alg_m['bic_nl']}",
        },
        {
            "Metric": "BIC-SAC",
            "Target": "~0.50000",
            "AES Standard": f"{aes_metrics['bic_sac']:.5f}",
            "S-box44 (Ref)": f"{sbox44_metrics['bic_sac']:.5f}",
            "Custom GA-SA": f"{custom_metrics['bic_sac']:.5f}",
            "Random Perm": f"{rand_perm_m['bic_sac']:.5f}",
            "Non-Alg WHS-SA": f"{non_alg_m['bic_sac']:.5f}",
        },
        {
            "Metric": "LAP",
            "Target": "<= 0.0625",
            "AES Standard": f"{aes_metrics['lap']:.5f}",
            "S-box44 (Ref)": f"{sbox44_metrics['lap']:.5f}",
            "Custom GA-SA": f"{custom_metrics['lap']:.5f}",
            "Random Perm": f"{rand_perm_m['lap']:.5f}",
            "Non-Alg WHS-SA": f"{non_alg_m['lap']:.5f}",
        },
        {
            "Metric": "DAP (DU)",
            "Target": "<= 0.0156 (4)",
            "AES Standard": f"{aes_metrics['dap']:.5f} ({aes_metrics['du']})",
            "S-box44 (Ref)": f"{sbox44_metrics['dap']:.5f} ({sbox44_metrics['du']})",
            "Custom GA-SA": f"{custom_metrics['dap']:.5f} ({custom_metrics['du']})",
            "Random Perm": f"{rand_perm_m['dap']:.5f} ({rand_perm_m['du']})",
            "Non-Alg WHS-SA": f"{non_alg_m['dap']:.5f} ({non_alg_m['du']})",
        },
        {
            "Metric": "S Score",
            "Target": "<= 0.0040",
            "AES Standard": f"{aes_metrics['s_score']:.6f}",
            "S-box44 (Ref)": f"{sbox44_metrics['s_score']:.6f}",
            "Custom GA-SA": f"{custom_metrics['s_score']:.6f}",
            "Random Perm": f"{rand_perm_m['s_score']:.6f}",
            "Non-Alg WHS-SA": f"{non_alg_m['s_score']:.6f}",
        },
        {
            "Metric": "SV Score",
            "Target": "~16.0000",
            "AES Standard": f"{aes_metrics['sv_score']:.5f}",
            "S-box44 (Ref)": f"{sbox44_metrics['sv_score']:.5f}",
            "Custom GA-SA": f"{custom_metrics['sv_score']:.5f}",
            "Random Perm": f"{rand_perm_m['sv_score']:.5f}",
            "Non-Alg WHS-SA": f"{non_alg_m['sv_score']:.5f}",
        },
        {
            "Metric": "Alg. Degree (min)",
            "Target": "7",
            "AES Standard": f"{aes_metrics['algebraic_degree_min']}",
            "S-box44 (Ref)": f"{sbox44_metrics['algebraic_degree_min']}",
            "Custom GA-SA": f"{custom_metrics['algebraic_degree_min']}",
            "Random Perm": f"{rand_perm_m['algebraic_degree_min']}",
            "Non-Alg WHS-SA": f"{non_alg_m['algebraic_degree_min']}",
        },
        {
            "Metric": "Cycles (Count)",
            "Target": "Info",
            "AES Standard": f"{aes_metrics['num_cycles']}",
            "S-box44 (Ref)": f"{sbox44_metrics['num_cycles']}",
            "Custom GA-SA": f"{custom_metrics['num_cycles']}",
            "Random Perm": f"{rand_perm_m['num_cycles']}",
            "Non-Alg WHS-SA": f"{non_alg_m['num_cycles']}",
        },
        {
            "Metric": "Fixed / Opp. Fixed",
            "Target": "0 / 0",
            "AES Standard": f"{aes_metrics['fixed_points']} / {aes_metrics['opposite_fixed_points']}",
            "S-box44 (Ref)": f"{sbox44_metrics['fixed_points']} / {sbox44_metrics['opposite_fixed_points']}",
            "Custom GA-SA": f"{custom_metrics['fixed_points']} / {custom_metrics['opposite_fixed_points']}",
            "Random Perm": f"{rand_perm_m['fixed_points']} / {rand_perm_m['opposite_fixed_points']}",
            "Non-Alg WHS-SA": f"{non_alg_m['fixed_points']} / {non_alg_m['opposite_fixed_points']}",
        },
    ]
    df = pd.DataFrame(comparison_rows)

    print("=" * 98)
    print("Dual-Track AES S-Box Construction & Optimization Summary")
    print("=" * 98)
    print(f"Track B (Algebraic)     : {opt_result.algorithm} ({opt_result.runtime_seconds:.3f} s)")
    print(f"Track A (Non-Algebraic) : {non_alg_res.algorithm} ({non_alg_res.runtime_seconds:.3f} s)")
    print(
        f"Fair Budget Benchmark   : Hybrid GA-SA S={custom_metrics['s_score']:.6f} vs "
        f"Random GL(8,2) ({rand_bench['budget']} evals) Best S={rand_bench['best_s_score']:.6f} "
        f"(Mean S={rand_bench['mean_s_score']:.6f})"
    )
    print("-" * 98)
    print(df.to_string(index=False))
    print("=" * 98)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "irreducible_poly": f"0x{IRREDUCIBLE_POLY:03X}",
            "seed": int(opt_result.seed),
            "algorithm": opt_result.algorithm,
            "runtime_seconds": float(opt_result.runtime_seconds),
            "non_algebraic_algorithm": non_alg_res.algorithm,
            "non_algebraic_runtime_seconds": float(non_alg_res.runtime_seconds),
            "hyperparameters": _to_json_safe(opt_result.hyperparameters),
            "non_algebraic_hyperparameters": _to_json_safe(non_alg_res.hyperparameters),
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
        "random_permutation": _serialize_sbox_entry(
            None,
            None,
            non_alg_res.initial_sbox,
            rand_perm_inv,
            rand_perm_m,
            construction_type="random_permutation",
        ),
        "non_algebraic_sbox": _serialize_sbox_entry(
            None,
            None,
            non_alg_res.sbox,
            non_alg_res.inverse_sbox,
            non_alg_m,
            construction_type="non_algebraic_whs_sa",
        ),
        "convergence_history": _to_json_safe(opt_result.history),
        "non_algebraic_convergence_history": _to_json_safe(non_alg_res.history),
        "random_search_benchmark": _to_json_safe(rand_bench),
        "comparison_summary": comparison_rows,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
