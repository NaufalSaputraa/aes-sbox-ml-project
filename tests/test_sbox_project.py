"""Unit tests for GF(2^8) math, S-box metrics evaluator, and Hybrid GA-SA optimizer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_AFFINE_CONSTANT, IRREDUCIBLE_POLY, SBOX44_CIRCULANT_FIRST_ROW
from src.gf_math import (
    build_aes_sbox,
    build_inverse_sbox,
    build_sbox,
    circulant_matrix,
    generate_inverse_table,
    gf_add,
    gf_inv,
    gf_mul,
)
from src.sbox_metrics import evaluate_sbox, validate_sbox
from src.sbox_optimizer import (
    HybridGASASboxOptimizer,
    NonAlgebraicWHSOptimizer,
    run_random_search_benchmark,
)


class TestGFMath(unittest.TestCase):
    """Tests for Galois Field GF(2^8) arithmetic and S-box construction."""

    def test_gf_add_and_mul_fips197(self) -> None:
        self.assertEqual(gf_add(0x57, 0x83), 0xD4)
        self.assertEqual(gf_mul(0x57, 0x83, IRREDUCIBLE_POLY), 0xC1)
        self.assertEqual(gf_mul(0x57, 0x13, IRREDUCIBLE_POLY), 0xFE)
        self.assertEqual(gf_mul(0x53, 0xCA, IRREDUCIBLE_POLY), 0x01)

    def test_inverse_table_properties(self) -> None:
        inv_table = generate_inverse_table(IRREDUCIBLE_POLY)
        self.assertEqual(len(inv_table), 256)
        self.assertEqual(int(inv_table[0]), 0)
        self.assertEqual(gf_inv(0), 0)
        for x in range(1, 256):
            self.assertEqual(gf_mul(x, int(inv_table[x]), IRREDUCIBLE_POLY), 1)

    def test_aes_sbox_and_inverse_roundtrip(self) -> None:
        sbox = build_aes_sbox()
        self.assertEqual(
            [int(v) for v in sbox[:8]],
            [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5],
        )
        self.assertEqual(int(sbox[255]), 0x16)
        inv_sbox = build_inverse_sbox(sbox)
        self.assertTrue(np.array_equal(inv_sbox[sbox], np.arange(256, dtype=np.uint8)))


class TestSBoxMetrics(unittest.TestCase):
    """Tests for S-box hard-constraint validators and cryptographic metrics."""

    def test_validate_sbox_rejects_invalid(self) -> None:
        bad_sbox = np.zeros(256, dtype=np.uint8)
        res = validate_sbox(bad_sbox)
        self.assertFalse(res["bijective"])
        self.assertFalse(res["balanced"])
        self.assertFalse(res["valid"])

    def test_aes_standard_metrics_match_reference(self) -> None:
        sbox = build_aes_sbox()
        m = evaluate_sbox(sbox)
        self.assertTrue(m["valid"])
        self.assertEqual(m["bit_ones_count"], [128] * 8)
        self.assertEqual(m["nl"], 112)
        self.assertAlmostEqual(m["sac"], 0.5048828125, places=6)
        self.assertEqual(m["bic_nl"], 112)
        self.assertAlmostEqual(m["bic_sac"], 0.5046037946, places=6)
        self.assertAlmostEqual(m["lap"], 0.0625, places=6)
        self.assertAlmostEqual(m["dap"], 0.015625, places=6)
        self.assertAlmostEqual(m["s_score"], 0.0047433036, places=5)
        self.assertAlmostEqual(m["sv_score"], 16.0094866071, places=5)
        self.assertEqual(m["du"], 4)
        self.assertEqual(m["algebraic_degree_min"], 7)
        self.assertEqual(m["algebraic_degree_max"], 7)
        self.assertEqual(m["cycles"], [87, 81, 59, 27, 2])
        self.assertEqual(m["num_cycles"], 5)

    def test_benchmark_sbox44_metrics(self) -> None:
        matrix = circulant_matrix(SBOX44_CIRCULANT_FIRST_ROW)
        sbox44 = build_sbox(matrix, constant=DEFAULT_AFFINE_CONSTANT, poly=IRREDUCIBLE_POLY)
        m = evaluate_sbox(sbox44)
        self.assertTrue(m["valid"])
        self.assertEqual(m["nl"], 112)
        self.assertAlmostEqual(m["sac"], 0.500732421875, places=6)
        self.assertAlmostEqual(m["bic_sac"], 0.5023716517857143, places=6)
        self.assertAlmostEqual(m["s_score"], 0.0015520368, places=5)


class TestSBoxOptimizer(unittest.TestCase):
    """Tests for deterministic Hybrid GA-SA optimization."""

    def test_optimizer_determinism_and_quality(self) -> None:
        opt1 = HybridGASASboxOptimizer(seed=42, population_size=20, generations=15, sa_steps=80)
        opt2 = HybridGASASboxOptimizer(seed=42, population_size=20, generations=15, sa_steps=80)

        res1 = opt1.optimize()
        res2 = opt2.optimize()

        self.assertTrue(np.array_equal(res1.sbox, res2.sbox))
        self.assertEqual(res1.affine_constant, res2.affine_constant)
        self.assertTrue(res1.metrics["valid"])
        self.assertEqual(res1.metrics["nl"], 112)
        self.assertEqual(res1.metrics["bic_nl"], 112)
        self.assertLess(res1.metrics["s_score"], 0.0040)
        self.assertGreater(len(res1.history), 15)
        self.assertTrue(
            np.array_equal(res1.inverse_sbox[res1.sbox], np.arange(256, dtype=np.uint8))
        )

        # Verify monotonic best_s_score improvement in convergence history
        s_scores = [entry["best_s_score"] for entry in res1.history]
        self.assertEqual(s_scores, sorted(s_scores, reverse=True))

    def test_non_algebraic_optimizer_improves_nl_without_algebra(self) -> None:
        opt = NonAlgebraicWHSOptimizer(seed=42, steps=2000)
        res = opt.optimize()
        self.assertTrue(res.metrics["valid"])
        self.assertGreater(res.metrics["nl"], res.initial_metrics["nl"])
        self.assertGreaterEqual(res.metrics["nl"], 100)
        self.assertLess(res.metrics["lap"], res.initial_metrics["lap"])
        self.assertTrue(
            np.array_equal(res.inverse_sbox[res.sbox], np.arange(256, dtype=np.uint8))
        )

    def test_random_search_benchmark(self) -> None:
        bench = run_random_search_benchmark(seed=42, budget=100)
        self.assertEqual(bench["budget"], 100)
        self.assertGreater(bench["mean_s_score"], bench["best_s_score"])


if __name__ == "__main__":
    unittest.main()
