"""Optimization and ML-inspired search for strong 8-bit AES S-boxes."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import (
    DEFAULT_AFFINE_CONSTANT,
    GA_CROSSOVER_RATE,
    GA_ELITE_SIZE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_POPULATION_SIZE,
    IRREDUCIBLE_POLY,
    RANDOM_SEED,
    SA_COOLING_RATE,
    SA_INITIAL_TEMP,
    SA_STEPS,
    SAC_STD_TIEBREAKER_WEIGHT,
)
from src.gf_math import (
    build_inverse_sbox,
    build_sbox,
    circulant_matrix,
    find_zero_fixed_point_constant,
    generate_inverse_table,
    is_invertible_gf2,
)
from src.sbox_metrics import (
    compute_composite_scores,
    evaluate_fast_sac_metrics,
    evaluate_sbox,
)


@dataclass
class OptimizationResult:
    """Container for the best discovered S-box, metrics, and convergence log."""

    sbox: np.ndarray
    inverse_sbox: np.ndarray
    affine_matrix: np.ndarray
    affine_constant: int
    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    algorithm: str
    seed: int
    runtime_seconds: float
    hyperparameters: dict[str, Any]


def random_invertible_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly distributed invertible 8x8 binary matrix over GF(2)."""
    while True:
        candidate = rng.integers(0, 2, size=(8, 8), dtype=np.uint8)
        if is_invertible_gf2(candidate):
            return candidate


def repair_invertible_gf2(
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Repair a binary 8x8 matrix to full rank over GF(2) by replacing dependent rows."""
    mat = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    if is_invertible_gf2(mat):
        return mat

    basis_pivots: dict[int, np.ndarray] = {}
    dependent_rows: list[int] = []

    def _reduce_row(vec: np.ndarray) -> tuple[np.ndarray, int]:
        v = vec.copy()
        for col in range(8):
            if v[col]:
                if col in basis_pivots:
                    v ^= basis_pivots[col]
                else:
                    return v, col
        return v, -1

    for r in range(8):
        reduced, pivot_col = _reduce_row(mat[r])
        if pivot_col == -1:
            dependent_rows.append(r)
        else:
            basis_pivots[pivot_col] = reduced

    for r in dependent_rows:
        while True:
            new_row = rng.integers(0, 2, size=8, dtype=np.uint8)
            reduced, pivot_col = _reduce_row(new_row)
            if pivot_col != -1:
                basis_pivots[pivot_col] = reduced
                mat[r] = new_row
                break

    return mat


def _apply_elementary_gf2_op(
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one rank-preserving elementary GF(2) row or column operation."""
    mutated = matrix.copy()
    op_type = int(rng.integers(0, 4))
    i, j = rng.choice(8, size=2, replace=False)

    if op_type == 0:
        mutated[i] ^= mutated[j]
    elif op_type == 1:
        mutated[:, i] ^= mutated[:, j]
    elif op_type == 2:
        mutated[[i, j]] = mutated[[j, i]]
    else:
        mutated[:, [i, j]] = mutated[:, [j, i]]

    return mutated


class HybridGASASboxOptimizer:
    """Hybrid Genetic Algorithm + Simulated Annealing over GL(8, 2) affine matrices."""

    def __init__(
        self,
        poly: int = IRREDUCIBLE_POLY,
        seed: int = RANDOM_SEED,
        population_size: int = GA_POPULATION_SIZE,
        generations: int = GA_GENERATIONS,
        elite_size: int = GA_ELITE_SIZE,
        crossover_rate: float = GA_CROSSOVER_RATE,
        mutation_rate: float = GA_MUTATION_RATE,
        sa_steps: int = SA_STEPS,
        sa_initial_temp: float = SA_INITIAL_TEMP,
        sa_cooling_rate: float = SA_COOLING_RATE,
        sac_std_weight: float = SAC_STD_TIEBREAKER_WEIGHT,
    ) -> None:
        self.poly = poly
        self.seed = seed
        self.population_size = population_size
        self.generations = generations
        self.elite_size = min(elite_size, population_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.sa_steps = sa_steps
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sac_std_weight = sac_std_weight
        self.inv_table = generate_inverse_table(poly)

    def _score_matrix(
        self,
        matrix: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Evaluate matrix fitness: (loss, s_score, sac, bic_sac, sac_std)."""
        sbox = build_sbox(
            matrix,
            constant=DEFAULT_AFFINE_CONSTANT,
            poly=self.poly,
            inv_table=self.inv_table,
        )
        sac, bic_sac, s_score, sac_std = evaluate_fast_sac_metrics(sbox)
        loss = s_score + self.sac_std_weight * sac_std
        return loss, s_score, sac, bic_sac, sac_std

    def _tournament_select(
        self,
        population: list[np.ndarray],
        losses: list[float],
        rng: np.random.Generator,
        k: int = 3,
    ) -> np.ndarray:
        """Select a parent matrix via k-way tournament selection."""
        indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
        best_idx = min(indices, key=lambda idx: losses[int(idx)])
        return population[int(best_idx)].copy()

    def _crossover(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Perform row-wise uniform crossover and repair to GL(8, 2)."""
        mask = rng.integers(0, 2, size=8, dtype=bool)
        child = np.where(mask[:, None], parent_a, parent_b)
        return repair_invertible_gf2(child, rng)

    def _mutate(self, matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mutate an invertible matrix using 1 or 2 elementary GF(2) operations."""
        num_ops = int(rng.integers(1, 3))
        mutated = matrix
        for _ in range(num_ops):
            mutated = _apply_elementary_gf2_op(mutated, rng)
        return mutated

    def optimize(self) -> OptimizationResult:
        """Run Hybrid GA-SA optimization and return the best validated S-box."""
        start_time = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        # Phase A: Genetic Algorithm global exploration
        population = [random_invertible_matrix(rng) for _ in range(self.population_size)]
        evaluated = [self._score_matrix(mat) for mat in population]

        history: list[dict[str, Any]] = []
        best_matrix = population[0].copy()
        best_loss, best_s, best_sac, best_bic_sac, best_sac_std = evaluated[0]

        for gen in range(1, self.generations + 1):
            losses = [item[0] for item in evaluated]
            s_scores = [item[1] for item in evaluated]
            order = np.argsort(losses)

            gen_best_idx = int(order[0])
            if losses[gen_best_idx] < best_loss:
                best_matrix = population[gen_best_idx].copy()
                best_loss, best_s, best_sac, best_bic_sac, best_sac_std = evaluated[gen_best_idx]

            _, best_sv = compute_composite_scores(112, best_sac, 112, best_bic_sac)
            history.append(
                {
                    "iteration": gen,
                    "phase": "GA",
                    "best_loss": float(best_loss),
                    "best_s_score": float(best_s),
                    "mean_s_score": float(np.mean(s_scores)),
                    "best_sv_score": float(best_sv),
                    "best_sac": float(best_sac),
                    "best_bic_sac": float(best_bic_sac),
                    "best_sac_std": float(best_sac_std),
                    "best_nl": 112,
                    "best_bic_nl": 112,
                }
            )

            next_population = [population[int(idx)].copy() for idx in order[: self.elite_size]]
            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(population, losses, rng)
                if rng.random() < self.crossover_rate:
                    parent_b = self._tournament_select(population, losses, rng)
                    child = self._crossover(parent_a, parent_b, rng)
                else:
                    child = parent_a

                if rng.random() < self.mutation_rate:
                    child = self._mutate(child, rng)

                next_population.append(child)

            population = next_population
            evaluated = [self._score_matrix(mat) for mat in population]

        # Phase B: Simulated Annealing local refinement
        current_matrix = best_matrix.copy()
        current_loss = best_loss
        current_s = best_s
        temperature = self.sa_initial_temp

        for step in range(1, self.sa_steps + 1):
            candidate = _apply_elementary_gf2_op(current_matrix, rng)
            cand_loss, cand_s, cand_sac, cand_bic_sac, cand_sac_std = self._score_matrix(candidate)

            delta = cand_loss - current_loss
            accept = delta <= 0.0 or (
                temperature > 1e-12 and rng.random() < math.exp(-delta / temperature)
            )
            if accept:
                current_matrix = candidate
                current_loss = cand_loss
                current_s = cand_s

                if cand_loss < best_loss:
                    best_matrix = candidate.copy()
                    best_loss = cand_loss
                    best_s = cand_s
                    best_sac = cand_sac
                    best_bic_sac = cand_bic_sac
                    best_sac_std = cand_sac_std

            if step % 10 == 0 or step == self.sa_steps:
                _, best_sv = compute_composite_scores(112, best_sac, 112, best_bic_sac)
                history.append(
                    {
                        "iteration": self.generations + step,
                        "phase": "SA",
                        "temperature": float(temperature),
                        "current_s_score": float(current_s),
                        "best_loss": float(best_loss),
                        "best_s_score": float(best_s),
                        "best_sv_score": float(best_sv),
                        "best_sac": float(best_sac),
                        "best_bic_sac": float(best_bic_sac),
                        "best_sac_std": float(best_sac_std),
                        "best_nl": 112,
                        "best_bic_nl": 112,
                    }
                )

            temperature *= self.sa_cooling_rate

        # Post-processing: select 8-bit constant C avoiding fixed & opposite-fixed points
        best_constant = find_zero_fixed_point_constant(
            best_matrix,
            poly=self.poly,
            preferred=DEFAULT_AFFINE_CONSTANT,
            inv_table=self.inv_table,
        )
        final_sbox = build_sbox(
            best_matrix,
            constant=best_constant,
            poly=self.poly,
            inv_table=self.inv_table,
        )
        final_inv_sbox = build_inverse_sbox(final_sbox)
        final_metrics = evaluate_sbox(final_sbox, include_details=True)

        if not final_metrics["valid"]:
            raise RuntimeError("Optimized S-box failed bijective/balance hard constraints.")
        if final_metrics["nl"] < 112:
            raise RuntimeError(f"Optimized S-box NL={final_metrics['nl']} is below 112.")

        runtime_seconds = time.perf_counter() - start_time
        return OptimizationResult(
            sbox=final_sbox,
            inverse_sbox=final_inv_sbox,
            affine_matrix=best_matrix,
            affine_constant=best_constant,
            metrics=final_metrics,
            history=history,
            algorithm="Hybrid GA-SA (GL(8,2) Affine Matrix Search)",
            seed=self.seed,
            runtime_seconds=float(runtime_seconds),
            hyperparameters={
                "poly": f"0x{self.poly:03X}",
                "population_size": self.population_size,
                "generations": self.generations,
                "elite_size": self.elite_size,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "sa_steps": self.sa_steps,
                "sa_initial_temp": self.sa_initial_temp,
                "sa_cooling_rate": self.sa_cooling_rate,
                "sac_std_weight": self.sac_std_weight,
            },
        )


def find_best_circulant_sbox(
    poly: int = IRREDUCIBLE_POLY,
    constant: int = DEFAULT_AFFINE_CONSTANT,
) -> dict[str, Any]:
    """Scan all 128 odd-weight circulant affine matrices (Alamsyah et al., 2025)."""
    inv_table = generate_inverse_table(poly)
    best_record: dict[str, Any] | None = None

    for mask in range(256):
        if bin(mask).count("1") % 2 == 1:
            matrix = circulant_matrix(mask)
            sbox = build_sbox(matrix, constant=constant, poly=poly, inv_table=inv_table)
            sac, bic_sac, s_score, sac_std = evaluate_fast_sac_metrics(sbox)
            if best_record is None or s_score < best_record["s_score"]:
                best_record = {
                    "first_row_mask": mask,
                    "matrix": matrix,
                    "sbox": sbox,
                    "s_score": s_score,
                    "sac": sac,
                    "bic_sac": bic_sac,
                    "sac_std": sac_std,
                }

    assert best_record is not None
    best_record["metrics"] = evaluate_sbox(best_record["sbox"], include_details=True)
    return best_record


# Precompute 256x256 Walsh sign matrix: (-1)^parity(a & x)
_PARITY_SIGN: np.ndarray = np.array(
    [[1 if (bin(a & x).count("1") & 1) == 0 else -1 for x in range(256)] for a in range(256)],
    dtype=np.int8,
)
_COORD_MASK_ROWS: np.ndarray = np.array([(1 << i) - 1 for i in range(8)], dtype=np.int64)


@dataclass
class NonAlgebraicOptimizationResult:
    """Container for pure non-algebraic S-box optimization (no GF(2^8) or affine matrix)."""

    initial_sbox: np.ndarray
    initial_metrics: dict[str, Any]
    sbox: np.ndarray
    inverse_sbox: np.ndarray
    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    algorithm: str
    seed: int
    runtime_seconds: float
    hyperparameters: dict[str, Any]


class NonAlgebraicWHSOptimizer:
    """Pure non-algebraic Simulated Annealing using Clark's WHS spectral cost."""

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        steps: int = 12000,
        whs_power: float = 12.0,
    ) -> None:
        self.seed = seed
        self.steps = steps
        self.whs_power = whs_power

    def optimize(self) -> NonAlgebraicOptimizationResult:
        """Optimize a random permutation directly via incremental WHT swap updates."""
        start_time = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        initial_sbox = rng.permutation(256).astype(np.uint8)
        initial_metrics = evaluate_sbox(initial_sbox, include_details=True)

        lut = (np.arange(257, dtype=np.float64)) ** self.whs_power
        sbox = initial_sbox.copy()

        # Full Walsh-Hadamard Table W of shape (255, 256) for output masks 1..255
        p_b = _PARITY_SIGN[1:, sbox].astype(np.int32)
        w_table = (p_b @ _PARITY_SIGN.T).astype(np.int16)
        current_cost = float(lut[np.abs(w_table)].sum())
        current_max_w = int(np.abs(w_table).max())

        # Calibrate initial temperature dynamically from 100 probe swaps
        probe_deltas: list[float] = []
        for _ in range(100):
            u, v = rng.choice(256, size=2, replace=False)
            y_u, y_v = int(sbox[u]), int(sbox[v])
            dy = (_PARITY_SIGN[1:, y_v] - _PARITY_SIGN[1:, y_u]).astype(np.int16)
            dx = (_PARITY_SIGN[u, :] - _PARITY_SIGN[v, :]).astype(np.int16)
            bm = dy != 0
            am = dx != 0
            sub_old = w_table[np.ix_(bm, am)]
            sub_new = sub_old + np.outer(dy[bm], dx[am])
            delta_c = abs(float(lut[np.abs(sub_new)].sum() - lut[np.abs(sub_old)].sum()))
            if delta_c > 0.0:
                probe_deltas.append(delta_c)

        med_delta = float(np.median(probe_deltas)) if probe_deltas else 1.0
        initial_temp = max(med_delta / 0.693, 1.0)
        cooling_rate = 0.0005 ** (1.0 / max(self.steps, 1))
        temperature = initial_temp

        best_sbox = sbox.copy()
        best_cost = current_cost
        best_max_w = current_max_w
        best_coord_nl = int(128 - int(np.abs(w_table[_COORD_MASK_ROWS]).max()) // 2)

        history: list[dict[str, Any]] = []
        log_interval = max(self.steps // 60, 1)

        for step in range(1, self.steps + 1):
            u = int(rng.integers(0, 256))
            v = int(rng.integers(0, 255))
            if v >= u:
                v += 1

            y_u, y_v = int(sbox[u]), int(sbox[v])
            dy = (_PARITY_SIGN[1:, y_v] - _PARITY_SIGN[1:, y_u]).astype(np.int16)
            dx = (_PARITY_SIGN[u, :] - _PARITY_SIGN[v, :]).astype(np.int16)
            bm = dy != 0
            am = dx != 0

            sub_old = w_table[np.ix_(bm, am)]
            sub_new = sub_old + np.outer(dy[bm], dx[am])
            delta_cost = float(lut[np.abs(sub_new)].sum() - lut[np.abs(sub_old)].sum())

            accept = delta_cost <= 0.0 or (
                temperature > 1e-12 and rng.random() < math.exp(-delta_cost / temperature)
            )
            if accept:
                w_table[np.ix_(bm, am)] = sub_new
                sbox[u], sbox[v] = sbox[v], sbox[u]
                current_cost += delta_cost
                current_max_w = int(np.abs(w_table).max())

                if (current_max_w < best_max_w) or (
                    current_max_w == best_max_w and current_cost < best_cost
                ):
                    best_max_w = current_max_w
                    best_cost = current_cost
                    best_sbox = sbox.copy()
                    best_coord_nl = int(128 - int(np.abs(w_table[_COORD_MASK_ROWS]).max()) // 2)

            if step % log_interval == 0 or step == self.steps:
                cur_nl_all = int(128 - current_max_w // 2)
                best_nl_all = int(128 - best_max_w // 2)
                history.append(
                    {
                        "iteration": step,
                        "temperature": float(temperature),
                        "current_nl_all": cur_nl_all,
                        "best_nl_all": best_nl_all,
                        "best_coord_nl": best_coord_nl,
                        "best_lap": float(best_max_w / 512.0),
                        "best_whs_cost": float(best_cost),
                    }
                )

            temperature *= cooling_rate

        final_inv_sbox = build_inverse_sbox(best_sbox)
        final_metrics = evaluate_sbox(best_sbox, include_details=True)
        runtime_seconds = time.perf_counter() - start_time

        return NonAlgebraicOptimizationResult(
            initial_sbox=initial_sbox,
            initial_metrics=initial_metrics,
            sbox=best_sbox,
            inverse_sbox=final_inv_sbox,
            metrics=final_metrics,
            history=history,
            algorithm="Non-Algebraic WHS Simulated Annealing (Permutation Space, Zero GF(2^8))",
            seed=self.seed,
            runtime_seconds=float(runtime_seconds),
            hyperparameters={
                "steps": self.steps,
                "whs_power": self.whs_power,
                "initial_temp": float(initial_temp),
                "cooling_rate": float(cooling_rate),
            },
        )


def run_random_search_benchmark(
    poly: int = IRREDUCIBLE_POLY,
    seed: int = RANDOM_SEED,
    budget: int = GA_POPULATION_SIZE * GA_GENERATIONS + SA_STEPS,
    sac_std_weight: float = SAC_STD_TIEBREAKER_WEIGHT,
) -> dict[str, Any]:
    """Sample `budget` random invertible GL(8, 2) matrices as a fair baseline vs Hybrid GA-SA."""
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed + 1000)
    inv_table = generate_inverse_table(poly)

    best_loss = float("inf")
    best_s = float("inf")
    best_sac = 0.0
    best_bic_sac = 0.0
    best_sac_std = 0.0
    s_scores: list[float] = []

    for _ in range(budget):
        mat = random_invertible_matrix(rng)
        sbox = build_sbox(mat, constant=DEFAULT_AFFINE_CONSTANT, poly=poly, inv_table=inv_table)
        sac, bic_sac, s_score, sac_std = evaluate_fast_sac_metrics(sbox)
        loss = s_score + sac_std_weight * sac_std
        s_scores.append(float(s_score))
        if loss < best_loss:
            best_loss = float(loss)
            best_s = float(s_score)
            best_sac = float(sac)
            best_bic_sac = float(bic_sac)
            best_sac_std = float(sac_std)

    return {
        "budget": int(budget),
        "best_s_score": best_s,
        "mean_s_score": float(np.mean(s_scores)) if s_scores else 0.0,
        "best_loss": best_loss,
        "best_sac": best_sac,
        "best_bic_sac": best_bic_sac,
        "best_sac_std": best_sac_std,
        "runtime_seconds": float(time.perf_counter() - start_time),
    }
