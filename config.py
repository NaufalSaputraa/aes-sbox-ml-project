"""Global configuration, cryptographic constants, and optimizer defaults."""

from pathlib import Path

# Directory and artifact paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SBOX_RESULTS_PATH: Path = DATA_DIR / "sbox_results.json"

# Galois Field GF(2^8) constants
# Standard AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
IRREDUCIBLE_POLY: int = 0x11B
GF_DEGREE: int = 8
GF_SIZE: int = 256

# Affine transformation constants
DEFAULT_AFFINE_CONSTANT: int = 0x63  # 99 decimal
# Standard AES circulant matrix first row (LSB-first: bits 0,4,5,6,7 = 1)
AES_CIRCULANT_FIRST_ROW: int = 0xF1
# Benchmark S-box44 (Alamsyah et al., 2025) circulant first row (87 decimal)
SBOX44_CIRCULANT_FIRST_ROW: int = 0x57

# Target cryptographic metrics (Alamsyah et al., 2025 & project criteria)
TARGET_METRICS: dict[str, float | int] = {
    "NL": 112,
    "SAC": 0.5,
    "BIC_NL": 112,
    "BIC_SAC": 0.5,
    "LAP": 0.0625,
    "DAP": 0.015625,
    "S_MAX": 0.0040,
    "SV_IDEAL": 16.0,
}

# Reference baseline metrics for standard AES S-box
AES_BASELINE_REFERENCE: dict[str, float | int] = {
    "NL": 112,
    "SAC": 0.50488,
    "BIC_NL": 112,
    "BIC_SAC": 0.50460,
    "LAP": 0.0625,
    "DAP": 0.015625,
    "S": 0.00474,
    "SV": 16.00948,
}

# Reproducibility & Hybrid GA-SA optimizer hyperparameters
RANDOM_SEED: int = 42
GA_POPULATION_SIZE: int = 48
GA_GENERATIONS: int = 60
GA_ELITE_SIZE: int = 6
GA_CROSSOVER_RATE: float = 0.8
GA_MUTATION_RATE: float = 0.35
SA_STEPS: int = 400
SA_INITIAL_TEMP: float = 0.005
SA_COOLING_RATE: float = 0.985
SAC_STD_TIEBREAKER_WEIGHT: float = 0.001

# Phase 3: Custom AES-128 and image evaluation constants
IMAGE_EVAL_RESULTS_PATH: Path = DATA_DIR / "image_eval_results.json"
AES_BLOCK_SIZE: int = 16
AES_KEY_SIZE: int = 16
AES_ROUNDS: int = 10
ALLOWED_AES_MODES: tuple[str, ...] = ("CBC", "CTR")
TARGET_IMAGE_METRICS: dict[str, float] = {
    "ENTROPY": 8.0,
    "NPCR": 99.6094,
    "UACI": 33.4635,
}
