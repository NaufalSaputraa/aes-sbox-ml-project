"""CLI runner for Fase 3: Custom AES-128 image encryption & security evaluation."""

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
    ALLOWED_AES_MODES,
    IMAGE_EVAL_RESULTS_PATH,
    RANDOM_SEED,
    SBOX_RESULTS_PATH,
    TARGET_IMAGE_METRICS,
)
from src.image_eval import (
    evaluate_image_encryption,
    generate_sample_images,
    load_image_array,
    load_sbox_results,
)


def _derive_locked_key_and_iv(seed: int) -> tuple[bytes, bytes]:
    """Derive deterministic 16-byte AES key and 16-byte IV/nonce from a locked seed."""
    rng = np.random.default_rng(seed)
    key = bytes(int(b) for b in rng.integers(0, 256, size=16, dtype=np.uint8))
    iv = bytes(int(b) for b in rng.integers(0, 256, size=16, dtype=np.uint8))
    return key, iv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reproducible Fase 3 image evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Fase 3 AES-128 CBC/CTR image encryption & security evaluation."
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Locked random seed")
    parser.add_argument("--size", type=int, default=128, help="Sample image width/height")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional external image path to evaluate instead of built-in samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMAGE_EVAL_RESULTS_PATH,
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Fase 3 image encryption evaluation across S-boxes and modes."""
    args = parse_args()
    sbox_data = load_sbox_results(SBOX_RESULTS_PATH)
    key, iv_or_nonce = _derive_locked_key_and_iv(args.seed)

    sboxes: dict[str, np.ndarray] = {
        "aes_standard": np.asarray(sbox_data["aes_standard"]["sbox_flat"], dtype=np.uint8),
        "benchmark_sbox44": np.asarray(sbox_data["benchmark_sbox44"]["sbox_flat"], dtype=np.uint8),
        "custom_sbox": np.asarray(sbox_data["custom_sbox"]["sbox_flat"], dtype=np.uint8),
    }

    if args.image is not None:
        images: dict[str, np.ndarray] = {
            args.image.stem: load_image_array(args.image),
        }
    else:
        images = generate_sample_images(size=(args.size, args.size))

    summary_rows: list[dict[str, Any]] = []
    detailed_results: dict[str, Any] = {}

    for img_name, img_arr in images.items():
        detailed_results[img_name] = {}
        for mode in ALLOWED_AES_MODES:
            detailed_results[img_name][mode] = {}
            for sbox_name, sbox_arr in sboxes.items():
                res = evaluate_image_encryption(
                    img_arr,
                    key=key,
                    sbox=sbox_arr,
                    mode=mode,
                    iv_or_nonce=iv_or_nonce,
                )

                summary_rows.append(
                    {
                        "Image": img_name,
                        "Mode": mode,
                        "S-box": sbox_name,
                        "Lossless": res["is_lossless"],
                        "Entropy(Orig)": f"{res['entropy_original']:.5f}",
                        "Entropy(Enc)": f"{res['entropy_encrypted']:.5f}",
                        "NPCR(%)": f"{res['npcr']:.4f}",
                        "UACI(%)": f"{res['uaci']:.4f}",
                        "Chi2(Orig)": f"{res['chi_square_original']['mean']:.1f}",
                        "Chi2(Enc)": f"{res['chi_square_encrypted']['mean']:.1f}",
                        "Enc(ms)": f"{res['encryption_time_ms']:.2f}",
                        "Dec(ms)": f"{res['decryption_time_ms']:.2f}",
                    }
                )

                detailed_results[img_name][mode][sbox_name] = {
                    "mode": res["mode"],
                    "shape": res["shape"],
                    "iv_or_nonce_hex": res["iv_or_nonce_hex"],
                    "is_lossless": bool(res["is_lossless"]),
                    "mse": float(res["mse"]),
                    "max_abs_diff": int(res["max_abs_diff"]),
                    "entropy_original": float(res["entropy_original"]),
                    "entropy_encrypted": float(res["entropy_encrypted"]),
                    "channel_entropy_original": res["channel_entropy_original"],
                    "channel_entropy_encrypted": res["channel_entropy_encrypted"],
                    "npcr": float(res["npcr"]),
                    "uaci": float(res["uaci"]),
                    "npcr_plain_vs_cipher": float(res["npcr_plain_vs_cipher"]),
                    "uaci_plain_vs_cipher": float(res["uaci_plain_vs_cipher"]),
                    "chi_square_original": res["chi_square_original"],
                    "chi_square_encrypted": res["chi_square_encrypted"],
                    "encryption_time_ms": float(res["encryption_time_ms"]),
                    "decryption_time_ms": float(res["decryption_time_ms"]),
                }

    df = pd.DataFrame(summary_rows)
    print("=" * 100)
    print("AES-128 Custom S-Box Image Encryption & Security Evaluation (Fase 3)")
    print("=" * 100)
    print(f"Locked Seed : {args.seed}")
    print(f"AES Key Hex : {key.hex()}")
    print(f"IV/Nonce Hex: {iv_or_nonce.hex()}")
    print(f"Target      : Entropy~{TARGET_IMAGE_METRICS['ENTROPY']}, NPCR~{TARGET_IMAGE_METRICS['NPCR']}%, UACI~{TARGET_IMAGE_METRICS['UACI']}%")
    print("-" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
            "key_hex": key.hex(),
            "iv_or_nonce_hex": iv_or_nonce.hex(),
            "target_image_metrics": TARGET_IMAGE_METRICS,
        },
        "summary_table": summary_rows,
        "results": detailed_results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved image evaluation results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
