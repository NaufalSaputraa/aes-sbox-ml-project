"""Digital image encryption/decryption and security evaluation (Entropy, NPCR, UACI, Histogram)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from config import SBOX_RESULTS_PATH
from src.aes_custom import AESCustom


def load_sbox_results(path: Path = SBOX_RESULTS_PATH) -> dict[str, Any]:
    """Load S-box optimization results JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sbox_from_results(
    name: str = "custom_sbox",
    path: Path = SBOX_RESULTS_PATH,
) -> np.ndarray:
    """Extract a flat 256-element uint8 S-box from `sbox_results.json`."""
    data = load_sbox_results(path)
    if name not in data or "sbox_flat" not in data[name]:
        raise KeyError(f"S-box entry {name!r} with 'sbox_flat' not found in {path}.")
    return np.asarray(data[name]["sbox_flat"], dtype=np.uint8)


def load_image_array(
    source: str | Path | Image.Image | np.ndarray,
    color_mode: str | None = None,
) -> np.ndarray:
    """Load an image source into a contiguous 2D (grayscale) or 3D (RGB) uint8 NumPy array."""
    if isinstance(source, np.ndarray):
        arr = np.asarray(source, dtype=np.uint8)
        if arr.ndim == 2:
            if color_mode == "RGB":
                return np.ascontiguousarray(np.stack((arr, arr, arr), axis=-1))
            return np.ascontiguousarray(arr)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
            if color_mode == "L":
                gray = np.asarray(Image.fromarray(rgb, mode="RGB").convert("L"), dtype=np.uint8)
                return np.ascontiguousarray(gray)
            return np.ascontiguousarray(rgb)
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

    if isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)

    if color_mode is not None:
        img = img.convert(color_mode)
    elif img.mode in ("L", "1", "P") and img.mode == "L":
        img = img.convert("L")
    elif img.mode == "L":
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def generate_sample_images(size: tuple[int, int] = (128, 128)) -> dict[str, np.ndarray]:
    """Generate deterministic structured grayscale and RGB test images with non-uniform histograms."""
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]

    # Structured grayscale image: radial rings + diagonal gradient + central block
    cx, cy = width / 2.0, height / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gray = (
        0.55 * (xx / max(width - 1, 1)) * 255.0
        + 0.45 * (np.sin(dist / 6.0) * 0.5 + 0.5) * 255.0
    )
    gray_arr = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    gray_arr[height // 4 : height // 2, width // 4 : width // 2] = 40

    # Structured RGB image: distinct R, G, B spatial gradients + geometric overlay
    r = np.clip(np.round((xx / max(width - 1, 1)) * 220.0 + 20.0), 0, 255).astype(np.uint8)
    g = np.clip(np.round((yy / max(height - 1, 1)) * 200.0 + 30.0), 0, 255).astype(np.uint8)
    b = np.clip(np.round((np.cos(dist / 8.0) * 0.5 + 0.5) * 240.0), 0, 255).astype(np.uint8)
    rgb_arr = np.stack((r, g, b), axis=-1)
    rgb_arr[height // 3 : (2 * height) // 3, width // 3 : (2 * width) // 3] = [25, 180, 230]

    return {
        "grayscale_sample": np.ascontiguousarray(gray_arr),
        "rgb_sample": np.ascontiguousarray(rgb_arr),
    }


def encrypt_image(
    image: str | Path | Image.Image | np.ndarray,
    key: bytes,
    sbox: Sequence[int] | np.ndarray | None = None,
    mode: str = "CBC",
    iv_or_nonce: bytes | None = None,
) -> dict[str, Any]:
    """Encrypt a grayscale or RGB image using AESCustom (CBC or CTR)."""
    img_arr = load_image_array(image)
    cipher = AESCustom(key=key, sbox=sbox)
    raw_bytes = img_arr.tobytes()

    t0 = time.perf_counter()
    ciphertext_bytes, used_iv_or_nonce = cipher.encrypt(
        raw_bytes, mode=mode, iv_or_nonce=iv_or_nonce
    )
    enc_ms = (time.perf_counter() - t0) * 1000.0

    display_bytes = ciphertext_bytes[: img_arr.nbytes]
    encrypted_image = np.frombuffer(display_bytes, dtype=np.uint8).reshape(img_arr.shape).copy()

    return {
        "ciphertext_bytes": ciphertext_bytes,
        "encrypted_image": encrypted_image,
        "iv_or_nonce": used_iv_or_nonce,
        "shape": img_arr.shape,
        "orig_nbytes": img_arr.nbytes,
        "mode": mode.strip().upper(),
        "encryption_time_ms": enc_ms,
    }


def decrypt_image(
    ciphertext_bytes: bytes,
    shape: tuple[int, ...],
    key: bytes,
    sbox: Sequence[int] | np.ndarray | None = None,
    mode: str = "CBC",
    iv_or_nonce: bytes = b"",
) -> tuple[np.ndarray, float]:
    """Decrypt ciphertext bytes back into an image array of `shape`."""
    cipher = AESCustom(key=key, sbox=sbox)
    t0 = time.perf_counter()
    plain_bytes = cipher.decrypt(ciphertext_bytes, iv_or_nonce=iv_or_nonce, mode=mode)
    dec_ms = (time.perf_counter() - t0) * 1000.0

    decrypted = np.frombuffer(plain_bytes, dtype=np.uint8).reshape(shape).copy()
    return decrypted, dec_ms


def verify_lossless(original: np.ndarray, decrypted: np.ndarray) -> dict[str, Any]:
    """Verify exact pixel-level lossless recovery between original and decrypted images."""
    orig = np.asarray(original, dtype=np.uint8)
    dec = np.asarray(decrypted, dtype=np.uint8)
    if orig.shape != dec.shape:
        return {
            "is_lossless": False,
            "mse": float("inf"),
            "max_abs_diff": 255,
            "psnr": 0.0,
        }

    diff = orig.astype(np.int32) - dec.astype(np.int32)
    mse = float(np.mean(diff * diff))
    max_diff = int(np.max(np.abs(diff))) if diff.size > 0 else 0
    is_lossless = bool(np.array_equal(orig, dec))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))

    return {
        "is_lossless": is_lossless,
        "mse": mse,
        "max_abs_diff": max_diff,
        "psnr": psnr,
    }


def compute_shannon_entropy(image: np.ndarray) -> float:
    """Compute global 8-bit Shannon entropy over all pixel/channel byte values."""
    arr = np.asarray(image, dtype=np.uint8).ravel()
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / float(arr.size)
    return float(-np.sum(probs * np.log2(probs)))


def compute_channel_entropies(image: np.ndarray) -> dict[str, float]:
    """Compute Shannon entropy per channel (`Gray` or `R`, `G`, `B`)."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        return {"Gray": compute_shannon_entropy(arr)}
    if arr.ndim == 3 and arr.shape[2] == 3:
        return {
            name: compute_shannon_entropy(arr[:, :, idx])
            for idx, name in enumerate(("R", "G", "B"))
        }
    raise ValueError(f"Unsupported image shape for entropy: {arr.shape}")


def compute_npcr(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """Compute Number of Pixels Change Rate (NPCR %) between two encrypted images."""
    c1 = np.asarray(cipher1, dtype=np.uint8)
    c2 = np.asarray(cipher2, dtype=np.uint8)
    if c1.shape != c2.shape:
        raise ValueError(f"Cipher images must have identical shapes, got {c1.shape} vs {c2.shape}.")
    return float(np.mean(c1 != c2) * 100.0)


def compute_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """Compute Unified Average Changing Intensity (UACI %) between two encrypted images."""
    c1 = np.asarray(cipher1, dtype=np.uint8).astype(np.float64)
    c2 = np.asarray(cipher2, dtype=np.uint8).astype(np.float64)
    if c1.shape != c2.shape:
        raise ValueError(f"Cipher images must have identical shapes, got {c1.shape} vs {c2.shape}.")
    return float(np.mean(np.abs(c1 - c2) / 255.0) * 100.0)


def compute_histogram(image: np.ndarray) -> dict[str, list[int]]:
    """Compute 256-bin intensity histograms per channel (`Gray` or `R`, `G`, `B`)."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        return {"Gray": [int(x) for x in np.bincount(arr.ravel(), minlength=256)]}
    if arr.ndim == 3 and arr.shape[2] == 3:
        return {
            name: [int(x) for x in np.bincount(arr[:, :, idx].ravel(), minlength=256)]
            for idx, name in enumerate(("R", "G", "B"))
        }
    raise ValueError(f"Unsupported image shape for histogram: {arr.shape}")


def compute_histogram_uniformity(histograms: dict[str, list[int]]) -> dict[str, float]:
    """Compute chi-square uniformity statistic per channel (lower means more uniform)."""
    result: dict[str, float] = {}
    for ch, bins in histograms.items():
        counts = np.asarray(bins, dtype=np.float64)
        expected = float(np.sum(counts)) / 256.0
        if expected == 0.0:
            result[ch] = 0.0
        else:
            result[ch] = float(np.sum(((counts - expected) ** 2) / expected))
    result["mean"] = float(np.mean(list(result.values()))) if result else 0.0
    return result


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient safely."""
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    sx = float(np.std(xf))
    sy = float(np.std(yf))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.mean((xf - np.mean(xf)) * (yf - np.mean(yf))) / (sx * sy))


def compute_adjacent_pixel_correlation(
    image: np.ndarray,
    max_pairs: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Horizontal, Vertical, and Diagonal adjacent-pixel correlation."""
    arr = np.asarray(image, dtype=np.uint8)
    channels = [arr] if arr.ndim == 2 else [arr[:, :, c] for c in range(arr.shape[2])]
    rng = np.random.default_rng(seed)

    horiz_vals: list[float] = []
    vert_vals: list[float] = []
    diag_vals: list[float] = []

    for ch in channels:
        h_a, h_b = ch[:, :-1].ravel(), ch[:, 1:].ravel()
        v_a, v_b = ch[:-1, :].ravel(), ch[1:, :].ravel()
        d_a, d_b = ch[:-1, :-1].ravel(), ch[1:, 1:].ravel()

        if h_a.size > max_pairs:
            idx = rng.choice(h_a.size, size=max_pairs, replace=False)
            h_a, h_b = h_a[idx], h_b[idx]
        if v_a.size > max_pairs:
            idx = rng.choice(v_a.size, size=max_pairs, replace=False)
            v_a, v_b = v_a[idx], v_b[idx]
        if d_a.size > max_pairs:
            idx = rng.choice(d_a.size, size=max_pairs, replace=False)
            d_a, d_b = d_a[idx], d_b[idx]

        horiz_vals.append(_pearson_corr(h_a, h_b))
        vert_vals.append(_pearson_corr(v_a, v_b))
        diag_vals.append(_pearson_corr(d_a, d_b))

    return {
        "horizontal": float(np.mean(horiz_vals)),
        "vertical": float(np.mean(vert_vals)),
        "diagonal": float(np.mean(diag_vals)),
    }


def compute_npcr_uaci_critical_test(
    cipher1: np.ndarray,
    cipher2: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run Wu et al. (2011) NPCR and UACI statistical hypothesis test at alpha=0.05."""
    npcr = compute_npcr(cipher1, cipher2)
    uaci = compute_uaci(cipher1, cipher2)

    n_pixels = int(np.asarray(cipher1).size)
    f_val = 255.0
    z_alpha = 1.6448536269514722  # one-sided alpha = 0.05
    z_half = 1.959963984540054    # two-sided alpha = 0.05

    mu_npcr = (f_val / (f_val + 1.0)) * 100.0
    sigma_npcr = float(np.sqrt(f_val / (((f_val + 1.0) ** 2) * n_pixels)) * 100.0)
    npcr_critical = float(mu_npcr - z_alpha * sigma_npcr)

    mu_uaci = ((f_val + 2.0) / (3.0 * f_val + 3.0)) * 100.0
    sigma_uaci = float(
        np.sqrt(
            ((f_val + 2.0) * (f_val * f_val + 2.0 * f_val + 3.0))
            / (18.0 * ((f_val + 1.0) ** 2) * f_val * n_pixels)
        )
        * 100.0
    )
    uaci_lower = float(mu_uaci - z_half * sigma_uaci)
    uaci_upper = float(mu_uaci + z_half * sigma_uaci)

    return {
        "alpha": float(alpha),
        "npcr": float(npcr),
        "npcr_critical": npcr_critical,
        "npcr_pass": bool(npcr >= npcr_critical),
        "uaci": float(uaci),
        "uaci_lower": uaci_lower,
        "uaci_upper": uaci_upper,
        "uaci_pass": bool(uaci_lower <= uaci <= uaci_upper),
    }


def evaluate_text_encryption(
    plaintext: str,
    key: bytes,
    sbox: Sequence[int] | np.ndarray | None = None,
    mode: str = "CBC",
    iv_or_nonce: bytes | None = None,
) -> dict[str, Any]:
    """Encrypt/decrypt UTF-8 text and measure 1-bit plaintext avalanche (%)."""
    cipher = AESCustom(key=key, sbox=sbox)
    raw = plaintext.encode("utf-8")

    t0 = time.perf_counter()
    ct, used_iv = cipher.encrypt(raw, mode=mode, iv_or_nonce=iv_or_nonce)
    enc_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    dec_raw = cipher.decrypt(ct, iv_or_nonce=used_iv, mode=mode)
    dec_ms = (time.perf_counter() - t1) * 1000.0

    dec_text = dec_raw.decode("utf-8", errors="replace")
    is_exact = dec_raw == raw

    if raw:
        flipped = bytes([raw[0] ^ 0x01]) + raw[1:]
        ct_flip, _ = cipher.encrypt(flipped, mode=mode, iv_or_nonce=used_iv)
        c1 = np.frombuffer(ct, dtype=np.uint8)
        c2 = np.frombuffer(ct_flip, dtype=np.uint8)
        diff_bits = int(np.unpackbits(np.bitwise_xor(c1, c2)).sum())
        total_bits = int(len(ct) * 8)
        avalanche_pct = float(diff_bits / total_bits * 100.0) if total_bits > 0 else 0.0
    else:
        diff_bits = 0
        total_bits = 0
        avalanche_pct = 0.0

    return {
        "mode": mode.strip().upper(),
        "iv_or_nonce_hex": used_iv.hex(),
        "ciphertext_hex": ct.hex(),
        "decrypted_text": dec_text,
        "is_exact_match": bool(is_exact),
        "bit_avalanche_pct": avalanche_pct,
        "differing_bits": diff_bits,
        "total_bits": total_bits,
        "encryption_time_ms": float(enc_ms),
        "decryption_time_ms": float(dec_ms),
    }


def evaluate_image_encryption(
    image: str | Path | Image.Image | np.ndarray,
    key: bytes,
    sbox: Sequence[int] | np.ndarray | None = None,
    mode: str = "CBC",
    iv_or_nonce: bytes | None = None,
    pixel_pos: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Run full image encryption, lossless decryption verification, and security metrics."""
    orig_arr = load_image_array(image)
    enc_res = encrypt_image(orig_arr, key=key, sbox=sbox, mode=mode, iv_or_nonce=iv_or_nonce)
    used_iv = enc_res["iv_or_nonce"]
    enc_arr = enc_res["encrypted_image"]

    dec_arr, dec_ms = decrypt_image(
        enc_res["ciphertext_bytes"],
        shape=orig_arr.shape,
        key=key,
        sbox=sbox,
        mode=mode,
        iv_or_nonce=used_iv,
    )
    lossless_info = verify_lossless(orig_arr, dec_arr)

    # 1-pixel differential test using the exact same key and IV/nonce
    mod_arr = orig_arr.copy()
    py, px = pixel_pos
    if mod_arr.ndim == 2:
        mod_arr[py, px] = np.uint8((int(mod_arr[py, px]) + 1) % 256)
    else:
        mod_arr[py, px, 0] = np.uint8((int(mod_arr[py, px, 0]) + 1) % 256)

    enc_mod_res = encrypt_image(mod_arr, key=key, sbox=sbox, mode=mode, iv_or_nonce=used_iv)
    enc_mod_arr = enc_mod_res["encrypted_image"]

    hist_orig = compute_histogram(orig_arr)
    hist_enc = compute_histogram(enc_arr)
    wu2011 = compute_npcr_uaci_critical_test(enc_arr, enc_mod_arr)

    return {
        "mode": enc_res["mode"],
        "shape": list(orig_arr.shape),
        "iv_or_nonce_hex": used_iv.hex(),
        "original_image": orig_arr,
        "encrypted_image": enc_arr,
        "decrypted_image": dec_arr,
        "ciphertext_bytes": enc_res["ciphertext_bytes"],
        "is_lossless": lossless_info["is_lossless"],
        "mse": lossless_info["mse"],
        "max_abs_diff": lossless_info["max_abs_diff"],
        "entropy_original": compute_shannon_entropy(orig_arr),
        "entropy_encrypted": compute_shannon_entropy(enc_arr),
        "channel_entropy_original": compute_channel_entropies(orig_arr),
        "channel_entropy_encrypted": compute_channel_entropies(enc_arr),
        "npcr": wu2011["npcr"],
        "uaci": wu2011["uaci"],
        "wu2011_test": wu2011,
        "npcr_plain_vs_cipher": compute_npcr(orig_arr, enc_arr),
        "uaci_plain_vs_cipher": compute_uaci(orig_arr, enc_arr),
        "correlation_original": compute_adjacent_pixel_correlation(orig_arr),
        "correlation_encrypted": compute_adjacent_pixel_correlation(enc_arr),
        "histogram_original": hist_orig,
        "histogram_encrypted": hist_enc,
        "chi_square_original": compute_histogram_uniformity(hist_orig),
        "chi_square_encrypted": compute_histogram_uniformity(hist_enc),
        "encryption_time_ms": enc_res["encryption_time_ms"],
        "decryption_time_ms": dec_ms,
    }
