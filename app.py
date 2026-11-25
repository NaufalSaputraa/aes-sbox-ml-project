"""Interactive Streamlit dashboard for AES S-box ML construction & image encryption."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import (
    ALLOWED_AES_MODES,
    DATA_DIR,
    IMAGE_EVAL_RESULTS_PATH,
    IRREDUCIBLE_POLY,
    RANDOM_SEED,
    SBOX_RESULTS_PATH,
    TARGET_IMAGE_METRICS,
    TARGET_METRICS,
)
from src.image_eval import (
    evaluate_image_encryption,
    generate_sample_images,
    load_image_array,
)

SBOX_LABELS: dict[str, str] = {
    "custom_sbox": "Custom GA-SA (Hasil Optimasi)",
    "aes_standard": "AES Standar (FIPS-197)",
    "benchmark_sbox44": "S-box44 (Alamsyah et al., 2025)",
}

DEFAULT_KEY_HEX = "8826d916cdfb21c6c1ff91a761565a70"
DEFAULT_IV_HEX = "2416da6ec212cddb8d8800160eb686b2"
REAL_SAMPLE_IMAGE_PATH = DATA_DIR / "samples" / "foto_asli.jpg"
REAL_EVAL_RESULTS_PATH = DATA_DIR / "image_eval_real.json"


@st.cache_data
def load_json_cached(path_str: str) -> dict[str, Any] | None:
    """Load a JSON file from disk if it exists."""
    path = Path(path_str)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 grayscale or RGB NumPy array into PNG bytes."""
    mode = "L" if arr.ndim == 2 else "RGB"
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _resize_if_needed(arr: np.ndarray, max_dim: int | None) -> np.ndarray:
    """Downscale image preserving aspect ratio when max dimension exceeds `max_dim`."""
    if max_dim is None:
        return arr
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return arr
    scale = max_dim / float(max(h, w))
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    mode = "L" if arr.ndim == 2 else "RGB"
    resized = Image.fromarray(arr, mode=mode).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def _resolve_aes_key(key_mode: str, key_input: str) -> tuple[bytes, str | None]:
    """Parse 16-byte AES-128 key from hex string or SHA-256 passphrase."""
    if key_mode == "Hex (32 karakter)":
        cleaned = key_input.strip().replace(" ", "")
        try:
            raw = bytes.fromhex(cleaned)
        except ValueError:
            return bytes.fromhex(DEFAULT_KEY_HEX), "Format hex tidak valid; menggunakan kunci default."
        if len(raw) != 16:
            return (
                bytes.fromhex(DEFAULT_KEY_HEX),
                f"Panjang kunci hex harus 16 byte (32 karakter), terdeteksi {len(raw)} byte; menggunakan kunci default.",
            )
        return raw, None
    digest = hashlib.sha256(key_input.encode("utf-8")).digest()[:16]
    return digest, None


def _render_sbox_grid_df(grid: list[list[Any]]) -> pd.DataFrame:
    """Format a 16x16 grid with standard hex row/column headers."""
    row_labels = [f"{r * 16:02X}" for r in range(16)]
    col_labels = [f"{c:X}" for c in range(16)]
    return pd.DataFrame(grid, index=row_labels, columns=col_labels)


def _plot_sbox_deviation_comparison(sbox_data: dict[str, Any]) -> plt.Figure:
    """Bar chart comparing |SAC-0.5|, |BIC-SAC-0.5|, and S score across S-boxes."""
    keys = ["aes_standard", "benchmark_sbox44", "custom_sbox"]
    labels = ["AES Standar", "S-box44 (2025)", "Custom GA-SA"]
    sac_dev = [abs(sbox_data[k]["metrics"]["sac"] - 0.5) for k in keys]
    bic_dev = [abs(sbox_data[k]["metrics"]["bic_sac"] - 0.5) for k in keys]
    s_scores = [sbox_data[k]["metrics"]["s_score"] for k in keys]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - width, sac_dev, width, label="|SAC - 0.5|", color="#3b82f6")
    ax.bar(x, bic_dev, width, label="|BIC-SAC - 0.5|", color="#8b5cf6")
    ax.bar(x + width, s_scores, width, label="Skor Komposit (S)", color="#10b981")

    ax.axhline(TARGET_METRICS["S_MAX"], color="#ef4444", linestyle="--", linewidth=1.2, label="Batas S <= 0.0040")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Deviasi (Lebih Kecil Lebih Baik)")
    ax.set_title("Perbandingan Deviasi SAC, BIC-SAC, dan Skor Komposit S")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_convergence_history(history: list[dict[str, Any]]) -> plt.Figure:
    """Line chart of Hybrid GA-SA convergence across GA and SA phases."""
    iters = [h["iteration"] for h in history]
    best_s = [h["best_s_score"] for h in history]
    phases = [h["phase"] for h in history]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(iters, best_s, color="#0ea5e9", linewidth=2.0, label="Best S Score")

    if "SA" in phases:
        first_sa_idx = phases.index("SA")
        sa_start_iter = iters[first_sa_idx]
        ax.axvline(sa_start_iter, color="#f59e0b", linestyle="--", linewidth=1.3, label="Transisi GA -> SA")

    ax.set_xlabel("Iterasi (Generasi GA / Langkah SA)")
    ax.set_ylabel("Best Skor Komposit (S)")
    ax.set_title("Kurva Konvergensi Hybrid GA-SA")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_sac_heatmap(sac_matrix: list[list[float]], title: str) -> plt.Figure:
    """Annotated 8x8 SAC matrix heatmap centered at 0.50."""
    mat = np.asarray(sac_matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    im = ax.imshow(mat, cmap="coolwarm", vmin=0.40, vmax=0.60)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probabilitas Flip")

    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="black", fontsize=7.5)

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("Bit Output (j)")
    ax.set_ylabel("Bit Input (i)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _plot_histograms(histograms: dict[str, list[int]], title: str) -> plt.Figure:
    """Plot 256-bin pixel intensity histograms per channel."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.arange(256)
    color_map = {"Gray": "#334155", "R": "#ef4444", "G": "#22c55e", "B": "#3b82f6"}

    for ch, counts in histograms.items():
        ax.plot(bins, counts, label=ch, color=color_map.get(ch, "#64748b"), linewidth=1.2, alpha=0.85)
        ax.fill_between(bins, counts, color=color_map.get(ch, "#64748b"), alpha=0.12)

    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensitas Piksel (0–255)")
    ax.set_ylabel("Frekuensi")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_image_benchmark(rows: list[dict[str, Any]], title_prefix: str) -> plt.Figure:
    """Plot Entropy and CBC NPCR/UACI comparisons across S-boxes."""
    df = pd.DataFrame(rows).copy()
    for col in ("Entropy(Enc)", "NPCR(%)", "UACI(%)"):
        df[col] = df[col].astype(float)

    cbc_df = df[df["Mode"] == "CBC"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    if not cbc_df.empty:
        labels = [f"{r['Image']}\n({r['S-box']})" for _, r in cbc_df.iterrows()]
        x = np.arange(len(labels))
        axes[0].bar(x, cbc_df["Entropy(Enc)"], color="#10b981", width=0.55)
        axes[0].axhline(TARGET_IMAGE_METRICS["ENTROPY"], color="#ef4444", linestyle="--", linewidth=1.1, label="Ideal 8.0")
        axes[0].set_ylim(7.90, 8.01)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
        axes[0].set_title(f"{title_prefix} — Entropi CBC")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].bar(x - 0.18, cbc_df["NPCR(%)"], width=0.35, color="#3b82f6", label="NPCR (%)")
        axes[1].bar(x + 0.18, cbc_df["UACI(%)"], width=0.35, color="#f59e0b", label="UACI (%)")
        axes[1].axhline(TARGET_IMAGE_METRICS["NPCR"], color="#1d4ed8", linestyle=":", linewidth=1.1)
        axes[1].axhline(TARGET_IMAGE_METRICS["UACI"], color="#b45309", linestyle=":", linewidth=1.1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
        axes[1].set_title(f"{title_prefix} — NPCR & UACI CBC")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="AES S-box ML & Image Cryptography Lab",
        page_icon="🔐",
        layout="wide",
    )

    st.title("Konstruksi S-box AES dengan Machine Learning & Enkripsi Citra Digital")
    st.caption(
        "Proyek UAS Kriptografi — Hybrid GA-SA di GL(8, 2) (Rujukan: Alamsyah et al., 2025) | "
        "AES-128 Custom Mode CBC & CTR"
    )

    sbox_data = load_json_cached(str(SBOX_RESULTS_PATH))
    if sbox_data is None:
        st.error(
            f"File hasil optimasi tidak ditemukan di `{SBOX_RESULTS_PATH}`. "
            "Jalankan `python scripts/run_experiment.py` terlebih dahulu."
        )
        return

    real_eval_data = load_json_cached(str(REAL_EVAL_RESULTS_PATH))
    sample_eval_data = load_json_cached(str(IMAGE_EVAL_RESULTS_PATH))

    with st.sidebar:
        st.header("Konfigurasi Proyek")
        st.markdown(
            f"- **Galois Field:** `GF(2^8)` (`0x{IRREDUCIBLE_POLY:03X}`)\n"
            f"- **Algoritma:** {sbox_data['metadata'].get('algorithm', 'Hybrid GA-SA')}\n"
            f"- **Locked Seed:** `{sbox_data['metadata'].get('seed', RANDOM_SEED)}`\n"
            f"- **Runtime Optimasi:** `{sbox_data['metadata'].get('runtime_seconds', 0.0):.3f} s`\n"
            f"- **Mode AES Diizinkan:** `{', '.join(ALLOWED_AES_MODES)}` (ECB dilarang)"
        )
        st.divider()
        st.info(
            "Gunakan tab **Enkripsi & Dekripsi Citra Real-Time** untuk mengunggah gambar, "
            "memilih S-box & mode CBC/CTR, dan memverifikasi dekripsi 100% lossless."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Ringkasan & Metrik S-box",
            "2. Tabel S-box 16×16 & Analisis SAC",
            "3. Enkripsi & Dekripsi Citra Real-Time",
            "4. Benchmark Enkripsi Citra",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: Ringkasan & Metrik S-box
    # ------------------------------------------------------------------
    with tab1:
        custom_m = sbox_data["custom_sbox"]["metrics"]
        aes_m = sbox_data["aes_standard"]["metrics"]

        st.subheader("Performa S-box Optimal (Hybrid GA-SA)")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Nonlinearity (NL)", f"{custom_m['nl']}", delta="Target >= 112")
        c2.metric("SAC (Mean)", f"{custom_m['sac']:.5f}", delta=f"{custom_m['sac'] - aes_m['sac']:+.5f} vs AES")
        c3.metric("BIC-NL", f"{custom_m['bic_nl']}", delta="Target >= 112")
        c4.metric("BIC-SAC (Mean)", f"{custom_m['bic_sac']:.5f}", delta=f"{custom_m['bic_sac'] - aes_m['bic_sac']:+.5f} vs AES")
        c5.metric("Skor Komposit (S)", f"{custom_m['s_score']:.6f}", delta=f"{custom_m['s_score'] - aes_m['s_score']:+.6f} vs AES")
        c6.metric(
            "Fixed / Opp. Fixed",
            f"{custom_m['fixed_points']} / {custom_m['opposite_fixed_points']}",
            delta=f"C = {sbox_data['custom_sbox']['affine_constant_hex']}",
        )

        st.subheader("Perbandingan Metrik Kriptografi")
        comp_df = pd.DataFrame(sbox_data["comparison_summary"])
        st.dataframe(comp_df, width="stretch", hide_index=True)

        col_l, col_r = st.columns(2)
        with col_l:
            fig_dev = _plot_sbox_deviation_comparison(sbox_data)
            st.pyplot(fig_dev, width="stretch")
            plt.close(fig_dev)
        with col_r:
            fig_conv = _plot_convergence_history(sbox_data["convergence_history"])
            st.pyplot(fig_conv, width="stretch")
            plt.close(fig_conv)

    # ------------------------------------------------------------------
    # TAB 2: Tabel S-box 16x16 & Analisis SAC
    # ------------------------------------------------------------------
    with tab2:
        sel_col1, sel_col2 = st.columns([2, 1])
        with sel_col1:
            selected_sbox_key = st.selectbox(
                "Pilih S-box untuk Diinspeksi",
                options=list(SBOX_LABELS.keys()),
                format_func=lambda k: SBOX_LABELS[k],
            )
        with sel_col2:
            display_fmt = st.radio("Format Nilai", ["Heksadesimal", "Desimal"], horizontal=True)

        entry = sbox_data[selected_sbox_key]
        metrics_entry = entry["metrics"]

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Bijective (256 Unik)", "Valid" if metrics_entry["bijective"] else "Tidak")
        b2.metric("Balance (128:128)", "Valid" if metrics_entry["balanced"] else "Tidak")
        b3.metric("Konstanta Afin (C)", f"{entry['affine_constant_hex']} ({entry['affine_constant']})")
        b4.metric("Strength Value (SV)", f"{metrics_entry['sv_score']:.5f}")

        grid_sbox = entry["sbox_hex_16x16"] if display_fmt == "Heksadesimal" else entry["sbox_16x16"]
        inv_grid_raw = entry["inverse_sbox_16x16"]
        grid_inv = (
            [[f"{int(v):02X}" for v in row] for row in inv_grid_raw]
            if display_fmt == "Heksadesimal"
            else inv_grid_raw
        )

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Tabel S-box 16×16**")
            st.dataframe(_render_sbox_grid_df(grid_sbox), width="stretch")
        with g2:
            st.markdown("**Tabel Inverse S-box 16×16**")
            st.dataframe(_render_sbox_grid_df(grid_inv), width="stretch")

        dl1, dl2 = st.columns(2)
        csv_bytes = _render_sbox_grid_df(grid_sbox).to_csv().encode("utf-8")
        json_bytes = json.dumps(entry, indent=2).encode("utf-8")
        dl1.download_button(
            "Unduh Tabel S-box (.csv)",
            data=csv_bytes,
            file_name=f"{selected_sbox_key}_16x16.csv",
            mime="text/csv",
            width="stretch",
        )
        dl2.download_button(
            "Unduh Spesifikasi Lengkap (.json)",
            data=json_bytes,
            file_name=f"{selected_sbox_key}.json",
            mime="application/json",
            width="stretch",
        )

        st.divider()
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Matriks Afin Biner $K \\in \\mathrm{GL}(8, 2)$**")
            k_df = pd.DataFrame(
                entry["affine_matrix"],
                index=[f"r{i}" for i in range(8)],
                columns=[f"c{j}" for j in range(8)],
            )
            st.dataframe(k_df, width="stretch")
            st.markdown("**Nonlinearity per Bit Output ($f_0 \\dots f_7$)**")
            nl_df = pd.DataFrame(
                {"Bit": [f"f{i}" for i in range(8)], "NL": metrics_entry["nl_per_bit"], "Ones": metrics_entry["bit_ones_count"]}
            )
            st.dataframe(nl_df, width="stretch", hide_index=True)
        with a2:
            fig_sac = _plot_sac_heatmap(
                metrics_entry["sac_matrix"],
                f"Heatmap Matriks SAC 8×8 — {SBOX_LABELS[selected_sbox_key]}",
            )
            st.pyplot(fig_sac, width="stretch")
            plt.close(fig_sac)

    # ------------------------------------------------------------------
    # TAB 3: Enkripsi & Dekripsi Citra Real-Time
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Pengujian Enkripsi & Dekripsi Citra Digital (AES-128 Custom)")

        source_options = ["Upload Citra (.png / .jpg)"]
        if REAL_SAMPLE_IMAGE_PATH.exists():
            source_options.insert(0, "Citra Nyata Lena (foto_asli.jpg)")
        source_options.extend(["Sampel Grayscale Bawaan", "Sampel RGB Bawaan"])

        ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.0, 1.0])
        with ctrl1:
            img_source = st.selectbox("Sumber Citra", source_options)
            uploaded_file = None
            if img_source == "Upload Citra (.png / .jpg)":
                uploaded_file = st.file_uploader("Unggah file citra", type=["png", "jpg", "jpeg", "bmp"])
        with ctrl2:
            rt_sbox_key = st.selectbox(
                "S-box yang Digunakan",
                options=list(SBOX_LABELS.keys()),
                format_func=lambda k: SBOX_LABELS[k],
                key="rt_sbox",
            )
            rt_mode = st.radio("Mode Operasi AES", list(ALLOWED_AES_MODES), horizontal=True)
        with ctrl3:
            resize_choice = st.selectbox(
                "Batas Resolusi Demo",
                options=["256 px (Disarankan)", "128 px", "512 px", "Ukuran Asli"],
            )
            max_dim_map = {"256 px (Disarankan)": 256, "128 px": 128, "512 px": 512, "Ukuran Asli": None}
            iv_policy = st.radio("IV / Nonce", ["Deterministik (Seed 42)", "Acak (os.urandom)"], horizontal=True)

        kcol1, kcol2 = st.columns([1, 2])
        with kcol1:
            key_mode = st.radio("Format Kunci AES-128", ["Hex (32 karakter)", "Passphrase Teks"], horizontal=True)
        with kcol2:
            default_val = DEFAULT_KEY_HEX if key_mode == "Hex (32 karakter)" else "kriptografi-unnes-2026"
            key_text = st.text_input("Kunci AES-128", value=default_val)

        aes_key, key_warn = _resolve_aes_key(key_mode, key_text)
        if key_warn:
            st.warning(key_warn)

        # Resolve image array
        target_img: np.ndarray | None = None
        if img_source == "Citra Nyata Lena (foto_asli.jpg)":
            target_img = load_image_array(REAL_SAMPLE_IMAGE_PATH)
        elif img_source == "Sampel Grayscale Bawaan":
            target_img = generate_sample_images((256, 256))["grayscale_sample"]
        elif img_source == "Sampel RGB Bawaan":
            target_img = generate_sample_images((256, 256))["rgb_sample"]
        elif uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            target_img = load_image_array(pil_img)

        if target_img is None:
            st.info("Silakan unggah gambar `.png` atau `.jpg` untuk memulai pengujian.")
        else:
            target_img = _resize_if_needed(target_img, max_dim_map[resize_choice])
            sbox_arr = np.asarray(sbox_data[rt_sbox_key]["sbox_flat"], dtype=np.uint8)
            iv_arg = bytes.fromhex(DEFAULT_IV_HEX) if iv_policy == "Deterministik (Seed 42)" else None

            res = evaluate_image_encryption(
                target_img,
                key=aes_key,
                sbox=sbox_arr,
                mode=rt_mode,
                iv_or_nonce=iv_arg,
            )

            if res["is_lossless"]:
                st.success(
                    f"Dekripsi **100% Lossless** terverifikasi (`MSE = {res['mse']:.1f}`, "
                    f"`Max Abs Diff = {res['max_abs_diff']}`, Shape = `{tuple(res['shape'])}`, "
                    f"IV/Nonce = `{res['iv_or_nonce_hex']}`)"
                )
            else:
                st.error("Dekripsi tidak lossless!")

            im1, im2, im3 = st.columns(3)
            with im1:
                st.image(res["original_image"], caption="Citra Asli", width="stretch")
            with im2:
                st.image(res["encrypted_image"], caption=f"Citra Terenkripsi ({rt_mode})", width="stretch")
            with im3:
                st.image(res["decrypted_image"], caption="Citra Hasil Dekripsi", width="stretch")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Entropi Asli", f"{res['entropy_original']:.5f}")
            m2.metric("Entropi Terenkripsi", f"{res['entropy_encrypted']:.5f}", delta="Ideal ~ 8.0")
            m3.metric("NPCR (1-Piksel Diff)", f"{res['npcr']:.4f}%")
            m4.metric("UACI (1-Piksel Diff)", f"{res['uaci']:.4f}%")
            m5.metric(
                "Chi-Square (Mean)",
                f"{res['chi_square_encrypted']['mean']:.1f}",
                delta=f"Asli: {res['chi_square_original']['mean']:.1f}",
            )

            t1, t2, t3 = st.columns(3)
            t1.caption(f"Waktu Enkripsi: `{res['encryption_time_ms']:.2f} ms` | Dekripsi: `{res['decryption_time_ms']:.2f} ms`")
            t2.caption(f"NPCR Plain-vs-Cipher: `{res['npcr_plain_vs_cipher']:.4f}%` | UACI Plain-vs-Cipher: `{res['uaci_plain_vs_cipher']:.4f}%`")
            if rt_mode == "CTR":
                t3.caption("Catatan CTR: tanpa chaining antar-blok, perubahan 1 piksel pada nonce sama hanya mengubah 1 posisi byte (~0% NPCR diferensial).")

            h1, h2 = st.columns(2)
            with h1:
                fig_ho = _plot_histograms(res["histogram_original"], "Histogram Citra Asli")
                st.pyplot(fig_ho, width="stretch")
                plt.close(fig_ho)
            with h2:
                fig_he = _plot_histograms(res["histogram_encrypted"], f"Histogram Citra Terenkripsi ({rt_mode})")
                st.pyplot(fig_he, width="stretch")
                plt.close(fig_he)

            d1, d2 = st.columns(2)
            d1.download_button(
                "Unduh Citra Terenkripsi (.png)",
                data=_array_to_png_bytes(res["encrypted_image"]),
                file_name=f"encrypted_{rt_sbox_key}_{rt_mode.lower()}.png",
                mime="image/png",
                width="stretch",
            )
            d2.download_button(
                "Unduh Citra Hasil Dekripsi (.png)",
                data=_array_to_png_bytes(res["decrypted_image"]),
                file_name=f"decrypted_{rt_sbox_key}_{rt_mode.lower()}.png",
                mime="image/png",
                width="stretch",
            )

    # ------------------------------------------------------------------
    # TAB 4: Benchmark Enkripsi Citra
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Benchmark Performa Enkripsi Citra (AES Standar vs S-box44 vs Custom GA-SA)")

        if real_eval_data is not None:
            st.markdown("#### 1. Hasil Uji Citra Nyata (`data/samples/foto_asli.jpg` — Lena 258×258×3)")
            st.dataframe(pd.DataFrame(real_eval_data["summary_table"]), width="stretch", hide_index=True)
            fig_real = _plot_image_benchmark(real_eval_data["summary_table"], "Lena")
            st.pyplot(fig_real, width="stretch")
            plt.close(fig_real)
        else:
            st.info("Belum ada data `data/image_eval_real.json`. Jalankan `python scripts/run_image_eval.py --image data/samples/foto_asli.jpg --output data/image_eval_real.json`.")

        if sample_eval_data is not None:
            st.markdown("#### 2. Hasil Uji Sampel Terstruktur (`grayscale_sample` & `rgb_sample` 128×128)")
            st.dataframe(pd.DataFrame(sample_eval_data["summary_table"]), width="stretch", hide_index=True)
            fig_sample = _plot_image_benchmark(sample_eval_data["summary_table"], "Sampel 128×128")
            st.pyplot(fig_sample, width="stretch")
            plt.close(fig_sample)


if __name__ == "__main__":
    main()
