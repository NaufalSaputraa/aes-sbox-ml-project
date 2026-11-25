"""Interactive 5-tab Streamlit dashboard for Dual-Track AES S-box ML & image/text cryptography."""

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
    evaluate_text_encryption,
    generate_sample_images,
    load_image_array,
)

SBOX_LABELS: dict[str, str] = {
    "custom_sbox": "Custom GA-SA (Aljabar GL(8,2))",
    "non_algebraic_sbox": "Non-Algebraic WHS-SA (Tanpa Aljabar/Polinomial)",
    "aes_standard": "AES Standar (FIPS-197)",
    "benchmark_sbox44": "S-box44 (Alamsyah et al., 2025)",
    "random_permutation": "Permutasi Acak Awal (Baseline Non-Aljabar)",
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


def _plot_track_b_convergence(history: list[dict[str, Any]]) -> plt.Figure:
    """Line chart of Track B Hybrid GA-SA convergence (S-score drop to 0.0)."""
    iters = [h["iteration"] for h in history]
    best_s = [h["best_s_score"] for h in history]
    phases = [h["phase"] for h in history]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(iters, best_s, color="#0ea5e9", linewidth=2.0, label="Best S Score")

    if "SA" in phases:
        sa_start_iter = iters[phases.index("SA")]
        ax.axvline(sa_start_iter, color="#f59e0b", linestyle="--", linewidth=1.3, label="Transisi GA -> SA")

    ax.set_xlabel("Iterasi (Generasi GA / Langkah SA)")
    ax.set_ylabel("Best Skor Komposit (S)")
    ax.set_title("Track B: Konvergensi Hybrid GA-SA (Ruang GL(8, 2))")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_track_a_convergence(history: list[dict[str, Any]], init_nl: int) -> plt.Figure:
    """Line chart of Track A Non-Algebraic WHS-SA NL climb from random permutation."""
    iters = [0] + [h["iteration"] for h in history]
    coord_nl = [init_nl] + [h["best_coord_nl"] for h in history]
    all_nl = [init_nl] + [h["best_nl_all"] for h in history]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(iters, coord_nl, color="#10b981", linewidth=2.0, label="Best Coord NL (min f0..f7)")
    ax.plot(iters, all_nl, color="#8b5cf6", linewidth=1.6, linestyle="--", label="Best All-Mask NL (255 mask)")
    ax.set_xlabel("Langkah Simulated Annealing (Swap)")
    ax.set_ylabel("Nonlinearity (NL)")
    ax.set_title("Track A: Konvergensi Non-Aljabar WHS-SA (Tanpa GF(2^8))")
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
        axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=7.2)
        axes[0].set_title(f"{title_prefix} — Entropi CBC")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].bar(x - 0.18, cbc_df["NPCR(%)"], width=0.35, color="#3b82f6", label="NPCR (%)")
        axes[1].bar(x + 0.18, cbc_df["UACI(%)"], width=0.35, color="#f59e0b", label="UACI (%)")
        axes[1].axhline(TARGET_IMAGE_METRICS["NPCR"], color="#1d4ed8", linestyle=":", linewidth=1.1)
        axes[1].axhline(TARGET_IMAGE_METRICS["UACI"], color="#b45309", linestyle=":", linewidth=1.1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=7.2)
        axes[1].set_title(f"{title_prefix} — NPCR & UACI CBC")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Dual-Track AES S-box ML & Image Cryptography Lab",
        page_icon="🔐",
        layout="wide",
    )

    st.title("Konstruksi S-box AES Dual-Track (Non-Aljabar & Hybrid GA-SA) + Evaluasi Citra")
    st.caption(
        "Track A: Optimasi Permutasi Non-Aljabar Murni (WHS-SA, tanpa GF(2^8)) | "
        "Track B: Hybrid GA-SA di GL(8, 2) (Alamsyah et al., 2025) | AES-128 CBC & CTR"
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
    available_sboxes = [k for k in SBOX_LABELS if k in sbox_data]

    with st.sidebar:
        st.header("Konfigurasi Dual-Track")
        st.markdown(
            f"- **Track A (Non-Aljabar):** WHS-SA di ruang permutasi $S_{{256}}$ (tanpa polinomial/matriks afin)\n"
            f"- **Track B (Aljabar):** Hybrid GA-SA di $\\mathrm{{GL}}(8,2)$ (`0x{IRREDUCIBLE_POLY:03X}`)\n"
            f"- **Locked Seed:** `{sbox_data['metadata'].get('seed', RANDOM_SEED)}`\n"
            f"- **Runtime Track A / B:** `{sbox_data['metadata'].get('non_algebraic_runtime_seconds', 0.0):.2f}s` / "
            f"`{sbox_data['metadata'].get('runtime_seconds', 0.0):.2f}s`\n"
            f"- **Mode AES Diizinkan:** `{', '.join(ALLOWED_AES_MODES)}` (ECB dilarang)"
        )
        st.divider()
        st.info(
            "Dual-Track membandingkan konstruksi S-box **murni tanpa aljabar** (belajar dari nol) "
            "melawan **Hybrid GA-SA** berbasis transformasi afin."
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Ringkasan & Dual-Track",
            "2. Tabel S-box 16×16 & Audit",
            "3. Enkripsi Citra Real-Time",
            "4. Demo Teks & Bit Avalanche",
            "5. Benchmark Citra",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: Ringkasan & Dual-Track Optimasi
    # ------------------------------------------------------------------
    with tab1:
        custom_m = sbox_data["custom_sbox"]["metrics"]
        non_alg_m = sbox_data.get("non_algebraic_sbox", {}).get("metrics")
        rand_perm_m = sbox_data.get("random_permutation", {}).get("metrics")
        rand_bench = sbox_data.get("random_search_benchmark")

        st.subheader("Ringkasan Dual-Track Optimasi S-box")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Track A — Non-Aljabar Murni (WHS-SA)")
            if non_alg_m and rand_perm_m:
                a1, a2, a3 = st.columns(3)
                a1.metric("NL (min)", f"{non_alg_m['nl']}", delta=f"+{non_alg_m['nl'] - rand_perm_m['nl']} dari Acak ({rand_perm_m['nl']})")
                a2.metric("BIC-NL", f"{non_alg_m['bic_nl']}", delta=f"+{non_alg_m['bic_nl'] - rand_perm_m['bic_nl']} dari Acak ({rand_perm_m['bic_nl']})")
                a3.metric("LAP", f"{non_alg_m['lap']:.5f}", delta=f"{non_alg_m['lap'] - rand_perm_m['lap']:+.5f} vs Acak")
                st.caption("Dibangun dari permutasi acak murni 0..255 tanpa polinomial GF(2^8) maupun matriks afin.")
        with col_b:
            st.markdown("#### Track B — Aljabar Hybrid GA-SA (GL(8, 2))")
            b1, b2, b3 = st.columns(3)
            b1.metric("NL / BIC-NL", f"{custom_m['nl']} / {custom_m['bic_nl']}", delta="Target >= 112")
            b2.metric("Skor Komposit (S)", f"{custom_m['s_score']:.6f}", delta="SAC=0.5, BIC-SAC=0.5")
            b3.metric("Fixed / Opp. Fixed", f"{custom_m['fixed_points']} / {custom_m['opposite_fixed_points']}", delta=f"C = {sbox_data['custom_sbox']['affine_constant_hex']}")
            if rand_bench:
                st.caption(
                    f"Fair-Budget Benchmark ({rand_bench['budget']} evals): Hybrid GA-SA S={custom_m['s_score']:.6f} "
                    f"vs Random GL(8,2) Mean S={rand_bench['mean_s_score']:.6f}."
                )

        st.subheader("Tabel Komparasi 5 S-box Lengkap")
        st.dataframe(pd.DataFrame(sbox_data["comparison_summary"]), width="stretch", hide_index=True)

        c_left, c_right = st.columns(2)
        with c_left:
            if "non_algebraic_convergence_history" in sbox_data and rand_perm_m:
                fig_a = _plot_track_a_convergence(
                    sbox_data["non_algebraic_convergence_history"],
                    init_nl=int(rand_perm_m["nl"]),
                )
                st.pyplot(fig_a, width="stretch")
                plt.close(fig_a)
        with c_right:
            fig_b = _plot_track_b_convergence(sbox_data["convergence_history"])
            st.pyplot(fig_b, width="stretch")
            plt.close(fig_b)

    # ------------------------------------------------------------------
    # TAB 2: Tabel S-box 16x16 & Audit Kriptografi Lengkap
    # ------------------------------------------------------------------
    with tab2:
        sel_col1, sel_col2 = st.columns([2, 1])
        with sel_col1:
            selected_sbox_key = st.selectbox(
                "Pilih S-box untuk Diinspeksi",
                options=available_sboxes,
                format_func=lambda k: SBOX_LABELS[k],
            )
        with sel_col2:
            display_fmt = st.radio("Format Nilai", ["Heksadesimal", "Desimal"], horizontal=True)

        entry = sbox_data[selected_sbox_key]
        metrics_entry = entry["metrics"]

        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Bijective / Balance", "Valid" if metrics_entry["valid"] else "Tidak")
        b2.metric("NL / BIC-NL", f"{metrics_entry['nl']} / {metrics_entry['bic_nl']}")
        b3.metric("Diff. Uniformity (DU)", f"{metrics_entry.get('du', '-')}")
        b4.metric("Alg. Degree (min/max)", f"{metrics_entry.get('algebraic_degree_min', '-')}/{metrics_entry.get('algebraic_degree_max', '-')}")
        b5.metric("Cycles (Count)", f"{metrics_entry.get('num_cycles', '-')}")
        b6.metric("Konstanta Afin (C)", f"{entry['affine_constant_hex']}")

        st.caption(f"Struktur Panjang Siklus: `{metrics_entry.get('cycles', [])}` | Fixed / Opp. Fixed: `{metrics_entry['fixed_points']} / {metrics_entry['opposite_fixed_points']}`")

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
        dl1.download_button(
            "Unduh Tabel S-box (.csv)",
            data=_render_sbox_grid_df(grid_sbox).to_csv().encode("utf-8"),
            file_name=f"{selected_sbox_key}_16x16.csv",
            mime="text/csv",
            width="stretch",
        )
        dl2.download_button(
            "Unduh Spesifikasi Lengkap (.json)",
            data=json.dumps(entry, indent=2).encode("utf-8"),
            file_name=f"{selected_sbox_key}.json",
            mime="application/json",
            width="stretch",
        )

        st.divider()
        a1, a2 = st.columns(2)
        with a1:
            if entry.get("affine_matrix") is not None:
                st.markdown("**Matriks Afin Biner $K \\in \\mathrm{GL}(8, 2)$**")
                k_df = pd.DataFrame(
                    entry["affine_matrix"],
                    index=[f"r{i}" for i in range(8)],
                    columns=[f"c{j}" for j in range(8)],
                )
                st.dataframe(k_df, width="stretch")
            else:
                st.info("**Konstruksi Non-Aljabar Murni:** S-box ini tidak menggunakan matriks afin $K$ maupun polinomial $\\mathrm{GF}(2^8)$.")

            st.markdown("**Nonlinearity & Algebraic Degree per Bit ($f_0 \\dots f_7$)**")
            nl_df = pd.DataFrame(
                {
                    "Bit": [f"f{i}" for i in range(8)],
                    "NL": metrics_entry["nl_per_bit"],
                    "Degree": metrics_entry.get("algebraic_degrees", ["-"] * 8),
                    "Ones": metrics_entry["bit_ones_count"],
                }
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
                options=available_sboxes,
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

        target_img: np.ndarray | None = None
        if img_source == "Citra Nyata Lena (foto_asli.jpg)":
            target_img = load_image_array(REAL_SAMPLE_IMAGE_PATH)
        elif img_source == "Sampel Grayscale Bawaan":
            target_img = generate_sample_images((256, 256))["grayscale_sample"]
        elif img_source == "Sampel RGB Bawaan":
            target_img = generate_sample_images((256, 256))["rgb_sample"]
        elif uploaded_file is not None:
            target_img = load_image_array(Image.open(uploaded_file))

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

            im1, im2, im3 = st.columns(3)
            im1.image(res["original_image"], caption="Citra Asli", width="stretch")
            im2.image(res["encrypted_image"], caption=f"Citra Terenkripsi ({rt_mode})", width="stretch")
            im3.image(res["decrypted_image"], caption="Citra Hasil Dekripsi", width="stretch")

            wu = res["wu2011_test"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Entropi Asli -> Enkripsi", f"{res['entropy_encrypted']:.5f}", delta=f"Asli: {res['entropy_original']:.5f}")
            m2.metric("NPCR (1-Piksel Diff)", f"{res['npcr']:.4f}%", delta=f"Wu2011: {'PASS' if wu['npcr_pass'] else 'FAIL'} (>={wu['npcr_critical']:.2f}%)")
            m3.metric("UACI (1-Piksel Diff)", f"{res['uaci']:.4f}%", delta=f"Wu2011: {'PASS' if wu['uaci_pass'] else 'FAIL'} ([{wu['uaci_lower']:.2f}, {wu['uaci_upper']:.2f}])")
            m4.metric("Chi-Square (Mean)", f"{res['chi_square_encrypted']['mean']:.1f}", delta=f"Asli: {res['chi_square_original']['mean']:.1f}")
            m5.metric("Enc / Dec (ms)", f"{res['encryption_time_ms']:.1f} / {res['decryption_time_ms']:.1f}")

            co, ce = res["correlation_original"], res["correlation_encrypted"]
            corr_df = pd.DataFrame(
                [
                    {"Arah": "Horizontal", "Citra Asli": f"{co['horizontal']:+.5f}", "Citra Terenkripsi": f"{ce['horizontal']:+.5f}"},
                    {"Arah": "Vertikal", "Citra Asli": f"{co['vertical']:+.5f}", "Citra Terenkripsi": f"{ce['vertical']:+.5f}"},
                    {"Arah": "Diagonal", "Citra Asli": f"{co['diagonal']:+.5f}", "Citra Terenkripsi": f"{ce['diagonal']:+.5f}"},
                ]
            )
            st.markdown("**Korelasi Piksel Bertetangga (Ideal Terenkripsi mendekati 0.0)**")
            st.dataframe(corr_df, width="stretch", hide_index=True)

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
    # TAB 4: Demo Enkripsi Teks & Bit Avalanche
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Demo Enkripsi & Dekripsi Teks — AES-128 Custom (CBC / CTR)")

        tc1, tc2, tc3 = st.columns([1.4, 1.0, 1.0])
        with tc1:
            txt_sbox_key = st.selectbox(
                "Pilih S-box",
                options=available_sboxes,
                format_func=lambda k: SBOX_LABELS[k],
                key="txt_sbox",
            )
        with tc2:
            txt_mode = st.radio("Mode AES", list(ALLOWED_AES_MODES), horizontal=True, key="txt_mode")
        with tc3:
            txt_pass = st.text_input("Passphrase Kunci (16-byte SHA-256)", value="kriptografi-unnes-2026", key="txt_pass")

        sample_plain = st.text_area(
            "Plaintext Uji",
            value="Pengujian AES-128 dengan S-box hasil optimasi Dual-Track (Non-Aljabar & Hybrid GA-SA)!",
            height=90,
        )

        txt_key = hashlib.sha256(txt_pass.encode("utf-8")).digest()[:16]
        txt_sbox_arr = np.asarray(sbox_data[txt_sbox_key]["sbox_flat"], dtype=np.uint8)
        txt_res = evaluate_text_encryption(
            sample_plain,
            key=txt_key,
            sbox=txt_sbox_arr,
            mode=txt_mode,
            iv_or_nonce=bytes.fromhex(DEFAULT_IV_HEX),
        )

        tm1, tm2, tm3, tm4 = st.columns(4)
        tm1.metric("Dekripsi Eksak", "100% Cocok" if txt_res["is_exact_match"] else "Gagal")
        tm2.metric("Bit Avalanche (1-bit Flip)", f"{txt_res['bit_avalanche_pct']:.2f}%", delta=f"{txt_res['differing_bits']}/{txt_res['total_bits']} bit")
        tm3.metric("Waktu Enkripsi", f"{txt_res['encryption_time_ms']:.3f} ms")
        tm4.metric("Waktu Dekripsi", f"{txt_res['decryption_time_ms']:.3f} ms")

        st.text_area("Ciphertext (Hex)", value=txt_res["ciphertext_hex"], height=90, disabled=True)
        st.text_area("Hasil Dekripsi", value=txt_res["decrypted_text"], height=90, disabled=True)

    # ------------------------------------------------------------------
    # TAB 5: Benchmark Enkripsi Citra
    # ------------------------------------------------------------------
    with tab5:
        st.subheader("Benchmark Performa Enkripsi Citra (Termasuk Non-Algebraic & Custom GA-SA)")

        if real_eval_data is not None:
            st.markdown("#### 1. Hasil Uji Citra Nyata (`data/samples/foto_asli.jpg` — Lena 258×258×3)")
            st.dataframe(pd.DataFrame(real_eval_data["summary_table"]), width="stretch", hide_index=True)
            fig_real = _plot_image_benchmark(real_eval_data["summary_table"], "Lena")
            st.pyplot(fig_real, width="stretch")
            plt.close(fig_real)

        if sample_eval_data is not None:
            st.markdown("#### 2. Hasil Uji Sampel Terstruktur (`grayscale_sample` & `rgb_sample` 128×128)")
            st.dataframe(pd.DataFrame(sample_eval_data["summary_table"]), width="stretch", hide_index=True)
            fig_sample = _plot_image_benchmark(sample_eval_data["summary_table"], "Sampel 128×128")
            st.pyplot(fig_sample, width="stretch")
            plt.close(fig_sample)


if __name__ == "__main__":
    main()
