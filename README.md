# AES S-box Construction with Machine Learning & Image Encryption

Proyek UAS Kriptografi (Teknik Informatika UNNES) untuk merancang, mengoptimasi, dan mengevaluasi S-box AES 8-bit ($16 \times 16$) menggunakan **Hybrid Genetic Algorithm – Simulated Annealing (GA-SA)** di ruang matriks afin $\mathrm{GL}(8, 2)$, mengintegrasikannya ke mesin **AES-128 custom** (mode **CBC** & **CTR**), serta menguji enkripsi citra digital secara *lossless*.

## Struktur Direktori

```text
aes-sbox-ml-project/
├── app.py                      # Dashboard interaktif Streamlit (Fase 4)
├── config.py                   # Konstanta GF(2^8), target metrik, seed, & hyperparameter
├── requirements.txt            # Dependensi Python
├── data/
│   ├── sbox_results.json       # Hasil S-box AES standar, S-box44, Custom GA-SA, & konvergensi
│   ├── image_eval_results.json # Hasil evaluasi citra sampel grayscale & RGB (128x128)
│   ├── image_eval_real.json    # Hasil evaluasi citra nyata Lena (foto_asli.jpg)
│   └── samples/                # Citra uji (foto_asli.jpg)
├── src/
│   ├── gf_math.py              # Aritmetika GF(2^8), invers perkalian, & transformasi afin
│   ├── sbox_metrics.py         # Validator bijektif/balance & evaluator NL, SAC, BIC, LAP, DAP
│   ├── sbox_optimizer.py       # Optimizer Hybrid GA-SA di GL(8, 2)
│   ├── aes_custom.py           # Mesin AES-128 custom (mode CBC & CTR, tanpa ECB)
│   └── image_eval.py           # Enkripsi/dekripsi citra lossless & Entropi, NPCR, UACI, Histogram
├── scripts/
│   ├── run_experiment.py       # CLI Fase 1 & 2 (optimasi S-box -> data/sbox_results.json)
│   └── run_image_eval.py       # CLI Fase 3 (evaluasi citra -> data/image_eval_*.json)
├── tests/
│   ├── test_sbox_project.py    # Unit test Fase 1 & 2
│   └── test_phase3_aes_image.py# Unit test Fase 3 (FIPS-197, CBC/CTR, lossless citra)
└── report/
    └── laporan_uas.md          # Laporan akhir proyek (Fase 1–4)
```

## Cara Menjalankan

Pastikan berada di direktori `aes-sbox-ml-project`:

```powershell
# 1. Install dependensi
python -m pip install -r requirements.txt

# 2. Jalankan seluruh unit test (Fase 1, 2, dan 3)
python -m unittest discover -s tests -v

# 3. Jalankan optimasi S-box (Fase 1 & 2)
python scripts/run_experiment.py

# 4. Jalankan evaluasi enkripsi citra (Fase 3)
python scripts/run_image_eval.py
python scripts/run_image_eval.py --image data/samples/foto_asli.jpg --output data/image_eval_real.json

# 5. Jalankan dashboard interaktif Streamlit (Fase 4)
python -m streamlit run app.py
```
