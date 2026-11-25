"""Unit tests for Phase 3: Custom AES-128 (CBC/CTR) and digital image evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aes_custom import AESCustom, pad_pkcs7, unpad_pkcs7
from src.image_eval import (
    compute_histogram,
    evaluate_image_encryption,
    generate_sample_images,
    load_sbox_from_results,
)


class TestAESCustom(unittest.TestCase):
    """Tests for AES-128 block cipher, FIPS-197 vector, and CBC/CTR modes."""

    def test_fips197_appendix_b_vector_single_and_batch(self) -> None:
        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
        plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
        expected_ct = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")

        cipher = AESCustom(key=key)  # default standard AES S-box
        ct = cipher.encrypt_block(plaintext)
        self.assertEqual(ct, expected_ct)
        self.assertEqual(cipher.decrypt_block(ct), plaintext)

        batch_pt = np.frombuffer(plaintext, dtype=np.uint8).reshape(1, 16)
        batch_ct = cipher._encrypt_blocks_batch(batch_pt)
        self.assertEqual(batch_ct.tobytes(), expected_ct)
        self.assertEqual(cipher._decrypt_blocks_batch(batch_ct).tobytes(), plaintext)

    def test_pkcs7_padding_roundtrip_and_validation(self) -> None:
        for length in (0, 1, 15, 16, 17, 31, 32):
            data = bytes((i % 256) for i in range(length))
            padded = pad_pkcs7(data, 16)
            self.assertEqual(len(padded) % 16, 0)
            self.assertGreater(len(padded), len(data))
            self.assertEqual(unpad_pkcs7(padded, 16), data)

        with self.assertRaises(ValueError):
            unpad_pkcs7(b"", 16)
        with self.assertRaises(ValueError):
            unpad_pkcs7(b"\x00" * 16, 16)
        with self.assertRaises(ValueError):
            unpad_pkcs7(b"A" * 14 + b"\x03\x02", 16)

    def test_custom_sbox_cbc_and_ctr_roundtrips(self) -> None:
        key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
        custom_sbox = load_sbox_from_results("custom_sbox")

        std_cipher = AESCustom(key=key)
        custom_cipher = AESCustom(key=key, sbox=custom_sbox)

        iv = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
        nonce8 = bytes.fromhex("f0f1f2f3f4f5f6f7")

        for length in (0, 1, 15, 16, 17, 64, 513):
            msg = bytes(((i * 7 + 3) % 256) for i in range(length))

            ct_cbc, used_iv = custom_cipher.encrypt(msg, mode="CBC", iv_or_nonce=iv)
            self.assertEqual(used_iv, iv)
            self.assertEqual(custom_cipher.decrypt(ct_cbc, used_iv, mode="CBC"), msg)

            ct_ctr, used_nonce = custom_cipher.encrypt(msg, mode="CTR", iv_or_nonce=nonce8)
            self.assertEqual(used_nonce, nonce8)
            self.assertEqual(len(ct_ctr), len(msg))
            self.assertEqual(custom_cipher.decrypt(ct_ctr, used_nonce, mode="CTR"), msg)

        sample = b"Custom S-box AES-128 Verification Message!"
        ct_std, _ = std_cipher.encrypt(sample, mode="CBC", iv_or_nonce=iv)
        ct_cus, _ = custom_cipher.encrypt(sample, mode="CBC", iv_or_nonce=iv)
        self.assertNotEqual(ct_std, ct_cus)

    def test_rejects_ecb_invalid_key_and_non_bijective_sbox(self) -> None:
        key = b"\x01" * 16
        cipher = AESCustom(key=key)

        with self.assertRaises(ValueError):
            cipher.encrypt(b"hello world 1234", mode="ECB")
        with self.assertRaises(ValueError):
            cipher.decrypt(b"0" * 16, iv_or_nonce=b"\x00" * 16, mode="ECB")
        with self.assertRaises(ValueError):
            AESCustom(key=b"\x01" * 15)
        with self.assertRaises(ValueError):
            AESCustom(key=key, sbox=np.zeros(256, dtype=np.uint8))


class TestImageEvaluation(unittest.TestCase):
    """Tests for lossless image encryption/decryption and image security metrics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.custom_sbox = load_sbox_from_results("custom_sbox")
        cls.key = bytes.fromhex("00112233445566778899aabbccddeeff")
        cls.iv = bytes.fromhex("a0a1a2a3a4a5a6a7a8a9aaabacadaeaf")
        cls.samples = generate_sample_images(size=(128, 128))

    def test_cbc_lossless_and_metrics_grayscale_and_rgb(self) -> None:
        for name, img in self.samples.items():
            with self.subTest(image=name, mode="CBC"):
                res = evaluate_image_encryption(
                    img,
                    key=self.key,
                    sbox=self.custom_sbox,
                    mode="CBC",
                    iv_or_nonce=self.iv,
                )
                self.assertTrue(res["is_lossless"])
                self.assertEqual(res["mse"], 0.0)
                self.assertEqual(res["max_abs_diff"], 0)
                self.assertGreater(res["entropy_encrypted"], 7.95)
                self.assertGreaterEqual(res["npcr"], 99.0)
                self.assertLessEqual(res["npcr"], 100.0)
                self.assertGreaterEqual(res["uaci"], 32.0)
                self.assertLessEqual(res["uaci"], 35.0)
                self.assertLess(
                    res["chi_square_encrypted"]["mean"],
                    res["chi_square_original"]["mean"],
                )

                hist = compute_histogram(res["encrypted_image"])
                expected_pixels = img.shape[0] * img.shape[1]
                for bins in hist.values():
                    self.assertEqual(sum(bins), expected_pixels)

    def test_ctr_lossless_and_entropy_grayscale_and_rgb(self) -> None:
        for name, img in self.samples.items():
            with self.subTest(image=name, mode="CTR"):
                res = evaluate_image_encryption(
                    img,
                    key=self.key,
                    sbox=self.custom_sbox,
                    mode="CTR",
                    iv_or_nonce=self.iv,
                )
                self.assertTrue(res["is_lossless"])
                self.assertEqual(res["mse"], 0.0)
                self.assertEqual(res["max_abs_diff"], 0)
                self.assertGreater(res["entropy_encrypted"], 7.95)


if __name__ == "__main__":
    unittest.main()
