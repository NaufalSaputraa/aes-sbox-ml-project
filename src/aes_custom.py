"""Custom AES-128 block cipher supporting arbitrary bijective 8-bit S-boxes (CBC & CTR)."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from config import AES_BLOCK_SIZE, AES_KEY_SIZE, AES_ROUNDS, ALLOWED_AES_MODES, IRREDUCIBLE_POLY
from src.gf_math import build_aes_sbox, build_inverse_sbox, gf_mul

_RCON: tuple[int, ...] = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)

# Precompute GF(2^8) multiplication lookup tables for MixColumns / InvMixColumns
_MUL2_TUPLE: tuple[int, ...] = tuple(gf_mul(x, 0x02, IRREDUCIBLE_POLY) for x in range(256))
_MUL3_TUPLE: tuple[int, ...] = tuple(gf_mul(x, 0x03, IRREDUCIBLE_POLY) for x in range(256))
_MUL9_TUPLE: tuple[int, ...] = tuple(gf_mul(x, 0x09, IRREDUCIBLE_POLY) for x in range(256))
_MULB_TUPLE: tuple[int, ...] = tuple(gf_mul(x, 0x0B, IRREDUCIBLE_POLY) for x in range(256))
_MULD_TUPLE: tuple[int, ...] = tuple(gf_mul(x, 0x0D, IRREDUCIBLE_POLY) for x in range(256))
_MULE_TUPLE: tuple[int, ...] = tuple(gf_mul(x, 0x0E, IRREDUCIBLE_POLY) for x in range(256))

_MUL2_NP: np.ndarray = np.array(_MUL2_TUPLE, dtype=np.uint8)
_MUL3_NP: np.ndarray = np.array(_MUL3_TUPLE, dtype=np.uint8)
_MUL9_NP: np.ndarray = np.array(_MUL9_TUPLE, dtype=np.uint8)
_MULB_NP: np.ndarray = np.array(_MULB_TUPLE, dtype=np.uint8)
_MULD_NP: np.ndarray = np.array(_MULD_TUPLE, dtype=np.uint8)
_MULE_NP: np.ndarray = np.array(_MULE_TUPLE, dtype=np.uint8)

_SHIFT_ROWS_IDX: np.ndarray = np.array(
    [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11], dtype=np.int64
)
_INV_SHIFT_ROWS_IDX: np.ndarray = np.array(
    [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3], dtype=np.int64
)


def pad_pkcs7(data: bytes, block_size: int = AES_BLOCK_SIZE) -> bytes:
    """Apply PKCS#7 padding to `data` for the given block size."""
    if block_size <= 0 or block_size > 255:
        raise ValueError(f"Invalid block_size={block_size}; must be in 1..255.")
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len


def unpad_pkcs7(data: bytes, block_size: int = AES_BLOCK_SIZE) -> bytes:
    """Validate and remove PKCS#7 padding from `data`."""
    if not data or len(data) % block_size != 0:
        raise ValueError("Padded data must be non-empty and a multiple of block_size.")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > block_size:
        raise ValueError(f"Invalid PKCS#7 padding byte: {pad_len}.")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("Corrupted PKCS#7 padding bytes.")
    return data[:-pad_len]


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().upper()
    if normalized not in ALLOWED_AES_MODES:
        raise ValueError(
            f"Unsupported AES mode {mode!r}. Allowed modes are {ALLOWED_AES_MODES} (ECB is forbidden)."
        )
    return normalized


class AESCustom:
    """AES-128 cipher engine parameterized by a custom bijective 8-bit S-box."""

    def __init__(self, key: bytes, sbox: Sequence[int] | np.ndarray | None = None) -> None:
        if not isinstance(key, (bytes, bytearray)) or len(key) != AES_KEY_SIZE:
            raise ValueError(f"AES-128 key must be exactly {AES_KEY_SIZE} bytes.")

        if sbox is None:
            sbox_arr = build_aes_sbox()
        else:
            sbox_arr = np.asarray(sbox, dtype=np.uint8)

        inv_sbox_arr = build_inverse_sbox(sbox_arr)

        self.key: bytes = bytes(key)
        self.sbox: np.ndarray = sbox_arr.copy()
        self.inv_sbox: np.ndarray = inv_sbox_arr.copy()
        self._sbox_tuple: tuple[int, ...] = tuple(int(v) for v in self.sbox)
        self._inv_sbox_tuple: tuple[int, ...] = tuple(int(v) for v in self.inv_sbox)

        self._round_keys_np: np.ndarray = self._expand_key(self.key)
        self._round_keys_tuple: tuple[tuple[int, ...], ...] = tuple(
            tuple(int(v) for v in rk) for rk in self._round_keys_np
        )

    def _expand_key(self, key: bytes) -> np.ndarray:
        """Expand a 16-byte AES-128 key into 11 round keys of 16 bytes each."""
        sbox = self._sbox_tuple
        words: list[tuple[int, int, int, int]] = [
            (key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]) for i in range(4)
        ]

        for i in range(4, 4 * (AES_ROUNDS + 1)):
            t0, t1, t2, t3 = words[i - 1]
            if i % 4 == 0:
                t0, t1, t2, t3 = (
                    sbox[t1] ^ _RCON[(i // 4) - 1],
                    sbox[t2],
                    sbox[t3],
                    sbox[t0],
                )
            p0, p1, p2, p3 = words[i - 4]
            words.append((p0 ^ t0, p1 ^ t1, p2 ^ t2, p3 ^ t3))

        round_keys = np.empty((AES_ROUNDS + 1, AES_BLOCK_SIZE), dtype=np.uint8)
        for r in range(AES_ROUNDS + 1):
            base = 4 * r
            rk_bytes = words[base] + words[base + 1] + words[base + 2] + words[base + 3]
            round_keys[r] = rk_bytes
        return round_keys

    def encrypt_block(self, block: bytes) -> bytes:
        """Encrypt a single 16-byte block using AES-128 with the configured S-box."""
        if len(block) != AES_BLOCK_SIZE:
            raise ValueError(f"Block must be exactly {AES_BLOCK_SIZE} bytes.")

        sbox = self._sbox_tuple
        mul2 = _MUL2_TUPLE
        mul3 = _MUL3_TUPLE
        rks = self._round_keys_tuple

        rk = rks[0]
        b0 = block[0] ^ rk[0]
        b1 = block[1] ^ rk[1]
        b2 = block[2] ^ rk[2]
        b3 = block[3] ^ rk[3]
        b4 = block[4] ^ rk[4]
        b5 = block[5] ^ rk[5]
        b6 = block[6] ^ rk[6]
        b7 = block[7] ^ rk[7]
        b8 = block[8] ^ rk[8]
        b9 = block[9] ^ rk[9]
        b10 = block[10] ^ rk[10]
        b11 = block[11] ^ rk[11]
        b12 = block[12] ^ rk[12]
        b13 = block[13] ^ rk[13]
        b14 = block[14] ^ rk[14]
        b15 = block[15] ^ rk[15]

        for r in range(1, AES_ROUNDS):
            # SubBytes + ShiftRows
            c0, c1, c2, c3 = sbox[b0], sbox[b5], sbox[b10], sbox[b15]
            c4, c5, c6, c7 = sbox[b4], sbox[b9], sbox[b14], sbox[b3]
            c8, c9, c10, c11 = sbox[b8], sbox[b13], sbox[b2], sbox[b7]
            c12, c13, c14, c15 = sbox[b12], sbox[b1], sbox[b6], sbox[b11]

            # MixColumns + AddRoundKey
            rk = rks[r]
            b0 = mul2[c0] ^ mul3[c1] ^ c2 ^ c3 ^ rk[0]
            b1 = c0 ^ mul2[c1] ^ mul3[c2] ^ c3 ^ rk[1]
            b2 = c0 ^ c1 ^ mul2[c2] ^ mul3[c3] ^ rk[2]
            b3 = mul3[c0] ^ c1 ^ c2 ^ mul2[c3] ^ rk[3]

            b4 = mul2[c4] ^ mul3[c5] ^ c6 ^ c7 ^ rk[4]
            b5 = c4 ^ mul2[c5] ^ mul3[c6] ^ c7 ^ rk[5]
            b6 = c4 ^ c5 ^ mul2[c6] ^ mul3[c7] ^ rk[6]
            b7 = mul3[c4] ^ c5 ^ c6 ^ mul2[c7] ^ rk[7]

            b8 = mul2[c8] ^ mul3[c9] ^ c10 ^ c11 ^ rk[8]
            b9 = c8 ^ mul2[c9] ^ mul3[c10] ^ c11 ^ rk[9]
            b10 = c8 ^ c9 ^ mul2[c10] ^ mul3[c11] ^ rk[10]
            b11 = mul3[c8] ^ c9 ^ c10 ^ mul2[c11] ^ rk[11]

            b12 = mul2[c12] ^ mul3[c13] ^ c14 ^ c15 ^ rk[12]
            b13 = c12 ^ mul2[c13] ^ mul3[c14] ^ c15 ^ rk[13]
            b14 = c12 ^ c13 ^ mul2[c14] ^ mul3[c15] ^ rk[14]
            b15 = mul3[c12] ^ c13 ^ c14 ^ mul2[c15] ^ rk[15]

        # Final round (no MixColumns)
        rk = rks[AES_ROUNDS]
        return bytes(
            (
                sbox[b0] ^ rk[0],
                sbox[b5] ^ rk[1],
                sbox[b10] ^ rk[2],
                sbox[b15] ^ rk[3],
                sbox[b4] ^ rk[4],
                sbox[b9] ^ rk[5],
                sbox[b14] ^ rk[6],
                sbox[b3] ^ rk[7],
                sbox[b8] ^ rk[8],
                sbox[b13] ^ rk[9],
                sbox[b2] ^ rk[10],
                sbox[b7] ^ rk[11],
                sbox[b12] ^ rk[12],
                sbox[b1] ^ rk[13],
                sbox[b6] ^ rk[14],
                sbox[b11] ^ rk[15],
            )
        )

    def decrypt_block(self, block: bytes) -> bytes:
        """Decrypt a single 16-byte block using AES-128 with the inverse S-box."""
        if len(block) != AES_BLOCK_SIZE:
            raise ValueError(f"Block must be exactly {AES_BLOCK_SIZE} bytes.")

        inv_sbox = self._inv_sbox_tuple
        mul9 = _MUL9_TUPLE
        mulb = _MULB_TUPLE
        muld = _MULD_TUPLE
        mule = _MULE_TUPLE
        rks = self._round_keys_tuple

        rk = rks[AES_ROUNDS]
        b0 = block[0] ^ rk[0]
        b1 = block[1] ^ rk[1]
        b2 = block[2] ^ rk[2]
        b3 = block[3] ^ rk[3]
        b4 = block[4] ^ rk[4]
        b5 = block[5] ^ rk[5]
        b6 = block[6] ^ rk[6]
        b7 = block[7] ^ rk[7]
        b8 = block[8] ^ rk[8]
        b9 = block[9] ^ rk[9]
        b10 = block[10] ^ rk[10]
        b11 = block[11] ^ rk[11]
        b12 = block[12] ^ rk[12]
        b13 = block[13] ^ rk[13]
        b14 = block[14] ^ rk[14]
        b15 = block[15] ^ rk[15]

        for r in range(AES_ROUNDS - 1, 0, -1):
            # InvShiftRows + InvSubBytes + AddRoundKey
            rk = rks[r]
            c0 = inv_sbox[b0] ^ rk[0]
            c1 = inv_sbox[b13] ^ rk[1]
            c2 = inv_sbox[b10] ^ rk[2]
            c3 = inv_sbox[b7] ^ rk[3]

            c4 = inv_sbox[b4] ^ rk[4]
            c5 = inv_sbox[b1] ^ rk[5]
            c6 = inv_sbox[b14] ^ rk[6]
            c7 = inv_sbox[b11] ^ rk[7]

            c8 = inv_sbox[b8] ^ rk[8]
            c9 = inv_sbox[b5] ^ rk[9]
            c10 = inv_sbox[b2] ^ rk[10]
            c11 = inv_sbox[b15] ^ rk[11]

            c12 = inv_sbox[b12] ^ rk[12]
            c13 = inv_sbox[b9] ^ rk[13]
            c14 = inv_sbox[b6] ^ rk[14]
            c15 = inv_sbox[b3] ^ rk[15]

            # InvMixColumns
            b0 = mule[c0] ^ mulb[c1] ^ muld[c2] ^ mul9[c3]
            b1 = mul9[c0] ^ mule[c1] ^ mulb[c2] ^ muld[c3]
            b2 = muld[c0] ^ mul9[c1] ^ mule[c2] ^ mulb[c3]
            b3 = mulb[c0] ^ muld[c1] ^ mul9[c2] ^ mule[c3]

            b4 = mule[c4] ^ mulb[c5] ^ muld[c6] ^ mul9[c7]
            b5 = mul9[c4] ^ mule[c5] ^ mulb[c6] ^ muld[c7]
            b6 = muld[c4] ^ mul9[c5] ^ mule[c6] ^ mulb[c7]
            b7 = mulb[c4] ^ muld[c5] ^ mul9[c6] ^ mule[c7]

            b8 = mule[c8] ^ mulb[c9] ^ muld[c10] ^ mul9[c11]
            b9 = mul9[c8] ^ mule[c9] ^ mulb[c10] ^ muld[c11]
            b10 = muld[c8] ^ mul9[c9] ^ mule[c10] ^ mulb[c11]
            b11 = mulb[c8] ^ muld[c9] ^ mul9[c10] ^ mule[c11]

            b12 = mule[c12] ^ mulb[c13] ^ muld[c14] ^ mul9[c15]
            b13 = mul9[c12] ^ mule[c13] ^ mulb[c14] ^ muld[c15]
            b14 = muld[c12] ^ mul9[c13] ^ mule[c14] ^ mulb[c15]
            b15 = mulb[c12] ^ muld[c13] ^ mul9[c14] ^ mule[c15]

        # Final inverse round (no InvMixColumns)
        rk = rks[0]
        return bytes(
            (
                inv_sbox[b0] ^ rk[0],
                inv_sbox[b13] ^ rk[1],
                inv_sbox[b10] ^ rk[2],
                inv_sbox[b7] ^ rk[3],
                inv_sbox[b4] ^ rk[4],
                inv_sbox[b1] ^ rk[5],
                inv_sbox[b14] ^ rk[6],
                inv_sbox[b11] ^ rk[7],
                inv_sbox[b8] ^ rk[8],
                inv_sbox[b5] ^ rk[9],
                inv_sbox[b2] ^ rk[10],
                inv_sbox[b15] ^ rk[11],
                inv_sbox[b12] ^ rk[12],
                inv_sbox[b9] ^ rk[13],
                inv_sbox[b6] ^ rk[14],
                inv_sbox[b3] ^ rk[15],
            )
        )

    def _encrypt_blocks_batch(self, blocks: np.ndarray) -> np.ndarray:
        """Vectorized AES-128 block encryption for `(N, 16)` uint8 arrays."""
        if blocks.shape[0] == 0:
            return blocks.copy()

        sbox = self.sbox
        rks = self._round_keys_np
        state = np.bitwise_xor(blocks, rks[0])

        for r in range(1, AES_ROUNDS):
            state = sbox[state[:, _SHIFT_ROWS_IDX]]
            cols = state.reshape(-1, 4, 4)
            a0, a1, a2, a3 = cols[:, :, 0], cols[:, :, 1], cols[:, :, 2], cols[:, :, 3]

            m0 = _MUL2_NP[a0] ^ _MUL3_NP[a1] ^ a2 ^ a3
            m1 = a0 ^ _MUL2_NP[a1] ^ _MUL3_NP[a2] ^ a3
            m2 = a0 ^ a1 ^ _MUL2_NP[a2] ^ _MUL3_NP[a3]
            m3 = _MUL3_NP[a0] ^ a1 ^ a2 ^ _MUL2_NP[a3]

            mixed = np.stack((m0, m1, m2, m3), axis=2).reshape(-1, 16)
            state = np.bitwise_xor(mixed, rks[r])

        state = sbox[state[:, _SHIFT_ROWS_IDX]]
        return np.bitwise_xor(state, rks[AES_ROUNDS])

    def _decrypt_blocks_batch(self, blocks: np.ndarray) -> np.ndarray:
        """Vectorized AES-128 block decryption for `(N, 16)` uint8 arrays."""
        if blocks.shape[0] == 0:
            return blocks.copy()

        inv_sbox = self.inv_sbox
        rks = self._round_keys_np
        state = np.bitwise_xor(blocks, rks[AES_ROUNDS])

        for r in range(AES_ROUNDS - 1, 0, -1):
            state = np.bitwise_xor(inv_sbox[state[:, _INV_SHIFT_ROWS_IDX]], rks[r])
            cols = state.reshape(-1, 4, 4)
            a0, a1, a2, a3 = cols[:, :, 0], cols[:, :, 1], cols[:, :, 2], cols[:, :, 3]

            m0 = _MULE_NP[a0] ^ _MULB_NP[a1] ^ _MULD_NP[a2] ^ _MUL9_NP[a3]
            m1 = _MUL9_NP[a0] ^ _MULE_NP[a1] ^ _MULB_NP[a2] ^ _MULD_NP[a3]
            m2 = _MULD_NP[a0] ^ _MUL9_NP[a1] ^ _MULE_NP[a2] ^ _MULB_NP[a3]
            m3 = _MULB_NP[a0] ^ _MULD_NP[a1] ^ _MUL9_NP[a2] ^ _MULE_NP[a3]

            state = np.stack((m0, m1, m2, m3), axis=2).reshape(-1, 16)

        state = inv_sbox[state[:, _INV_SHIFT_ROWS_IDX]]
        return np.bitwise_xor(state, rks[0])

    def encrypt_cbc(self, plaintext: bytes, iv: bytes | None = None) -> tuple[bytes, bytes]:
        """Encrypt arbitrary-length bytes in CBC mode with PKCS#7 padding."""
        if iv is None:
            iv = os.urandom(AES_BLOCK_SIZE)
        elif len(iv) != AES_BLOCK_SIZE:
            raise ValueError(f"CBC IV must be exactly {AES_BLOCK_SIZE} bytes.")

        padded = pad_pkcs7(plaintext, AES_BLOCK_SIZE)
        out = bytearray(len(padded))
        prev = iv

        for offset in range(0, len(padded), AES_BLOCK_SIZE):
            xored = bytes(a ^ b for a, b in zip(padded[offset : offset + AES_BLOCK_SIZE], prev))
            cipher_block = self.encrypt_block(xored)
            out[offset : offset + AES_BLOCK_SIZE] = cipher_block
            prev = cipher_block

        return bytes(out), iv

    def decrypt_cbc(self, ciphertext: bytes, iv: bytes) -> bytes:
        """Decrypt CBC ciphertext and validate/strip PKCS#7 padding."""
        if len(iv) != AES_BLOCK_SIZE:
            raise ValueError(f"CBC IV must be exactly {AES_BLOCK_SIZE} bytes.")
        if not ciphertext or len(ciphertext) % AES_BLOCK_SIZE != 0:
            raise ValueError(f"CBC ciphertext must be non-empty and a multiple of {AES_BLOCK_SIZE} bytes.")

        c_blocks = np.frombuffer(ciphertext, dtype=np.uint8).reshape(-1, AES_BLOCK_SIZE)
        dec_blocks = self._decrypt_blocks_batch(c_blocks)

        iv_arr = np.frombuffer(iv, dtype=np.uint8).reshape(1, AES_BLOCK_SIZE)
        prev_blocks = np.vstack((iv_arr, c_blocks[:-1])) if len(c_blocks) > 1 else iv_arr
        plain_padded = np.bitwise_xor(dec_blocks, prev_blocks).tobytes()
        return unpad_pkcs7(plain_padded, AES_BLOCK_SIZE)

    @staticmethod
    def _normalize_ctr_nonce(nonce: bytes | None) -> tuple[bytes, bytes]:
        """Return `(initial_counter_16B, returned_nonce_bytes)`."""
        if nonce is None:
            full = os.urandom(AES_BLOCK_SIZE)
            return full, full
        if len(nonce) == AES_BLOCK_SIZE:
            return bytes(nonce), bytes(nonce)
        if len(nonce) == 8:
            return bytes(nonce) + b"\x00" * 8, bytes(nonce)
        raise ValueError("CTR nonce must be 8 bytes (nonce prefix) or 16 bytes (initial counter).")

    def _generate_ctr_keystream(self, initial_counter: bytes, n_bytes: int) -> bytes:
        if n_bytes == 0:
            return b""
        num_blocks = (n_bytes + AES_BLOCK_SIZE - 1) // AES_BLOCK_SIZE
        base_int = int.from_bytes(initial_counter, "big")
        mod_128 = 1 << 128

        counter_buf = bytearray(num_blocks * AES_BLOCK_SIZE)
        for idx in range(num_blocks):
            val = (base_int + idx) % mod_128
            counter_buf[idx * AES_BLOCK_SIZE : (idx + 1) * AES_BLOCK_SIZE] = val.to_bytes(
                AES_BLOCK_SIZE, "big"
            )

        counter_blocks = np.frombuffer(counter_buf, dtype=np.uint8).reshape(num_blocks, AES_BLOCK_SIZE)
        keystream_blocks = self._encrypt_blocks_batch(counter_blocks)
        return keystream_blocks.tobytes()[:n_bytes]

    def encrypt_ctr(self, plaintext: bytes, nonce: bytes | None = None) -> tuple[bytes, bytes]:
        """Encrypt arbitrary-length bytes in CTR mode (no padding required)."""
        initial_counter, returned_nonce = self._normalize_ctr_nonce(nonce)
        if not plaintext:
            return b"", returned_nonce

        keystream = self._generate_ctr_keystream(initial_counter, len(plaintext))
        pt_arr = np.frombuffer(plaintext, dtype=np.uint8)
        ks_arr = np.frombuffer(keystream, dtype=np.uint8)
        return np.bitwise_xor(pt_arr, ks_arr).tobytes(), returned_nonce

    def decrypt_ctr(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """Decrypt CTR ciphertext using the same keystream XOR operation."""
        initial_counter, _ = self._normalize_ctr_nonce(nonce)
        if not ciphertext:
            return b""

        keystream = self._generate_ctr_keystream(initial_counter, len(ciphertext))
        ct_arr = np.frombuffer(ciphertext, dtype=np.uint8)
        ks_arr = np.frombuffer(keystream, dtype=np.uint8)
        return np.bitwise_xor(ct_arr, ks_arr).tobytes()

    def encrypt(
        self,
        plaintext: bytes,
        mode: str = "CBC",
        iv_or_nonce: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Encrypt plaintext using an allowed AES mode (`CBC` or `CTR`)."""
        norm_mode = _normalize_mode(mode)
        if norm_mode == "CBC":
            return self.encrypt_cbc(plaintext, iv=iv_or_nonce)
        return self.encrypt_ctr(plaintext, nonce=iv_or_nonce)

    def decrypt(
        self,
        ciphertext: bytes,
        iv_or_nonce: bytes,
        mode: str = "CBC",
    ) -> bytes:
        """Decrypt ciphertext using an allowed AES mode (`CBC` or `CTR`)."""
        norm_mode = _normalize_mode(mode)
        if norm_mode == "CBC":
            return self.decrypt_cbc(ciphertext, iv=iv_or_nonce)
        return self.decrypt_ctr(ciphertext, nonce=iv_or_nonce)
