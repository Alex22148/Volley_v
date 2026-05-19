import base64
import hmac
import hashlib
import json
from pathlib import Path




class FilenameCodecStream:
    def __init__(self, secret_key: bytes, map_file: str = "filename_map.json"):
        self.secret_key = secret_key
        self.map_file = Path(map_file)

        if self.map_file.exists():
            self.mapping = json.loads(self.map_file.read_text())
        else:
            self.mapping = {}

        # licznik = ile już zakodowano nazw
        self.counter = len(self.mapping)

    # -------------------------------------------------------
    # generuje krótką nazwę na podstawie numeru pozycji
    # -------------------------------------------------------
    def encode_index(self, index: int) -> str:
        data = str(index).encode()
        digest = hmac.new(self.secret_key, data, hashlib.sha256).digest()
        short = digest[:6]  # 48 bit → 8 znaków Base32
        return base64.b32encode(short).decode("ascii").rstrip("=")

    # -------------------------------------------------------
    # kodowanie strumieniowe
    # -------------------------------------------------------
    def encode(self, filename: str) -> str:
        if filename not in self.mapping:
            encoded = self.encode_index(self.counter)
            self.mapping[filename] = encoded
            self.counter += 1
            self.save()

        return self.mapping[filename]

    # -------------------------------------------------------
    # dekodowanie — odtworzenie oryginalnej nazwy
    # -------------------------------------------------------
    def decode(self, encoded: str):
        for original, enc in self.mapping.items():
            if enc == encoded:
                return original
        return None  # nie znaleziono (nie było wcześniej kodowane)

    # -------------------------------------------------------
    def save(self):
        self.map_file.write_text(json.dumps(self.mapping, indent=2))



# =================================================
# pip install zstandard cryptography
import struct
import time
from pathlib import Path

import zstandard as zstd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import hashlib
import os as _os


# ======================================================
#  PARAMETRY (z rozsądnymi domyślnymi wartościami)
# ======================================================

# PBKDF2 (hasło -> klucz AES-256)
PBKDF2_ITER = 200_000
KEY_LEN = 32          # 32 bajty = AES-256
NONCE_LEN = 12        # AES-GCM nonce
SALT_LEN = 16         # PBKDF2 salt

# Magiczki formatu
OUTER_MAGIC = b"BWPZ"  # zewnętrzny nagłówek archiwum
INNER_MAGIC = b"BWPA"  # wewnętrzny format plików


# ======================================================
#  POMOCNICZE – KDF: hasło -> klucz AES-256
# ======================================================

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LEN,
        salt=salt,
        iterations=PBKDF2_ITER,
        backend=default_backend()
    )
    return kdf.derive(password.encode("utf-8"))


# ======================================================
#  PAKOWANIE
# ======================================================

def pack_webp_pro(
    folder_path,
    archive_path=None,
    password: str | None = None,
    delete_originals: bool = False,
    threads: int = 1,
    zstd_level: int = 15,
):
    """
    Pakuje wszystkie *.webp z folderu do jednego archiwum:
    - buduje wewnętrzny strumień (INNER_MAGIC + nagłówki + dane)
    - kompresuje go ZSTD (domyślnie level=15, 1 wątek – mało RAMu)
    - opcjonalnie szyfruje AES-256-GCM (hasło -> klucz przez PBKDF2)
    - zapisuje plik .bwpz z nagłówkiem OUTER_MAGIC
    """

    folder = Path(folder_path)
    if archive_path is None:
        archive_path = folder.with_suffix(".bwpz")
    archive_path = Path(archive_path)

    files = sorted(folder.glob("*.webp"))
    if not files:
        print("[WARN] Brak plików WEBP w folderze.")
        return None

    # print(f"[INFO] Znaleziono {len(files)} plików WEBP w {folder}")

    total_size = sum(f.stat().st_size for f in files)
    # print(f"[INFO] Łączny rozmiar wejściowy: {total_size/1024/1024:.2f} MB")

    # -------------------------------------
    # Budowa wewnętrznego bufora "raw"
    # -------------------------------------
    raw = bytearray()
    raw += INNER_MAGIC           # 4 bajty
    raw += struct.pack("B", 1)   # wersja
    raw += b"\x00\x00\x00"       # reserved

    raw += struct.pack("<I", len(files))  # liczba plików

    # print("[INFO] Liczenie SHA256 i budowa wewnętrznego bufora...")
    for f in files:
        data = f.read_bytes()
        name_bytes = f.name.encode("utf-8")
        sha256 = hashlib.sha256(data).digest()

        if len(name_bytes) > 65535:
            raise ValueError("Nazwa pliku za długa (>65535 bajtów).")

        # [name_len][name][size][sha256][data]
        raw += struct.pack("<H", len(name_bytes))
        raw += name_bytes
        raw += struct.pack("<Q", len(data))  # uint64
        raw += sha256
        raw += data

    # -------------------------------------
    # Kompresja ZSTD
    # -------------------------------------
    if threads < 1:
        threads = 1
    level = int(zstd_level)

    # print(f"[INFO] Kompresja ZSTD level={level}, threads={threads}...")
    t0_comp = time.perf_counter()
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    compressed = cctx.compress(bytes(raw))
    t1_comp = time.perf_counter()
    # print(f"[INFO] Kompresja zakończona, dt={t1_comp - t0_comp:.3f}s, "
    #       f"rozmiar skompresowany: {len(compressed)/1024/1024:.2f} MB")

    # -------------------------------------
    # Opcjonalne szyfrowanie AES-256-GCM
    # -------------------------------------
    flags = 0
    header = bytearray()
    header += OUTER_MAGIC      # 4 bajty magic
    header += struct.pack("B", 1)  # wersja
    # bit0 w flags = 1 oznacza "zaszyfrowane"
    if password is not None:
        flags |= 1
    header += struct.pack("B", flags)
    header += b"\x00\x00"      # reserved

    if password is not None:
        # print("[INFO] Szyfrowanie AES-256-GCM...")
        salt = _os.urandom(SALT_LEN)
        nonce = _os.urandom(NONCE_LEN)
        key = derive_key(password, salt)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, compressed, None)

        # do pliku: [OUTER_MAGIC][ver][flags][res][salt][nonce][ciphertext]
        payload = header + salt + nonce + ct
    else:
        # bez szyfrowania: [OUTER_MAGIC][ver][flags=0][res][compressed]
        payload = header + compressed

    archive_path.write_bytes(payload)
    out_size = archive_path.stat().st_size
    print(f"   [INFO] Zapisano archiwum: {archive_path}"
          f"\n          Rozmiar archiwum: {out_size / 1024 / 1024:.2f} MB ")
          #f"(ratio: {out_size / total_size:.3f}x)")
    # out_size = archive_path.stat().st_size
    # print(f"[INFO] Rozmiar archiwum: {out_size/1024/1024:.2f} MB "
    #       f"(ratio: {out_size/total_size:.3f}x)")

    if delete_originals:
        for f in files:
            f.unlink()
        folder.rmdir()
        # for f in files:
        #     f.unlink()
        # print("[INFO] Usunięto oryginalne pliki WEBP.")

    # print(f"*** pack_webp_pro: dt_kopmresja={t1_comp - t0_comp:.6f}s")
    return archive_path


# ======================================================
#  ROZPAKOWYWANIE
# ======================================================

def unpack_webp_pro(
    archive_path,
    output_folder=None,
    password: str | None = None,
    verify_checksums: bool = True,
):
    """
    Rozpakowuje archiwum stworzone przez pack_webp_pro.
    - rozpoznaje, czy jest szyfrowane
    - odszyfrowuje (jeśli trzeba)
    - dekompresuje ZSTD
    - odtwarza pliki + opcjonalnie sprawdza SHA256
    """

    archive_path = Path(archive_path)
    if output_folder is None:
        output_folder = archive_path.with_suffix("")

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    print(f"[INFO] Rozpakowuję archiwum: {archive_path}")
    comp_all = archive_path.read_bytes()

    if len(comp_all) < 8:
        raise ValueError("Plik za krótki, brak nagłówka.")

    magic = comp_all[:4]
    if magic != OUTER_MAGIC:
        print(f"[DEBUG] magic z pliku: {magic}")
        raise ValueError("Nieprawidłowa sygnatura pliku (magic != BWPZ).")

    ver = comp_all[4]
    flags = comp_all[5]
    # comp_all[6:8] reserved
    offset = 8

    encrypted = bool(flags & 1)

    # -------------------------------------
    # Odszyfrowanie (jeśli trzeba)
    # -------------------------------------
    if encrypted:
        if password is None:
            raise ValueError("Archiwum jest zaszyfrowane, podaj hasło.")
        if len(comp_all) < offset + SALT_LEN + NONCE_LEN + 16:
            raise ValueError("Archiwum zaszyfrowane wydaje się uszkodzone.")

        salt = comp_all[offset:offset + SALT_LEN]
        offset += SALT_LEN
        nonce = comp_all[offset:offset + NONCE_LEN]
        offset += NONCE_LEN

        key = derive_key(password, salt)
        aesgcm = AESGCM(key)
        ciphertext = comp_all[offset:]

        print("[INFO] Odszyfrowywanie AES-256-GCM...")
        t0_dec = time.perf_counter()
        compressed = aesgcm.decrypt(nonce, ciphertext, None)
        t1_dec = time.perf_counter()
        print(f"[INFO] Odszyfrowano w {t1_dec - t0_dec:.3f}s")
    else:
        compressed = comp_all[offset:]

    # -------------------------------------
    # Dekompresja ZSTD
    # -------------------------------------
    print("[INFO] Dekompresja ZSTD...")
    t0_dcomp = time.perf_counter()
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(compressed)
    t1_dcomp = time.perf_counter()
    print(f"[INFO] Dekompresja zakończona, dt={t1_dcomp - t0_dcomp:.3f}s")

    # -------------------------------------
    # Parsowanie wewnętrznego formatu
    # -------------------------------------
    if len(raw) < 8:
        raise ValueError("Wewnętrzny strumień za krótki.")

    inner_magic = raw[:4]
    if inner_magic != INNER_MAGIC:
        print(f"[DEBUG] inner_magic z wewnętrznego strumienia: {inner_magic}")
        raise ValueError("Nieprawidłowy format wewnętrzny (magic != BWPA).")

    inner_ver = raw[4]
    # raw[5:8] reserved
    offset = 8
    (count,) = struct.unpack_from("<I", raw, offset)
    offset += 4

    print(f"[INFO] W archiwum wewnętrznym: {count} plików")

    errors = 0

    for _ in range(count):
        name_len = struct.unpack_from("<H", raw, offset)[0]
        offset += 2
        name = raw[offset:offset + name_len].decode("utf-8")
        offset += name_len
        (size,) = struct.unpack_from("<Q", raw, offset)
        offset += 8
        sha_ref = raw[offset:offset + 32]
        offset += 32
        data = raw[offset:offset + size]
        offset += size

        out_path = output_folder / name
        out_path.write_bytes(data)

        if verify_checksums:
            sha_real = hashlib.sha256(data).digest()
            if sha_real != sha_ref:
                print(f"[ERR] Niezgodna suma SHA256 dla pliku: {name}")
                errors += 1

    if verify_checksums:
        if errors == 0:
            print("[INFO] Wszystkie sumy SHA256 poprawne.")
        else:
            print(f"[WARN] Wykryto {errors} błędów sum kontrolnych: {errors}")

    print("[INFO] Rozpakowywanie zakończone.")
    return output_folder
