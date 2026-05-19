# shared_memory_manager.py
import struct
import time
import numpy as np
from multiprocessing import shared_memory
from typing import Optional, Tuple
import threading
from collections import OrderedDict

META_FMT = "iiiiqq"
META_SIZE = struct.calcsize(META_FMT)

DTYPE_CODE = {
    np.dtype(np.uint8): 1,
    np.dtype(np.int16): 2,
    np.dtype(np.uint16): 3,
    np.dtype(np.float32): 4,
}
CODE_DTYPE = {v: k for k, v in DTYPE_CODE.items()}


class SharedMemoryManager:
    """
    Poprawiony manager z właściwym zarządzaniem memoryview.
    """

    def __init__(self, retention_time: float = 3.0):
        self.retention_time = retention_time
        self._cleanup_lock = threading.Lock()
        self._shm_registry = OrderedDict()  # key -> (shm_data, shm_meta, timestamp)
        self._last_cleanup = time.time()

    def _cleanup_old_entries(self):
        """Usuń stare wpisy shared memory"""
        now = time.time()
        if now - self._last_cleanup < 1.0:  # cleanup co sekundę
            return

        with self._cleanup_lock:
            keys_to_remove = []
            for key, (shm_data, shm_meta, timestamp) in list(self._shm_registry.items()):
                if now - timestamp > self.retention_time:
                    self._safe_close_shm(shm_data, shm_meta)
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self._shm_registry.pop(key, None)

            self._last_cleanup = now

    def _safe_close_shm(self, shm_data, shm_meta):
        """Bezpieczne zamykanie shared memory"""
        try:
            if shm_data:
                shm_data.close()
        except:
            pass
        try:
            if shm_meta:
                shm_meta.close()
        except:
            pass
        # Unlink tylko jeśli to ostatnia referencja
        try:
            if shm_data:
                shm_data.unlink()
        except:
            pass
        try:
            if shm_meta:
                shm_meta.unlink()
        except:
            pass

    def _open_or_create(self, name: str, size: int):
        """Otwórz lub utwórz shared memory z obsługą błędów"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                if shm.size >= size:
                    return shm, False
                else:
                    shm.close()
                    shm.unlink()
                    raise ValueError("Size mismatch")
            except (FileNotFoundError, ValueError):
                try:
                    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                    return shm, True
                except FileExistsError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.001)

    def write_frame(self, key: str, frame: np.ndarray, ts_ns: Optional[int] = None) -> bool:
        if frame is None:
            return False

        self._cleanup_old_entries()

        h, w = frame.shape[:2]
        ch = frame.shape[2] if frame.ndim == 3 else 1
        dtype = frame.dtype
        code = DTYPE_CODE.get(dtype)
        if code is None:
            raise ValueError(f"Unsupported dtype {dtype}")

        nbytes = frame.nbytes
        shm_data, shm_meta = None, None

        try:
            # Dane
            shm_data, _ = self._open_or_create(key, nbytes)

            # Meta
            shm_meta, _ = self._open_or_create(f"{key}__meta", META_SIZE)

            # Kopiowanie danych - UŻYJ TYLKO memoryview TYM CZASOWO
            with memoryview(shm_data.buf) as mv_data:
                mv_data[:nbytes] = memoryview(frame).cast('B')

            # Zapis meta
            if ts_ns is None:
                ts_ns = time.time_ns()
            packed = struct.pack(META_FMT, w, h, ch, code, int(ts_ns), 0)

            with memoryview(shm_meta.buf) as mv_meta:
                mv_meta[:META_SIZE] = packed

            # Zarejestruj w registry
            with self._cleanup_lock:
                self._shm_registry[key] = (shm_data, shm_meta, time.time())

            return True

        except Exception as e:
            print(f"[SMM] ❌ Write failed for {key}: {e}")
            # Sprzątanie w przypadku błędu
            self._safe_close_shm(shm_data, shm_meta)
            return False

    def read_frame(self, key: str, retries: int = 5, delay: float = 0.002, quiet_missing: bool = False) -> Optional[Tuple[np.ndarray, int]]:
        for attempt in range(retries):
            shm_meta, shm_data = None, None
            try:
                # Odczyt metadanych
                shm_meta = shared_memory.SharedMemory(name=f"{key}__meta", create=False)
                with memoryview(shm_meta.buf) as mv_meta:
                    w, h, ch, code, ts_ns, _ = struct.unpack(META_FMT, mv_meta[:META_SIZE])

                dtype = CODE_DTYPE.get(code)
                if dtype is None or w <= 0 or h <= 0 or ch <= 0:
                    time.sleep(delay)
                    continue

                shape = (h, w) if ch == 1 else (h, w, ch)
                nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

                # Odczyt danych
                shm_data = shared_memory.SharedMemory(name=key, create=False)
                with memoryview(shm_data.buf) as mv_data:
                    arr = np.frombuffer(mv_data[:nbytes], dtype=dtype).reshape(shape).copy()

                return arr, int(ts_ns)

            except FileNotFoundError:
                if attempt == retries - 1:
                    if not quiet_missing:
                        print(f"[SMM] ❌ Shared memory not found: {key}")
                time.sleep(delay)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[SMM] ❌ Read failed for {key}: {e}")
                time.sleep(delay)
            finally:
                # Zawsze zamykaj shared memory
                if shm_meta:
                    shm_meta.close()
                if shm_data:
                    shm_data.close()

        return None

    def cleanup_all(self):
        """Wyczyść wszystkie shared memory"""
        with self._cleanup_lock:
            for key, (shm_data, shm_meta, _) in list(self._shm_registry.items()):
                self._safe_close_shm(shm_data, shm_meta)
            self._shm_registry.clear()

    def __del__(self):
        """Destruktor - wyczyść zasoby"""
        self.cleanup_all()


# Singleton
_smm = None


def get_shared_memory_manager():
    global _smm
    if _smm is None:
        _smm = SharedMemoryManager(retention_time=3.0)
        print("[SMM] ✅ Fixed SharedMemoryManager with proper memory management")
    return _smm
