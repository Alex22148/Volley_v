# shared_buffer.py
import numpy as np
import json
from multiprocessing import shared_memory, Lock, Value

_META_BYTES = 256  # max rozmiar JSON metadanych na slot

class SharedImageBuffer:
    """
    Współdzielony cykliczny bufor obrazów (N slotów).
    - Obraz: surowy bufor shm (N * image_size)
    - Metadane: osobny shm (N * _META_BYTES), per-slot
    - Synchronizacja: lock + liczniki (write/read/available)
    Tryby:
      - creator=True  -> tworzy segmenty
      - creator=False -> dołącza po nazwach
    """
    def __init__(self, buffer_size: int, image_shape: tuple, dtype=np.uint8,
                 creator: bool = True,
                 shm_name: str = None, meta_name: str = None):
        self.buffer_size = int(buffer_size)
        self.image_shape = tuple(image_shape)
        self.dtype = np.dtype(dtype)
        self.image_size = int(np.prod(self.image_shape) * self.dtype.itemsize)
        self.total_size = self.image_size * self.buffer_size

        self.lock = Lock()
        self.write_index = Value('i', 0)
        self.read_index  = Value('i', 0)
        self.available   = Value('i', 0)

        if creator:
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=shm_name)
            self.meta_shm = shared_memory.SharedMemory(create=True, size=self.buffer_size * _META_BYTES, name=meta_name)
        else:
            if not shm_name or not meta_name:
                raise ValueError("Client needs shm_name and meta_name")
            self.shm = shared_memory.SharedMemory(create=False, name=shm_name)
            self.meta_shm = shared_memory.SharedMemory(create=False, name=meta_name)

        self._arr = np.ndarray((self.buffer_size, *self.image_shape), dtype=self.dtype, buffer=self.shm.buf)
        self._meta = np.ndarray((self.buffer_size, _META_BYTES), dtype=np.uint8, buffer=self.meta_shm.buf)

    # --------- identyfikatory / opis do przekazania między procesami ----------
    def export_spec(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "image_shape": self.image_shape,
            "dtype": self.dtype.str,
            "shm_name": self.shm.name,
            "meta_name": self.meta_shm.name
        }

    @staticmethod
    def client_from_spec(spec: dict) -> "SharedImageBuffer":
        return SharedImageBuffer(
            buffer_size=spec["buffer_size"],
            image_shape=tuple(spec["image_shape"]),
            dtype=np.dtype(spec["dtype"]),
            creator=False,
            shm_name=spec["shm_name"],
            meta_name=spec["meta_name"]
        )

    # --------- zapis / odczyt jako ring-buffer ----------
    def write_image(self, image: np.ndarray, metadata: dict) -> int | None:
        """
        Kopiuje obraz do następnego slotu. Zwraca slot_id (int) lub None (gdy pełny).
        Uwaga: przechowujemy dane do czasu odczytu przez konsumenta.
        """
        if image.shape != self.image_shape or image.dtype != self.dtype:
            raise ValueError(f"Invalid image shape/dtype: got {image.shape} {image.dtype}, expected {self.image_shape} {self.dtype}")

        with self.lock:
            if self.available.value >= self.buffer_size:
                return None  # pełny – nie nadpisujemy, sygnalizujemy drop
            idx = self.write_index.value
            # kopiuj dane
            self._arr[idx, ...] = image
            # metadane per slot
            meta_str = json.dumps(metadata or {}, ensure_ascii=False)
            meta_bytes = meta_str.encode("utf-8")[:_META_BYTES]
            self._meta[idx, :len(meta_bytes)] = np.frombuffer(meta_bytes, dtype=np.uint8)
            if len(meta_bytes) < _META_BYTES:
                self._meta[idx, len(meta_bytes):] = 0

            self.write_index.value = (idx + 1) % self.buffer_size
            self.available.value += 1
            return idx

    def read_image(self) -> tuple | None:
        """
        Zwraca (image_copy, metadata_dict, slot_id) lub None gdy pusto.
        """
        with self.lock:
            if self.available.value <= 0:
                return None
            idx = self.read_index.value
            image = self._arr[idx].copy()  # konsument dostaje kopię (bez blokowania slotu)
            meta_bytes = self._meta[idx].tobytes()
            try:
                meta_str = meta_bytes.split(b"\x00", 1)[0].decode("utf-8")
                metadata = json.loads(meta_str) if meta_str else {}
            except Exception:
                metadata = {}
            self.read_index.value = (idx + 1) % self.buffer_size
            self.available.value -= 1
            return image, metadata, idx

    # --------- odczyt po uchwycie (slot_id) bez modyfikacji wskaźników ----------
    def peek_by_slot(self, slot_id: int) -> tuple[np.ndarray, dict]:
        """
        Zwraca (view_na_bufor, metadata) dla podanego slotu bez zmiany liczników.
        UWAGA: zwracamy *widok* na shm – nie modyfikuj go in-place między procesami!
        """
        if not (0 <= slot_id < self.buffer_size):
            raise ValueError("Invalid slot_id")
        img_view = self._arr[slot_id]  # bez copy
        meta_bytes = self._meta[slot_id].tobytes()
        try:
            meta_str = meta_bytes.split(b"\x00", 1)[0].decode("utf-8")
            metadata = json.loads(meta_str) if meta_str else {}
        except Exception:
            metadata = {}
        return img_view, metadata

    def cleanup(self):
        try:
            self.shm.close()
            self.meta_shm.close()
        finally:
            # unlink tylko po stronie tworzącej
            try: self.shm.unlink()
            except: pass
            try: self.meta_shm.unlink()
            except: pass
