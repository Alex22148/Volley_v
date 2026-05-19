# lib_webp.py

from lib_pack import (FilenameCodecStream, pack_webp_pro)
import os
import struct
from typing import Iterator, Tuple, Iterable
import multiprocessing
import msvcrt

import cv2
import numpy as np
from PIL import Image


HEADER_TAG = b"BFRM"
HEADER_FMT = "<4sIIIQI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def wait_for_escape():
    print("\nNaciśnij ESC aby zamknąć aplikację...")
    while True:
        key = msvcrt.getch()
        if key in (b'\x1b',):  # ESC
            break


def estimate_frames_in_bin(bin_path: str) -> int:
    """
    Szacuje liczbę ramek w pliku .bin na podstawie pierwszego nagłówka.
    Zakładamy, że wszystkie ramki mają ten sam payload_nbytes.
    """
    file_size = os.path.getsize(bin_path)
    if file_size < HEADER_SIZE:
        return 0

    with open(bin_path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            return 0
        tag, w, h, c, ts_ns, payload_nbytes = struct.unpack(HEADER_FMT, header)
        if tag != HEADER_TAG or payload_nbytes <= 0:
            return 0

    frame_size = HEADER_SIZE + payload_nbytes
    if frame_size <= 0:
        return 0

    return max(1, file_size // frame_size)


# 6) Pasek postępu
def _print_progress2(prefix: str, current: int, total: int, bar_length: int = 30):
    if total <= 0:
        return
    frac = current / total
    filled = int(bar_length * frac)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = frac * 100.0
    sys.stdout.write(f"\r{prefix} [{bar}] {percent:6.2f}% ({current}/{total})")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")



def _save_webp_image(
    out_path: str,
    img: Image.Image,
    quality: int,
    method: int,
) -> tuple[bool, str | None]:
    """
    Funkcja pomocnicza – zapisuje pojedynczy obraz do WEBP.
    Zwraca (success, error_message).
    """
    try:
        # zakładam, że img jest już w trybie "RGB" / docelowym – nie robimy zbędnych konwersji
        img.save(
            out_path,
            format="WEBP",
            quality=quality,
            method=method,
        )
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        # zwolnij pamięć jak najszybciej
        try:
            img.close()
        except Exception:
            pass



def export_single_bin_to_webp(
    bin_path: str,
    out_dir: str,
    quality: int = 100,
    method: int = 6,
    progress_cb=True,                # callback po każdej klatce  None
    estimated_frames: int | None = None,
    is_aborted=None,                 # funkcja bool -> True = przerwij
) -> int:
    """
    Konwersja JEDNEGO pliku .bin na serię .webp.

    progress_cb(frames_done_in_this_bin, estimated_frames, bin_path)
    is_aborted() -> True  => przerwij pętlę.
    """
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(bin_path))[0]
    frame_all = 0
    frame_converted = 0

    if estimated_frames is None:
        try:
            estimated_frames = estimate_frames_in_bin(bin_path)
        except Exception:
            estimated_frames = None

    # print(estimated_frames)

    frame_iter = iter_frames_from_bin(bin_path)  # domyślnie strict=False

    show100 = False
    for idx, (img, w, h, c, ts_ns) in enumerate(frame_iter):
        if is_aborted is not None and is_aborted():
            print(f"   [WEBP] Przerwano plik {bin_path} po {frame_all} ramkach (przed kolejną).")
            break

        out_name = f"{base}_{idx:04d}_ts_ns_{ts_ns}.webp"
        out_path = os.path.join(out_dir, out_name)

        try:
            img.save(
                out_path,
                format="WEBP",
                quality=quality,
                method=method,
            )
            frame_converted += 1
        except Exception as e:
            print(f"   [WEBP][ERROR] {e}")
        frame_all += 1

        if progress_cb is not None:
            try:
                # progress_cb(frame_all, estimated_frames or 0, bin_path)
                _print_progress2(f"   [WEBP] {base}", frame_all, estimated_frames)
                if frame_all == estimated_frames:
                    show100=True
            except Exception:
                pass

        # przerwanie PO zapisaniu
        if is_aborted is not None and is_aborted():
            print(f"   [WEBP] Przerwano plik {bin_path} po {frame_all} ramkach (po tej klatce).")
            break

    if not show100:
        _print_progress2(f"   [WEBP] {base}", estimated_frames, estimated_frames)

    # Po iteracji statystyki:
    stats = frame_iter.stats
    print(
        f"   [WEBP] {bin_path}: "
        f"\n          dobre ramki = {stats['good_frames']}, "
        f"podejrzanie uszkodzone = {stats['bad_frames']}, "
        f"resynców = {stats['resync_events']}"
    )

    return frame_all




from typing import Iterator, Tuple, Iterable, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, Future
import multiprocessing

def export_single_bin_to_webp_fast(
    bin_path: str,
    out_dir: str,
    quality: int = 100,
    method: int = 6,
    max_workers: int | None = None,
    progress_cb: Optional[Callable[[int, Optional[int], str], None]] = None,
    estimated_frames: int | None = None,
    is_aborted: Optional[Callable[[], bool]] = None,
    secret_key=b"SuperTajnyKluczKodowania123",
    PASSWORD = "MojeBardzoTajneHaslo123",
) -> int:
    """
    SZYBSZA wersja export_single_bin_to_webp, z równoległym zapisem WEBP.

    Kluczowe różnice:
      - dla każdej klatki uruchamiamy zapis WEBP w osobnym wątku (ThreadPoolExecutor),
      - w głównej pętli czytamy kolejne ramki z pliku .bin,
      - liczba jednocześnie przetwarzanych zadań jest ograniczona, aby nie zalać RAM-u.

    Parametry:
        bin_path        – wejściowy plik .bin
        out_dir         – katalog wyjściowy na pliki .webp
        quality, method – parametry kodowania WEBP (tak jak dotąd)
        progress_cb     – opcjonalny callback: progress_cb(frames_done, estimated_frames, bin_path)
        estimated_frames – opcjonalna szacowana liczba klatek
        is_aborted      – opcjonalna funkcja bool -> True = przerwij jak najszybciej
        max_workers     – liczba wątków; domyślnie os.cpu_count()

    Zwraca:
        frame_converted – liczba poprawnie zapisanych klatek.
    """


    os.makedirs(out_dir, exist_ok=True)

    folder = Path(out_dir)
    archive_pathx = folder.with_suffix(".json")

    # folder, w którym znajduje się plik
    parent_dir = archive_pathx.parent

    # folder INFO o jeden poziom wyżej
    info_dir = parent_dir.parent / "INFO"
    info_dir.mkdir(exist_ok=True)

    # nowa ścieżka docelowa
    archive_path = info_dir / archive_pathx.name


    codec = FilenameCodecStream(
        secret_key=secret_key,
        map_file=str(archive_path)
    )

    base = os.path.splitext(os.path.basename(bin_path))[0]
    frame_all = 0
    frame_converted = 0

    if estimated_frames is None:
        try:
            estimated_frames = estimate_frames_in_bin(bin_path)
        except Exception:
            estimated_frames = None

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() or 1)

    # # import lokalny, żeby nie zrobił się import cykliczny w innych modułach
    # from eksportBinWebp import iter_frames_from_bin  # jeśli masz w innym module, popraw import

    frame_iter = iter_frames_from_bin(bin_path)  # domyślnie strict=False

    # w pending trzymamy mapowanie Future -> (idx, ts_ns)
    pending: dict[Future, tuple[int, int]] = {}

    def harvest_some(done_futures):
        nonlocal frame_all, frame_converted
        show100 = True
        for fut in done_futures:
            idx, ts_ns = pending.pop(fut)
            ok, err = fut.result()
            frame_all += 1
            if ok:
                frame_converted += 1
                _print_progress2(f"   [WEBP][FAST] {base}", frame_all, estimated_frames)
                if frame_all == estimated_frames:
                    show100=True
            else:
                print(f"   [WEBP][ERROR][frame {idx}] {err}")
            if progress_cb is not None:
                progress_cb(frame_converted, estimated_frames, bin_path)
        if not show100:
            _print_progress2(f"   [WEBP] {base}", estimated_frames, estimated_frames)

    # ile zadań maksymalnie trzymamy „w locie” (żeby nie zjadać RAM-u)
    max_pending = max_workers * 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (img, w, h, c, ts_ns) in enumerate(frame_iter):
            if is_aborted is not None and is_aborted():
                break

            # opcjonalnie: upewnij się, że wszystkie obrazy są w tym samym trybie (np. RGB)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            out_name = f"{base}_{idx:04d}_ts_ns_{ts_ns}"
            short_name = codec.encode(out_name)+".webp"
            out_path = os.path.join(out_dir, short_name)



            fut = executor.submit(_save_webp_image, out_path, img, quality, method)
            pending[fut] = (idx, ts_ns)

            # kontrola liczby zadań w locie – jeśli za dużo, poczekaj na część z nich
            if len(pending) >= max_pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                harvest_some(done)

                if is_aborted is not None and is_aborted():
                    break

        # po zakończeniu dopisz resztę
        if pending:
            done, _ = wait(pending.keys())
            harvest_some(done)

    # możesz w razie potrzeby zalogować statystyki
    print(
        f"   [WEBP][FAST] {bin_path}: \n                pozyskano {frame_converted}/{frame_all} klatek "
        f"(workers={max_workers}, quality={quality}, method={method})"
    )

    archive = pack_webp_pro(
        folder_path=out_dir,
        archive_path=None,  # automatycznie <folder>.bwpz
        password=PASSWORD,  # None -> bez szyfrowania
        delete_originals=True,  # True -> usuń WEBP po spakowaniu
        threads=2,  # mało RAMu, bezpiecznie
        zstd_level=15  # spokojny poziom kompresji
    )


    return frame_converted



# ==========================
#  Bayer -> RGB (OpenCV + PIL)
# ==========================

def bayer_rg8_to_rgb_pil(bayer_arr: np.ndarray) -> Image.Image:
    """
    bayer_arr: 2D numpy array (h, w), dtype=uint8, wzór Bayer (u Ciebie BG/RG).
    Zwraca: PIL.Image w trybie 'RGB'.

    U mnie jest COLOR_BayerBG2RGB, bo pisałaś, że tak kolory się zgadzają.
    Jeśli faktycznie jest BayerRG – zmień na COLOR_BayerRG2RGB.
    """
    if bayer_arr.dtype != np.uint8:
        bayer_arr = bayer_arr.astype(np.uint8)

    if bayer_arr.ndim != 2:
        raise ValueError(f"Oczekiwana tablica 2D (h,w), dostałem shape={bayer_arr.shape}")

    if not bayer_arr.flags.c_contiguous:
        bayer_arr = np.ascontiguousarray(bayer_arr)

    # OpenCV: BayerBG8 -> RGB
    rgb = cv2.cvtColor(bayer_arr, cv2.COLOR_BayerBG2RGB)

    img = Image.fromarray(rgb, mode="RGB")
    return img


# dodatkowa klasa
class _BinFrameIterator:
    """
    Iterator po ramkach z obsługą błędów i resynchronizacją.

    Atrybuty:
        good_frames   – liczba poprawnie odczytanych ramek
        bad_frames    – liczba „zdarzeń błędu” (podejrzanie uszkodzonych ramek)
        resync_events – ile razy trzeba było szukać kolejnego nagłówka BFRM

    Dostępne też jako dict:
        .stats -> {"good_frames": ..., "bad_frames": ..., "resync_events": ...}
    """

    def __init__(self, bin_path: str, strict: bool = False, max_resync_bytes: int | None = 50_000_000):
        self.bin_path = bin_path
        self.strict = strict
        self.max_resync_bytes = max_resync_bytes

        self._f = open(bin_path, "rb")
        self._file_size = os.path.getsize(bin_path)

        self.frame_idx = 0
        self.good_frames = 0
        self.bad_frames = 0
        self.resync_events = 0

        # „Oczekiwana” geometria i wielkość payloadu – na podstawie pierwszej dobrej ramki.
        self._expected_w: int | None = None
        self._expected_h: int | None = None
        self._expected_c: int | None = None
        self._expected_payload_nbytes: int | None = None

        self._done = False

    # --- API pomocnicze ---

    @property
    def stats(self) -> dict:
        return {
            "good_frames": self.good_frames,
            "bad_frames": self.bad_frames,
            "resync_events": self.resync_events,
        }

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __iter__(self):
        return self

    # --- Walidacja nagłówka ---

    def _header_valid(self, tag, w, h, c, payload_nbytes) -> bool:
        # TAG musi się zgadzać
        if tag != HEADER_TAG:
            return False

        # Wymiary w sensownym zakresie (tu: „rozsądne” limity)
        if w <= 0 or h <= 0 or w > 10000 or h > 10000:
            return False

        # Dozwolone kanały – u Ciebie 1 (Bayer), ewentualnie 3/4
        if c not in (1, 3, 4):
            return False

        if payload_nbytes <= 0:
            return False

        # Czy payload_nbytes odpowiada w*h*c?
        try:
            expected_payload = w * h * c
        except OverflowError:
            return False

        if payload_nbytes != expected_payload:
            return False

        # Spójność z pierwszą poprawną ramką (stała geometria)
        if self._expected_w is not None:
            if (
                w != self._expected_w
                or h != self._expected_h
                or c != self._expected_c
                or payload_nbytes != self._expected_payload_nbytes
            ):
                return False

        return True

    # --- Szukanie kolejnego nagłówka BFRM po błędzie ---

    def _resync_from(self, start_offset: int) -> bool:
        """
        Szuka kolejnego poprawnego nagłówka (BFRM + sensowne w,h,c,payload)
        zaczynając od `start_offset`.

        Zwraca:
            True  – jeśli znalazł (ustawia wskaźnik pliku NA POCZĄTKU nagłówka)
            False – jeśli nie znalazł (EOF lub przekroczono max_resync_bytes)
        """
        self._f.seek(start_offset)
        scanned = 0
        buf = bytearray()

        while True:
            b = self._f.read(1)
            if not b:
                return False

            scanned += 1
            if self.max_resync_bytes is not None and scanned > self.max_resync_bytes:
                return False

            buf.append(b[0])
            if len(buf) > len(HEADER_TAG):
                buf.pop(0)

            if bytes(buf) == HEADER_TAG:
                candidate_pos = self._f.tell() - len(HEADER_TAG)
                saved_pos = self._f.tell()

                # Sprawdź pełny nagłówek od tego miejsca
                self._f.seek(candidate_pos)
                header = self._f.read(HEADER_SIZE)
                if len(header) != HEADER_SIZE:
                    return False  # koniec pliku / ucięty nagłówek

                tag, w, h, c, ts_ns, payload_nbytes = struct.unpack(HEADER_FMT, header)

                if self._header_valid(tag, w, h, c, payload_nbytes):
                    payload_start = candidate_pos + HEADER_SIZE
                    if payload_start + payload_nbytes <= self._file_size:
                        # Mamy sensowną ramkę – zsynchronizowaliśmy się.
                        self.resync_events += 1
                        self._f.seek(candidate_pos)
                        return True

                # Kandydat był fałszywy – wróć i szukaj dalej
                self._f.seek(saved_pos)
                buf.clear()

    # --- Główna logika iteratora ---

    def __next__(self):
        if self._done:
            raise StopIteration

        while True:
            header_pos = self._f.tell()
            header = self._f.read(HEADER_SIZE)

            # EOF
            if not header:
                self._done = True
                self.close()
                raise StopIteration

            # ucięty nagłówek
            if len(header) != HEADER_SIZE:
                self.bad_frames += 1
                msg = f"Ucięty nagłówek przy ramce {self.frame_idx} (offset {header_pos})."
                if self.strict:
                    self._done = True
                    self.close()
                    raise ValueError(msg)

                # spróbuj resynchronizacji
                if not self._resync_from(header_pos + 1):
                    self._done = True
                    self.close()
                    raise StopIteration
                # jeśli resync się udał -> ponownie wejdź w pętlę i czytaj nowy nagłówek
                continue

            tag, w, h, c, ts_ns, payload_nbytes = struct.unpack(HEADER_FMT, header)

            # błędny / podejrzany nagłówek
            if not self._header_valid(tag, w, h, c, payload_nbytes):
                self.bad_frames += 1
                msg = (
                    f"Zły nagłówek przy ramce {self.frame_idx}: "
                    f"tag={tag!r}, w={w}, h={h}, c={c}, payload_nbytes={payload_nbytes} "
                    f"(offset {header_pos})."
                )
                if self.strict:
                    self._done = True
                    self.close()
                    raise ValueError(msg)

                if not self._resync_from(header_pos + 1):
                    self._done = True
                    self.close()
                    raise StopIteration
                continue

            # Pierwsza dobra ramka ustala „normę”.
            if self._expected_w is None:
                self._expected_w = w
                self._expected_h = h
                self._expected_c = c
                self._expected_payload_nbytes = payload_nbytes

            # upewnij się, że payload mieści się w pliku
            payload_start = self._f.tell()
            if payload_start + payload_nbytes > self._file_size:
                self.bad_frames += 1
                msg = (
                    f"Ucięty payload przy ramce {self.frame_idx} "
                    f"(offset {payload_start}, payload_nbytes={payload_nbytes})."
                )
                if self.strict:
                    self._done = True
                    self.close()
                    raise ValueError(msg)

                if not self._resync_from(header_pos + 1):
                    self._done = True
                    self.close()
                    raise StopIteration
                continue

            payload = self._f.read(payload_nbytes)
            if len(payload) != payload_nbytes:
                # Teoretycznie nie powinno się zdarzyć po wcześniejszym sprawdzeniu,
                # ale na wszelki wypadek:
                self.bad_frames += 1
                msg = (
                    f"Ucięty payload przy ramce {self.frame_idx} "
                    f"(read={len(payload)}, expected={payload_nbytes})."
                )
                if self.strict:
                    self._done = True
                    self.close()
                    raise ValueError(msg)

                if not self._resync_from(header_pos + 1):
                    self._done = True
                    self.close()
                    raise StopIteration
                continue

            # --- przekształcenie payloadu w obraz (jak w Twojej oryginalnej funkcji) ---
            arr = np.frombuffer(payload, dtype=np.uint8)

            try:
                if c == 1:
                    arr = arr.reshape((h, w))
                    img = bayer_rg8_to_rgb_pil(arr)
                else:
                    arr = arr.reshape((h, w, c))
                    if c == 3:
                        img = Image.fromarray(arr, mode="RGB")
                    elif c == 4:
                        img = Image.fromarray(arr, mode="RGBA")
                    else:
                        raise ValueError(f"Nieobsługiwane c={c} przy ramce {self.frame_idx}")
            except Exception:
                # coś poszło nie tak przy reshape/konwersji – traktuj jako uszkodzoną ramkę
                self.bad_frames += 1
                if self.strict:
                    self._done = True
                    self.close()
                    raise

                if not self._resync_from(header_pos + 1):
                    self._done = True
                    self.close()
                    raise StopIteration
                continue

            # sukces – zwróć ramkę
            self.good_frames += 1
            out = (img, w, h, c, ts_ns)
            self.frame_idx += 1
            return out



# ==========================
#  Iterator po ramkach w .bin
# ==========================

def iter_frames_from_bin(
    bin_path: str,
    *,
    strict: bool = False,
    max_resync_bytes: int | None = 50_000_000,
) -> Iterator[Tuple[Image.Image, int, int, int, int]]:
    """
    Generator zwracający kolejne ramki z pliku .bin zapisanego przez saver_worker_bin.

    Jeśli strict=False (domyślnie):
        - przy błędzie nagłówka/payloadu próbuje się zsynchronizować do kolejnego
          poprawnego nagłówka 'BFRM' o oczekiwanej geometrii i wielkości payloadu,
        - uszkodzone fragmenty są pomijane, kolejne dobre ramki są nadal zwracane.

    Jeśli strict=True:
        - zachowuje się podobnie do Twojej starej wersji – przy pierwszym błędzie
          rzuca wyjątek (ale z dodatkowymi komunikatami).

    Po zakończeniu iteracji można odczytać statystyki:
        it = iter_frames_from_bin(bin_path)
        for idx, (img, w, h, c, ts_ns) in enumerate(it):
            ...
        print(it.stats)  # {"good_frames": ..., "bad_frames": ..., "resync_events": ...}
    """
    return _BinFrameIterator(bin_path, strict=strict, max_resync_bytes=max_resync_bytes)


# def iter_frames_from_bin(bin_path: str) -> Iterator[Tuple[Image.Image, int, int, int, int]]:
#     """
#     Generator zwracający kolejne ramki z pliku .bin zapisanego przez saver_worker_bin.
#
#     Yields:
#         img   : PIL.Image (RGB / RGBA)
#         w, h  : szerokość, wysokość
#         c     : liczba kanałów
#         ts_ns : timestamp w nanosekundach
#     """
#     with open(bin_path, "rb") as f:
#         frame_idx = 0
#         # print(HEADER_TAG)
#         while True:
#             header = f.read(HEADER_SIZE)
#             # print(f" {header}")
#             if not header:
#                 break
#
#             if len(header) != HEADER_SIZE:
#                 raise ValueError(f"Ucięty nagłówek przy ramce {frame_idx}")
#
#             tag, w, h, c, ts_ns, payload_nbytes = struct.unpack(HEADER_FMT, header)
#             # print(f'   ---')
#             # print(f'   {tag}')
#             # print(f'   {w}')
#             # print(f'   {h}')
#             # print(f'   {c}')
#             # print(f'   {ts_ns}')
#             if tag != HEADER_TAG:
#                 raise ValueError(f"Zły TAG przy ramce {frame_idx}: {tag!r}")
#
#             payload = f.read(payload_nbytes)
#             if len(payload) != payload_nbytes:
#                 raise ValueError(f"Ucięty payload przy ramce {frame_idx}")
#
#             arr = np.frombuffer(payload, dtype=np.uint8)
#
#             if c == 1:
#                 # Bayer -> RGB
#                 arr = arr.reshape((h, w))
#                 img = bayer_rg8_to_rgb_pil(arr)
#             else:
#                 arr = arr.reshape((h, w, c))
#                 if c == 3:
#                     img = Image.fromarray(arr, mode="RGB")
#                 elif c == 4:
#                     img = Image.fromarray(arr, mode="RGBA")
#                 else:
#                     raise ValueError(f"Nieobsługiwane c={c} przy ramce {frame_idx}")
#
#             yield img, w, h, c, ts_ns
#             frame_idx += 1


def convert_bin_batch(
    bin_path: str,
    out_dir: str,
    quality: int = 100,
    method: int = 6
) -> int:
    """
    Funkcja 1:
    Konwersja JEDNEGO pliku .bin (batcha) do wskazanego katalogu.

    Przykład:
        convert_bin_batch(
            r"...\\batch_001.bin",
            r"...\\output\\batch_001_frames",
            quality=90,
            method=4,
        )
    """
    return export_single_bin_to_webp(bin_path, out_dir, quality=quality, method=method)




# =======================================================
def convert_session(
    session_num: int,
    session_max: int,
    session_dir: str,
    output_root: str,
    quality: int = 1, #100
    method: int = 1, # 6
    max_workers: int | None = None,
    cams: Iterable[str] = ("CENTER_L", "CENTER_R", "LEFT", "RIGHT")
) -> None:
    """
    Funkcja 3:
    Konwersja wszystkich plików .bin w całej sesji (4 foldery kamer).

    Struktura wejścia (session_dir):
        session_YYYYMMDD_HHMMSS\\CENTER_L\\*.bin
                              \\CENTER_R\\*.bin
                              \\RIGHT\\*.bin
                              \\LEFT\\*.bin

    Struktura wyjścia (output_root):
        output_root\\webp\\CENTER_L_0000\\*.webp
                            CENTER_L_0001\\*.webp
                            ...
                            CENTER_R_0000\\*.webp
                            ...

    Czyli:
      - nazewnictwo folderów jak w Twoim bazowym przykładzie (cam_idx),
      - ale GŁÓWNY root zapisu (output_root) jest wybierany (np. z okienka).
    """
    session_dir = os.path.abspath(session_dir)
    output_root = os.path.abspath(output_root)

    webp_root = os.path.join(output_root, "Raw_sets")
    os.makedirs(webp_root, exist_ok=True)

    for cam in cams:
        cam_dir = os.path.join(session_dir, cam)
        if not os.path.isdir(cam_dir):
            print(f"[SESSION] Uwaga: brak katalogu kamery {cam_dir}")
            continue

        files = sorted([
            f for f in os.listdir(cam_dir)
            if f.lower().endswith(".bin")
        ])

        cam_old=""
        for idx, fname in enumerate(files):
            bin_path = os.path.join(cam_dir, fname)

            # Tu zachowujemy logikę jak w Twoim bazowym kodzie:
            # outdir = <output_root>/webp/<cam>_<idx:04d>
            out_dir = os.path.join(webp_root,str(cam), f"{cam}_{idx:04d}")

            if cam_old != cam:
                cam_old = cam
                print("")
                print("-" * 50)

            print(f"\n[SESSION {session_num}/{session_max}] {cam} | plik {idx}: {bin_path} \n          -> {out_dir}")
            export_single_bin_to_webp_fast(bin_path, out_dir, quality=quality, method=method,max_workers=max_workers)




# =======================================================
import shutil
def sort_folders(
    output_dir: str,
    N=250,
    show_info = False,
    cams=("CENTER_L", "CENTER_R", "RIGHT", "LEFT"),
) -> None:
    """
    Grupuje podfoldery webp z kamer w B000, B001, ...

    Zakładamy strukturę po konwersji sesji:

        output_dir/
            webp/
                CENTER_L/
                    CENTER_L_0000/
                    CENTER_L_0001/
                    ...
                CENTER_R/
                    CENTER_R_0000/
                    ...
                RIGHT/
                LEFT/

    Tworzymy:

        output_folder/
            B000/
                CENTER_L_0000/
                CENTER_R_0000/
                RIGHT_0000/
                LEFT_0000/
            B001/
                CENTER_L_0001/
                CENTER_R_0001/
                ...

    UWAGA: kopiujemy katalogi (copy), nie przenosimy.
    """

    # session_dir = os.path.abspath(session_dir)      # na wszelki wypadek, choć tu nie używamy
    output_dir = os.path.abspath(output_dir)
    output_folder = os.path.join(output_dir,"Learning_sets") #os.path.abspath(output_folder)

    webp_root = os.path.join(output_dir, "Raw_sets")
    if not os.path.isdir(webp_root):
        raise RuntimeError(f"Nie znaleziono katalogu webp: {webp_root}")

    os.makedirs(output_folder, exist_ok=True)

    # zbierz listę podfolderów dla każdej kamery
    cam_subdirs: dict[str, list[str]] = {}
    for cam in cams:
        cam_dir = os.path.join(webp_root, cam)
        if not os.path.isdir(cam_dir):
            print(f"[SORT] Uwaga: brak katalogu kamery {cam_dir}, pomijam tę kamerę.")
            continue

        subdirs = sorted(
            d for d in os.listdir(cam_dir)
            if os.path.isdir(os.path.join(cam_dir, d))
        )
        cam_subdirs[cam] = subdirs
        # print(f"[SORT] {cam}: znaleziono {len(subdirs)} podfolderów.")

    if not cam_subdirs:
        raise RuntimeError("Brak podfolderów webp dla jakiejkolwiek kamery.")

    # # ile „batchy” maksymalnie
    # max_batches = max(len(subs) for subs in cam_subdirs.values())
    # print(f"[SORT] Maksymalna liczba batchy: {max_batches}")

    print()
    records_cam={}
    for cam, subdirs in cam_subdirs.items():
        records_all=[]
        # print(f"{cam}: {subdirs}")
        for dirx in subdirs:
            folder = os.path.join(webp_root, cam, dirx)
            # print(folder)
            records = cam_batch_timestamps(folder)
            records_all.extend(records)
        records_cam[cam] = records_all


    time_records = build_time_records(records_cam, period_ms=20.0, tolerance_ms=5.0)



    # informacje: dane opisowe, tabelaryczne, wykresy
    num_slots = len(time_records["slots"])
    print("[SORT] Liczba slotów:", num_slots)


    slots = time_records["slots"]
    if slots:
        t0_ns = time_records["t0_ns"]  # czas startu (nominalny slot 0)
        t_last_ns = slots[-1]["t_nominal_ns"]  # czas ostatniego slotu
        duration_ns = t_last_ns - t0_ns
        duration_ms = duration_ns / 1e6
        duration_s = duration_ns / 1e9

        print("[SORT] t0_ns:", t0_ns)
        print("[SORT] t_last_ns:", t_last_ns)
        print(f"[SORT] Przedział czasowy: {duration_ms:.3f} ms (~{duration_s:.3f} s)")

    print()
    if N>1000:
        N=1000
    print(f"[SORT] The sets will contain {N*4} pieces each.")
    # zapis paczek do uczenia
    out_root = os.path.join(output_dir, "Learning_sets")
    os.makedirs(out_root, exist_ok=True)
    summary = export_batches_from_time_records_v3(
        time_records,
        out_root=out_root,
        N=N,
        cams=["CENTER_L", "CENTER_R", "LEFT", "RIGHT"],
        copy_real_images=True,  # <<< największy zysk
        quality=50,
        method=6,
        show_progress=True,
        progress_update_every=500,  # pasek co 500 zapisanych plików
    )



    # print(time_records['slots'][570])
    # A = time_records['slots'][570]['frames']
    # print(A['CENTER_L'])
    # print(A['CENTER_R'])
    # print(A['LEFT'])
    # print(A['RIGHT'])

    if show_info:
        print()
        # Przykład: jak znaleźć tylko te sloty, w których są wszystkie 4 kamery
        full_quads = [
            slot for slot in time_records["slots"]
            if all(cam in slot["frames"] for cam in ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"])
        ]

        full_quads = [
            slot for slot in time_records["slots"]
            if all(cam in slot["frames"] for cam in ["CENTER_L", "CENTER_R", "LEFT", "RIGHT"])
        ]

        print("Liczba slotów z kompletem 4 kamer:", len(full_quads))
        for slot in full_quads:
            print(
                "slot_idx =", slot["slot_idx"],
                "t_rel_ms =", slot["t_rel_ms"],
                "kamery =", list(slot["frames"].keys()),
            )


        # Dostęp do ścieżek i różnic czasowych:
        for slot in full_quads[:3]:  # trzy pierwsze komplety
            print("\nSlot:", slot["slot_idx"], "t_rel_ms =", slot["t_rel_ms"])
            for cam, frame in slot["frames"].items():
                print(
                    f"  {cam}: index={frame['index']}, dt_ms={frame['dt_ms']:.3f}, path={frame['path']}"
                )

        # 1) Heatmapa dostępności
        plot_camera_availability_heatmap(
            time_records,
            cams=["CENTER_L", "CENTER_R", "LEFT", "RIGHT"],  # kolejność na osi Y
            max_slots=300,  # np. pierwsze 300 slotów
        )

        # 2) Różnice czasowe (dt_ms) dla każdej kamery
        plot_dt_per_camera(
            time_records,
            cams=["CENTER_L", "CENTER_R", "LEFT", "RIGHT"],
            max_slots=300,
        )

        # 3) Liczba kamer w każdym slocie
        plot_num_cams_per_slot(time_records, max_slots=None)



    print(f"[SORT] Zapisano dane do etykietowania {out_root}")


# ========================================================

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


def build_time_records(
    records_cam: Dict[str, List[Tuple[int, int, Path]]],
    period_ms: float = 20.0,
    tolerance_ms: float = 5.0,
) -> Dict[str, Any]:
    """
    Tworzy uporządkowane rekordy czasowe ze zbioru records_cam.

    records_cam:
        {
            "CENTER_L": [(idx, ts_ns, path), ...],
            "CENTER_R": [...],
            "LEFT":     [...],
            "RIGHT":    [...],
        }

    Zwracany słownik zawiera:
        {
            "t0_ns": int,               # pierwszy (najmniejszy) znacznik czasowy
            "period_ms": float,         # użyty krok, domyślnie 20.0
            "tolerance_ms": float,      # użyta tolerancja, domyślnie 5.0
            "slots": List[dict],        # lista slotów po kolei w czasie
        }

    Każdy element w "slots" ma postać:
        {
            "slot_idx": int,            # indeks slotu (0,1,2,... liczone od t0)
            "t_nominal_ns": int,        # nominalny czas slotu w ns
            "t_rel_ms": float,          # czas slotu względem t0 w ms
            "frames": {
                "CENTER_L": {
                    "cam": "CENTER_L",
                    "index": idx,
                    "ts_ns": ts_ns,
                    "path": Path(...),
                    "dt_ns": int,       # ts_ns - t_nominal_ns
                    "dt_ms": float,     # dt_ns / 1e6
                    "within_tolerance": bool,
                },
                "CENTER_R": {...},
                "LEFT": {...},
                "RIGHT": {...},
            }
        }

    Slot może mieć pusty "frames", jeśli dla tej chwili czasowej nie ma żadnego obrazu.
    """

    period_ns = int(round(period_ms * 1e6))
    tolerance_ns = int(round(tolerance_ms * 1e6))

    # 1) Zbierz wszystkie obrazy w jedną listę: (ts_ns, cam, idx, path)
    events: List[Tuple[int, str, int, Path]] = []
    for cam, records in records_cam.items():
        for idx, ts_ns, path in records:
            events.append((int(ts_ns), cam, int(idx), path))

    if not events:
        return {
            "t0_ns": None,
            "period_ms": period_ms,
            "tolerance_ms": tolerance_ms,
            "slots": [],
        }

    # 2) Posortuj po czasie i wyznacz t0 (początek osi czasu)
    events.sort(key=lambda e: e[0])
    t0_ns = events[0][0]

    # 3) Przypisz każdy obraz do najbliższego slotu co 20 ms
    #    slot_idx = zaokrąglony ((ts - t0) / period)
    #    przy zaokrąglaniu używamy integerowej wersji "round":
    #    (delta + period/2) // period
    slots_by_idx: Dict[int, Dict[str, Any]] = {}

    for ts_ns, cam, idx, path in events:
        delta_ns = ts_ns - t0_ns
        # najbliższy slot
        slot_idx = int((delta_ns + period_ns // 2) // period_ns)
        t_nominal_ns = t0_ns + slot_idx * period_ns
        dt_ns = ts_ns - t_nominal_ns
        dt_ms = dt_ns / 1e6
        within_tolerance = abs(dt_ns) <= tolerance_ns

        # utworzenie slotu, jeśli nie istnieje
        if slot_idx not in slots_by_idx:
            slots_by_idx[slot_idx] = {
                "slot_idx": slot_idx,
                "t_nominal_ns": t_nominal_ns,
                "t_rel_ms": (t_nominal_ns - t0_ns) / 1e6,
                "frames": {},
            }

        slot = slots_by_idx[slot_idx]
        frames = slot["frames"]

        frame_info = {
            "cam": cam,
            "index": idx,
            "ts_ns": ts_ns,
            "path": path,
            "dt_ns": dt_ns,
            "dt_ms": dt_ms,
            "within_tolerance": within_tolerance,
        }

        # Jeśli w tym slocie jest już obraz z tej kamery – wybierz bliższy czasowo
        if cam in frames:
            existing = frames[cam]
            if abs(frame_info["dt_ns"]) < abs(existing["dt_ns"]):
                frames[cam] = frame_info
        else:
            frames[cam] = frame_info

    # 4) Zbuduj ciągłą listę slotów: od min do max indeksu,
    #    także te, które są zupełnie puste (brak obrazów)
    slot_indices = sorted(slots_by_idx.keys())
    min_idx, max_idx = slot_indices[0], slot_indices[-1]

    slots: List[Dict[str, Any]] = []
    for slot_idx in range(min_idx, max_idx + 1):
        if slot_idx in slots_by_idx:
            slot = slots_by_idx[slot_idx]
        else:
            t_nominal_ns = t0_ns + slot_idx * period_ns
            slot = {
                "slot_idx": slot_idx,
                "t_nominal_ns": t_nominal_ns,
                "t_rel_ms": (t_nominal_ns - t0_ns) / 1e6,
                "frames": {},  # pusty slot – brak obrazów w tej chwili
            }
        slots.append(slot)

    return {
        "t0_ns": t0_ns,
        "period_ms": period_ms,
        "tolerance_ms": tolerance_ms,
        "slots": slots,
    }




# =======================================================
from pathlib import Path
from typing import Dict, List, Any, Optional

from PIL import Image


def export_batches_from_time_records(
    time_records: Dict[str, Any],
    out_root: str | Path,
    N: int,
    cams: Optional[List[str]] = None,
    quality: int = 50,
    method: int = 6,
) -> Dict[str, Any]:
    """
    Dzieli dane z time_records na paczki po N *niepustych* slotów
    i zapisuje je w strukturze katalogów:

        out_root/
          B_000/
            CENTER_L/
              CENTER_L_000_000.webp
              CENTER_L_000_001.webp
              ...
            CENTER_R/
            LEFT/
            RIGHT/
          B_001/
          ...

    Parametry:
        time_records – wynik funkcji build_time_records(...)
        out_root     – katalog wyjściowy (zostanie utworzony, jeśli nie istnieje)
        N            – liczba niepustych slotów na paczkę
        cams         – lista kamer; jeśli None, wykrywana jest automatycznie
        quality      – jakość WEBP (0–100), im niższa tym mniejszy plik
        method       – poziom kompresji WEBP (0–6), 6 = najlepsza kompresja

    Zwraca:
        {
            "total_non_empty_slots": int,
            "N": int,
            "num_batches": int,
            "batches": [
                {
                    "batch_idx": int,
                    "batch_id_str": str,
                    "batch_dir": Path,
                    "num_slots": int,
                },
                ...
            ],
        }
    """
    if N <= 0:
        raise ValueError("N musi być > 0")

    slots = time_records.get("slots", [])
    if not slots:
        raise ValueError("time_records['slots'] jest puste – brak danych do eksportu.")

    # 1) Lista kamer – jeśli nie podano, bierzemy wszystkie występujące
    if cams is None:
        cam_set = set()
        for slot in slots:
            cam_set.update(slot["frames"].keys())
        cams = sorted(cam_set)

    if not cams:
        raise ValueError("Nie znaleziono żadnych kamer w time_records.")

    # 2) Filtrujemy tylko sloty niepuste (tam, gdzie jest co najmniej jedna klatka)
    non_empty_slots = [slot for slot in slots if slot["frames"]]
    total_non_empty = len(non_empty_slots)

    if total_non_empty == 0:
        raise ValueError("Brak niepustych slotów – nie ma czego eksportować.")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 3) Ustalenie rozmiarów obrazów dla każdej kamery
    #    (bierzemy pierwszy napotkany obraz z danej kamery).
    size_by_cam: Dict[str, tuple[int, int]] = {}
    for cam in cams:
        for slot in slots:
            frame = slot["frames"].get(cam)
            if frame is not None:
                with Image.open(frame["path"]) as im:
                    size_by_cam[cam] = im.size  # (width, height)
                break
        if cam not in size_by_cam:
            # Domyślny fallback – np. fullHD
            size_by_cam[cam] = (1920, 1080)

    # 4) Przygotowanie czarnych obrazów jako placeholderów dla brakujących klatek
    black_by_cam: Dict[str, Image.Image] = {}
    for cam, (w, h) in size_by_cam.items():
        black_by_cam[cam] = Image.new("RGB", (w, h), color=(0, 0, 0))

    # 5) Podział na paczki po N niepustych slotów
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    batches = chunk_list(non_empty_slots, N)

    batches_info: List[Dict[str, Any]] = []

    for batch_idx, batch_slots in enumerate(batches):
        batch_id_str = f"{batch_idx:03d}"  # B_000, B_001, ...
        batch_dir = out_root / f"B_{batch_id_str}"

        # Utwórz katalogi dla każdej kamery
        for cam in cams:
            (batch_dir / cam).mkdir(parents=True, exist_ok=True)

        # Przetwarzanie slotów w paczce
        for local_slot_idx, slot in enumerate(batch_slots):
            slot_str = f"{local_slot_idx:03d}"  # 000..N-1 w obrębie paczki

            for cam in cams:
                out_name = f"{cam}_{batch_id_str}_{slot_str}.webp"
                out_path = batch_dir / cam / out_name

                frame = slot["frames"].get(cam)
                if frame is not None:
                    # Mamy prawdziwy obraz – wczytujemy i zapisujemy jako WEBP
                    src_path = frame["path"]
                    try:
                        with Image.open(src_path) as im:
                            im = im.convert("RGB")
                            im.save(
                                out_path,
                                format="WEBP",
                                quality=quality,
                                method=method,
                            )
                    except Exception as e:
                        # Gdyby coś poszło nie tak – awaryjnie zapisujemy czarny obraz
                        black_by_cam[cam].save(
                            out_path,
                            format="WEBP",
                            quality=quality,
                            method=method,
                        )
                else:
                    # Brak obrazu z tej kamery w tym slocie – zapisujemy czarny placeholder
                    black_by_cam[cam].save(
                        out_path,
                        format="WEBP",
                        quality=quality,
                        method=method,
                    )

        batches_info.append(
            {
                "batch_idx": batch_idx,
                "batch_id_str": batch_id_str,
                "batch_dir": batch_dir,
                "num_slots": len(batch_slots),
            }
        )

    summary = {
        "total_non_empty_slots": total_non_empty,
        "N": N,
        "num_batches": len(batches_info),
        "batches": batches_info,
    }

    # Proste logowanie – możesz usunąć lub dostosować
    print(
        f"Wyeksportowano {total_non_empty} niepustych slotów "
        f"w {len(batches_info)} paczkach po N={N} (ostatnia może być krótsza)."
    )
    for b in batches_info:
        print(
            f"  B_{b['batch_id_str']} -> {b['num_slots']} slotów, katalog: {b['batch_dir']}"
        )

    return summary


from pathlib import Path
from typing import Dict, List, Any, Optional

import shutil
from PIL import Image
import sys
import math


def export_batches_from_time_records_v2(
    time_records: Dict[str, Any],
    out_root: str | Path,
    N: int,
    cams: Optional[List[str]] = None,
    copy_real_images: bool = True,
    quality: int = 1,
    method: int = 1,
    show_progress: bool = True,
    progress_update_every: int = 200,
) -> Dict[str, Any]:
    """
    SZYBSZA wersja eksportu paczek z time_records.

    Kluczowe optymalizacje:
        - Prawdziwe obrazy (.webp) są domyślnie TYLKO KOPIOWANE (shutil.copy2),
          bez ponownego otwierania i kompresji przez PIL.
        - Czarny placeholder dla brakujących obrazów jest zakodowany do WEBP
          RAZ na kamerę, a potem tylko kopiowany jako plik.

    Parametry:
        time_records    – wynik build_time_records(...)
        out_root        – katalog wyjściowy
        N               – liczba NIEPUSTYCH slotów na paczkę
        cams            – lista kamer; jeśli None, wykrywana z danych
        copy_real_images – jeśli True: prawdziwe obrazy są tylko kopiowane;
                           jeśli False: zachowaj zachowanie z v1 (open+save WEBP)
        quality, method – parametry kompresji WEBP dla CZARNYCH placeholderów
                          oraz dla ewentualnego re-kodowania, jeśli copy_real_images=False
        show_progress   – czy pokazywać pasek postępu
        progress_update_every – co ile plików odświeżać pasek postępu

    Zwraca:
        {
            "total_non_empty_slots": int,
            "N": int,
            "num_batches": int,
            "batches": [
                {
                    "batch_idx": int,
                    "batch_id_str": str,
                    "batch_dir": Path,
                    "num_slots": int,
                },
                ...
            ],
        }
    """
    if N <= 0:
        raise ValueError("N musi być > 0")

    slots = time_records.get("slots", [])
    if not slots:
        raise ValueError("time_records['slots'] jest puste – brak danych do eksportu.")

    # 1) Lista kamer
    if cams is None:
        cam_set = set()
        for slot in slots:
            cam_set.update(slot["frames"].keys())
        cams = sorted(cam_set)

    if not cams:
        raise ValueError("Nie znaleziono żadnych kamer w time_records.")

    # 2) Filtrujemy tylko sloty niepuste (tam, gdzie jest co najmniej jedna klatka)
    non_empty_slots = [slot for slot in slots if slot["frames"]]
    total_non_empty = len(non_empty_slots)
    if total_non_empty == 0:
        raise ValueError("Brak niepustych slotów – nie ma czego eksportować.")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 3) Wyznacz rozmiary obrazów dla każdej kamery (na podstawie pierwszego napotkanego obrazu)
    size_by_cam: Dict[str, tuple[int, int]] = {}
    for cam in cams:
        for slot in slots:
            frame = slot["frames"].get(cam)
            if frame is not None:
                with Image.open(frame["path"]) as im:
                    size_by_cam[cam] = im.size  # (width, height)
                break
        if cam not in size_by_cam:
            # Domyślna rozdzielczość, jeśli dla danej kamery nie ma ani jednego obrazu
            size_by_cam[cam] = (1920, 1080)

    # 4) Przygotuj „szablonowe” czarne obrazy WEBP (jeden plik na kamerę)
    #    Później będziemy TYLKO je kopiować.
    black_template_dir = out_root / "_black_templates"
    black_template_dir.mkdir(parents=True, exist_ok=True)

    black_template_paths: Dict[str, Path] = {}

    for cam, (w, h) in size_by_cam.items():
        black_img = Image.new("RGB", (w, h), color=(0, 0, 0))
        tmpl_path = black_template_dir / f"{cam}_black.webp"
        black_img.save(
            tmpl_path,
            format="WEBP",
            quality=quality,
            method=method,
        )
        black_template_paths[cam] = tmpl_path

    # 5) Funkcja do dzielenia na paczki po N niepustych slotów
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    batches = chunk_list(non_empty_slots, N)

    # 6) Pasek postępu – prosty tekstowy
    def _print_progress(prefix: str, current: int, total: int, bar_length: int = 30):
        if total <= 0:
            return
        frac = current / total
        filled = int(bar_length * frac)
        bar = "#" * filled + "-" * (bar_length - filled)
        percent = frac * 100.0
        sys.stdout.write(f"\r{prefix} [{bar}] {percent:6.2f}% ({current}/{total})")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    batches_info: List[Dict[str, Any]] = []

    # Ile łącznie plików powstanie? (do globalnego progress bara, jeśli chcesz)
    # total_files = total_non_empty * len(cams)

    for batch_idx, batch_slots in enumerate(batches):
        batch_id_str = f"{batch_idx:03d}"  # np. 000, 001, ...
        batch_dir = out_root / f"B_{batch_id_str}"

        print(f"[SORT] B_{batch_id_str} - Starting to copy")

        # Utwórz katalogi dla każdej kamery
        for cam in cams:
            (batch_dir / cam).mkdir(parents=True, exist_ok=True)

        num_slots_in_batch = len(batch_slots)
        files_in_batch = num_slots_in_batch * len(cams)
        processed_files = 0

        # Przetwarzanie slotów w paczce
        for local_slot_idx, slot in enumerate(batch_slots):
            slot_str = f"{local_slot_idx:03d}"  # 000..N-1 w obrębie paczki

            for cam in cams:
                out_name = f"{cam}_{batch_id_str}_{slot_str}.webp"
                out_path = batch_dir / cam / out_name
                frame = slot["frames"].get(cam)

                if frame is not None:
                    src_path = frame["path"]
                    if copy_real_images:
                        # SZYBKA ŚCIEŻKA: kopiujemy istniejący WEBP 1:1
                        try:
                            shutil.copy2(src_path, out_path)
                        except Exception:
                            # awaryjnie – czarny placeholder
                            shutil.copy2(black_template_paths[cam], out_path)
                    else:
                        # STARA ŚCIEŻKA: re-kodowanie przez PIL (wolniejsze)
                        try:
                            with Image.open(src_path) as im:
                                im = im.convert("RGB")
                                im.save(
                                    out_path,
                                    format="WEBP",
                                    quality=quality,
                                    method=method,
                                )
                        except Exception:
                            shutil.copy2(black_template_paths[cam], out_path)
                else:
                    # Brak obrazu – czarny placeholder
                    shutil.copy2(black_template_paths[cam], out_path)

                processed_files += 1
                if show_progress and (processed_files % progress_update_every == 0 or processed_files == files_in_batch):
                    _print_progress(f"[SORT] B_{batch_id_str}", processed_files, files_in_batch)

        if show_progress and processed_files == files_in_batch:
            # Upewnij się, że linia postępu dla paczki kończy się nową linią
            _print_progress(f"[SORT] B_{batch_id_str}", files_in_batch, files_in_batch)

        print(f"[SORT] B_{batch_id_str} - Done ({num_slots_in_batch} slots, {files_in_batch} files)")

        batches_info.append(
            {
                "batch_idx": batch_idx,
                "batch_id_str": batch_id_str,
                "batch_dir": batch_dir,
                "num_slots": num_slots_in_batch,
            }
        )

    summary = {
        "total_non_empty_slots": total_non_empty,
        "N": N,
        "num_batches": len(batches_info),
        "batches": batches_info,
    }

    print(
        f"[SORT] Wyeksportowano {total_non_empty} niepustych slotów "
        f"w {len(batches_info)} paczkach po N={N} (ostatnia może być krótsza)."
    )

    return summary







# TABELA --------------------------
def print_time_records_table(
    time_records: Dict[str, Any],
    cams: Optional[List[str]] = None,
    max_rows: Optional[int] = 50,
    show_only_nonempty: bool = False,
) -> None:
    """
    Prosty, tekstowy podgląd rekordów czasowych.

    time_records – wynik funkcji build_time_records(...)

    cams – kolejność kamer w tabeli; jeśli None, używa wszystkich znalezionych
           w danych (posortowanych alfabetycznie).

    max_rows – maksymalna liczba wierszy do wypisania (None = wszystkie).

    show_only_nonempty – jeśli True, pokazuje tylko sloty, w których
                         jest przynajmniej jeden obraz.
    """
    slots = time_records["slots"]
    if not slots:
        print("Brak slotów (pusty zestaw danych).")
        return

    # Wyznacz listę kamer, jeśli nie podano
    if cams is None:
        cam_set = set()
        for slot in slots:
            cam_set.update(slot["frames"].keys())
        cams = sorted(cam_set)

    # Nagłówek tabeli
    header = ["slot", "t_rel_ms"]
    for cam in cams:
        header.append(f"{cam}_dt[ms]")
    print(" | ".join(f"{h:>15}" for h in header))
    print("-" * (len(header) * 18))

    num_printed = 0
    for slot in slots:
        if show_only_nonempty and not slot["frames"]:
            continue

        if max_rows is not None and num_printed >= max_rows:
            print(f"... (obcięto dalsze wiersze, pokazano {max_rows})")
            break

        row = [
            f"{slot['slot_idx']:>15d}",
            f"{slot['t_rel_ms']:>15.3f}",
        ]

        for cam in cams:
            frame = slot["frames"].get(cam)
            if frame is None:
                cell = "–"
            else:
                # dt_ms z dokładnością do 3 miejsc
                cell = f"{frame['dt_ms']:.3f}"
            row.append(f"{cell:>15}")

        print(" | ".join(row))
        num_printed += 1

# GRAFIKI ---------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional


def plot_camera_availability_heatmap(
    time_records: Dict[str, Any],
    cams: Optional[List[str]] = None,
    max_slots: Optional[int] = None,
) -> None:
    """
    Heatmapa dostępności obrazów z kamer w kolejnych slotach czasowych.

    time_records – wynik build_time_records(...)
    cams         – kolejność kamer w osi Y; jeśli None, używa wszystkich z danych
    max_slots    – maksymalna liczba slotów do pokazania (od początku), None = wszystkie
    """
    slots = time_records["slots"]
    if not slots:
        print("Brak slotów do wizualizacji.")
        return

    # ograniczenie liczby slotów (opcjonalnie)
    if max_slots is not None:
        slots = slots[:max_slots]

    # wyznacz listę kamer, jeśli nie podano
    if cams is None:
        cam_set = set()
        for slot in slots:
            cam_set.update(slot["frames"].keys())
        cams = sorted(cam_set)

    num_cams = len(cams)
    num_slots = len(slots)

    # macierz [cam_idx, slot_idx] = 1 jeśli jest obraz z danej kamery w slocie, else 0
    mat = np.zeros((num_cams, num_slots), dtype=np.int32)

    for j, slot in enumerate(slots):
        for i, cam in enumerate(cams):
            if cam in slot["frames"]:
                mat[i, j] = 1

    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_xlabel("slot czasowy (kolejny indeks)")
    ax.set_ylabel("kamera")

    # opisy osi Y – nazwy kamer
    ax.set_yticks(np.arange(num_cams))
    ax.set_yticklabels(cams)

    # trochę mniej etykiet na osi X, np. co 10 slotów
    step = max(1, num_slots // 10)
    ax.set_xticks(np.arange(0, num_slots, step))
    ax.set_xticklabels(np.arange(0, num_slots, step))

    ax.set_title("Dostępność obrazów z kamer w kolejnych slotach czasowych")
    plt.colorbar(im, ax=ax, label="obecność obrazu (1=jest, 0=brak)")
    plt.tight_layout()
    plt.show()

def plot_dt_per_camera(
    time_records: Dict[str, Any],
    cams: Optional[List[str]] = None,
    max_slots: Optional[int] = None,
) -> None:
    """
    Wykres dt_ms (różnica czasowa względem nominalnego czasu slotu) dla każdej kamery.

    Dla każdej kamery rysujemy wykres:
        X – indeks slotu
        Y – dt_ms (ms)
    """
    slots = time_records["slots"]
    if not slots:
        print("Brak slotów do wizualizacji.")
        return

    if max_slots is not None:
        slots = slots[:max_slots]

    # lista kamer
    if cams is None:
        cam_set = set()
        for slot in slots:
            cam_set.update(slot["frames"].keys())
        cams = sorted(cam_set)

    fig, ax = plt.subplots()

    for cam in cams:
        xs = []
        ys = []
        for slot in slots:
            if cam in slot["frames"]:
                xs.append(slot["slot_idx"])
                ys.append(slot["frames"][cam]["dt_ms"])
        if xs:
            ax.plot(xs, ys, marker="o", linestyle="-", label=cam)

    ax.set_xlabel("slot czasowy (indeks)")
    ax.set_ylabel("dt_ms względem nominalnego czasu slotu [ms]")
    ax.set_title("Różnice czasowe (dt_ms) względem nominalnych slotów")
    ax.axhline(0.0, linestyle="--")  # linia 0 ms
    # tolerancja z time_records (jeśli jest)
    tol = time_records.get("tolerance_ms", None)
    if tol is not None:
        ax.axhline(tol, linestyle=":", label=f"+tolerancja ({tol} ms)")
        ax.axhline(-tol, linestyle=":", label=f"-tolerancja ({tol} ms)")

    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_num_cams_per_slot(
    time_records: Dict[str, Any],
    max_slots: Optional[int] = None,
) -> None:
    """
    Wykres liczby dostępnych kamer (z obrazem) w każdym slocie czasowym.

    X – indeks slotu
    Y – liczba kamer w danym slocie (0..4)
    """
    slots = time_records["slots"]
    if not slots:
        print("Brak slotów do wizualizacji.")
        return

    if max_slots is not None:
        slots = slots[:max_slots]

    xs = [slot["slot_idx"] for slot in slots]
    ys = [len(slot["frames"]) for slot in slots]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("slot czasowy (indeks)")
    ax.set_ylabel("liczba kamer z obrazem [szt.]")
    ax.set_title("Liczba dostępnych kamer w kolejnych slotach czasowych")
    ax.set_ylim(-0.2, max(4, max(ys) + 0.5))
    ax.grid(True)
    plt.tight_layout()
    plt.show()










# ===============================================================
import re
from typing import List, Tuple, Dict
from pathlib import Path

def cam_batch_timestamps(folder):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"{folder!r} nie jest katalogiem")

    webp_files = sorted(folder_path.glob("*.webp"))
    if not webp_files:
        raise ValueError(f"W katalogu {folder!r} nie znaleziono plików .webp")

    # Wzorzec: dowolny prefix, potem blok 4 cyfr (indeks), '_ts_ns_', timestamp, '.webp'
    # Przykład dopasowania:
    #   CENTER_L_19700101_033916_0000_0000_ts_ns_9556873601304.webp
    #                                         ^^^^ index  ^^^^^ ts_ns
    name_pattern = re.compile(r".*_(\d{4})_ts_ns_(\d+)\.webp$", re.IGNORECASE)

    records: List[Tuple[int, int, Path]] = []

    for p in webp_files:
        m = name_pattern.match(p.name)
        if not m:
            # Jeśli trafi się plik o innym formacie nazwy – pomijamy (lub możesz dać warning)
            # print(f"Pomijam plik o niepasującej nazwie: {p.name}")
            continue

        idx_str, ts_str = m.groups()
        idx = int(idx_str)
        ts_ns = int(ts_str)
        records.append((idx, ts_ns, p))

    if not records:
        raise ValueError("Nie udało się wyciągnąć indeksów i znaczników czasu z żadnego pliku.")

    # Sortowanie po indeksie
    records.sort(key=lambda r: r[0])

    # print('---')
    # for r in records:
    #     print(r)
    # print('---')
    #
    # indices = [r[0] for r in records]
    # ts_ns = [r[1] for r in records]
    return records


import json
from pathlib import Path
from typing import Dict, Any


def _write_conversion_info_json(
    out_root: Path,
    conversion_info: Dict[str, Any],
    filename: str = "info_conversion_export.json",
) -> Path:
    """
    Zapisuje zebrane informacje o konwersji do pliku JSON w katalogu out_root.

    Struktura conversion_info powinna zawierać m.in.:
        - meta (period, tolerance, t0_ns, kamery, itp.)
        - batches -> lista paczek, w każdej:
            - batch_idx, batch_id_str, batch_dir
            - slots -> lista slotów:
                - global_slot_idx, local_slot_idx, t_nominal_ns, t_rel_ms
                - frames -> per kamera:
                    - Raw_set
                    - Learning_set
    """
    out_path = out_root / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conversion_info, f, indent=2, ensure_ascii=False)
    return out_path


from pathlib import Path
from typing import Dict, List, Any, Optional

import shutil
from PIL import Image
import sys
import math
import json  # jeśli jeszcze nie było


def _write_conversion_info_json(
    out_root: Path,
    conversion_info: Dict[str, Any],
    filename: str = "info_conversion_export.json",
) -> Path:
    out_path = out_root / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conversion_info, f, indent=2, ensure_ascii=False)
    return out_path


def export_batches_from_time_records_v3(
    time_records: Dict[str, Any],
    out_root: str | Path,
    N: int,
    cams: Optional[List[str]] = None,
    copy_real_images: bool = True,
    quality: int = 50,
    method: int = 6,
    show_progress: bool = True,
    progress_update_every: int = 200,
) -> Dict[str, Any]:
    """
    SZYBSZA wersja eksportu paczek z time_records + zapis metadanych do JSON.

    Kluczowe optymalizacje:
        - Prawdziwe obrazy (.webp) są domyślnie TYLKO KOPIOWANE (shutil.copy2),
          bez ponownego otwierania i kompresji przez PIL.
        - Czarny placeholder dla brakujących obrazów jest zakodowany do WEBP
          RAZ na kamerę, a potem tylko kopiowany jako plik.

    Dodatkowo:
        - Tworzy plik info_conversion_export.json w out_root zawierający
          szczegółową informację o:
            - slotach,
            - batchach B_xxx,
            - mapowaniu RAW -> LEARNING (nazwy przed/po, czasy, dt, placeholdery).

    Parametry:
        time_records    – wynik build_time_records(...)
        out_root        – katalog wyjściowy
        N               – liczba NIEPUSTYCH slotów na paczkę
        cams            – lista kamer; jeśli None, wykrywana z danych
        copy_real_images – jeśli True: prawdziwe obrazy są tylko kopiowane;
                           jeśli False: re-kodowanie przez PIL
        quality, method – parametry kompresji WEBP dla CZARNYCH placeholderów
                          oraz dla ewentualnego re-kodowania, jeśli copy_real_images=False
        show_progress   – czy pokazywać pasek postępu
        progress_update_every – co ile plików odświeżać pasek postępu

    Zwraca:
        summary = {
            "total_non_empty_slots": int,
            "N": int,
            "num_batches": int,
            "batches": [...],
            "conversion_info_json": Path,  # ścieżka do pliku JSON
        }
    """
    if N <= 0:
        raise ValueError("N musi być > 0")

    slots = time_records.get("slots", [])
    if not slots:
        raise ValueError("time_records['slots'] jest puste – brak danych do eksportu.")

    # 1) Lista kamer
    if cams is None:
        cam_set = set()
        for slot in slots:
            cam_set.update(slot["frames"].keys())
        cams = sorted(cam_set)

    if not cams:
        raise ValueError("Nie znaleziono żadnych kamer w time_records.")

    # 2) Filtrujemy tylko sloty niepuste (tam, gdzie jest co najmniej jedna klatka)
    non_empty_slots = [slot for slot in slots if slot["frames"]]
    total_non_empty = len(non_empty_slots)
    if total_non_empty == 0:
        raise ValueError("Brak niepustych slotów – nie ma czego eksportować.")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 3) Wyznacz rozmiary obrazów dla każdej kamery
    size_by_cam: Dict[str, tuple[int, int]] = {}
    for cam in cams:
        for slot in slots:
            frame = slot["frames"].get(cam)
            if frame is not None:
                with Image.open(frame["path"]) as im:
                    size_by_cam[cam] = im.size  # (width, height)
                break
        if cam not in size_by_cam:
            size_by_cam[cam] = (1920, 1080)  # fallback

    # 4) Szablonowe czarne WEBP (jeden plik na kamerę) – potem tylko kopiujemy
    black_template_dir = out_root / "_black_templates"
    black_template_dir.mkdir(parents=True, exist_ok=True)

    black_template_paths: Dict[str, Path] = {}
    for cam, (w, h) in size_by_cam.items():
        black_img = Image.new("RGB", (w, h), color=(0, 0, 0))
        tmpl_path = black_template_dir / f"{cam}_black.webp"
        black_img.save(
            tmpl_path,
            format="WEBP",
            quality=quality,
            method=method,
        )
        black_template_paths[cam] = tmpl_path

    # 5) Podział na paczki po N niepustych slotów
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    batches = chunk_list(non_empty_slots, N)

    # 6) Pasek postępu
    def _print_progress(prefix: str, current: int, total: int, bar_length: int = 30):
        if total <= 0:
            return
        frac = current / total
        filled = int(bar_length * frac)
        bar = "#" * filled + "-" * (bar_length - filled)
        percent = frac * 100.0
        sys.stdout.write(f"\r{prefix} [{bar}] {percent:6.2f}% ({current}/{total})")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    batches_info: List[Dict[str, Any]] = []

    # --- NOWOŚĆ: struktura na dane do JSON ---
    conversion_info: Dict[str, Any] = {
        "meta": {
            "period_ms": time_records.get("period_ms", None),
            "tolerance_ms": time_records.get("tolerance_ms", None),
            "t0_ns": time_records.get("t0_ns", None),
            "cameras": cams,
        },
        "total_non_empty_slots": total_non_empty,
        "N": N,
        "batches": [],  # wypełnimy poniżej
    }

    for batch_idx, batch_slots in enumerate(batches):
        batch_id_str = f"{batch_idx:03d}"  # np. 000, 001, ...
        batch_dir = out_root / f"B_{batch_id_str}"

        print(f"[SORT] B{batch_id_str} - Starting to copy")

        # Utwórz katalogi dla każdej kamery
        for cam in cams:
            (batch_dir / cam).mkdir(parents=True, exist_ok=True)

        num_slots_in_batch = len(batch_slots)
        files_in_batch = num_slots_in_batch * len(cams)
        processed_files = 0

        # --- NOWOŚĆ: info o tej paczce do JSON ---
        batch_info_json: Dict[str, Any] = {
            "batch_idx": batch_idx,
            "batch_id_str": batch_id_str,
            "batch_dir": str(batch_dir),
            "num_slots": num_slots_in_batch,
            "slots": [],  # wypełnimy slotami
        }

        # Przetwarzanie slotów w paczce
        for local_slot_idx, slot in enumerate(batch_slots):
            slot_str = f"{local_slot_idx:03d}"  # 000..N-1 w obrębie paczki

            # info o slocie do JSON
            slot_info_json: Dict[str, Any] = {
                "global_slot_idx": slot["slot_idx"],
                "local_slot_idx": local_slot_idx,
                "t_nominal_ns": slot["t_nominal_ns"],
                "t_rel_ms": slot["t_rel_ms"],
                "frames": {},  # per kamera
            }

            for cam in cams:
                out_name = f"{cam}_{batch_id_str}_{slot_str}.webp"
                out_path = batch_dir / cam / out_name
                frame = slot["frames"].get(cam)

                placeholder = False
                raw_info = {
                    "path": None,
                    "filename": None,
                    "ts_ns": None,
                    "dt_ms": None,
                    "within_tolerance": None,
                }

                if frame is not None:
                    # Raw_sets info
                    src_path = Path(frame["path"])
                    raw_info = {
                        "path": str(src_path),
                        "filename": src_path.name,
                        "ts_ns": frame["ts_ns"],
                        "dt_ms": frame["dt_ms"],
                        "within_tolerance": frame["within_tolerance"],
                    }

                    # Kopiowanie / rekodowanie
                    if copy_real_images:
                        try:
                            shutil.copy2(src_path, out_path)
                        except Exception:
                            # awaryjnie placeholder
                            shutil.copy2(black_template_paths[cam], out_path)
                            placeholder = True
                    else:
                        try:
                            with Image.open(src_path) as im:
                                im = im.convert("RGB")
                                im.save(
                                    out_path,
                                    format="WEBP",
                                    quality=quality,
                                    method=method,
                                )
                        except Exception:
                            shutil.copy2(black_template_paths[cam], out_path)
                            placeholder = True
                else:
                    # brak obrazu z tej kamery w tym slocie – placeholder
                    shutil.copy2(black_template_paths[cam], out_path)
                    placeholder = True

                # Learning_sets info
                learning_info = {
                    "path": str(out_path),
                    "filename": out_path.name,
                    "is_placeholder": placeholder,
                }

                # zapis do struktury slota
                slot_info_json["frames"][cam] = {
                    "camera": cam,
                    "Raw_set": raw_info,
                    "Learning_set": learning_info,
                }

                processed_files += 1
                if show_progress and (
                    processed_files % progress_update_every == 0
                    or processed_files == files_in_batch
                ):
                    _print_progress(f"[SORT] B{batch_id_str}", processed_files, files_in_batch)

            # dodaj slot do paczki
            batch_info_json["slots"].append(slot_info_json)

        if show_progress and processed_files == files_in_batch:
            _print_progress(f"[SORT] B{batch_id_str}", files_in_batch, files_in_batch)

        print(f"[SORT] B{batch_id_str} - Done ({num_slots_in_batch} slots, {files_in_batch} files)")

        batches_info.append(
            {
                "batch_idx": batch_idx,
                "batch_id_str": batch_id_str,
                "batch_dir": batch_dir,
                "num_slots": num_slots_in_batch,
            }
        )

        # dodaj info o paczce do conversion_info
        conversion_info["batches"].append(batch_info_json)

    # --- zapis JSON z informacjami o konwersji ---
    json_path = _write_conversion_info_json(out_root, conversion_info)

    summary = {
        "total_non_empty_slots": total_non_empty,
        "N": N,
        "num_batches": len(batches_info),
        "batches": batches_info,
        "conversion_info_json": json_path,
    }

    print(
        f"Wyeksportowano {total_non_empty} niepustych slotów "
        f"w {len(batches_info)} paczkach po N={N} (ostatnia może być krótsza)."
    )
    print(f"Informacje o konwersji zapisano w: {json_path}")

    return summary


# ===============================================================================
# ===============================================================================

import json
from pathlib import Path
from typing import List, Optional, Dict, Any


def print_dt_table_from_conversion_json(
    json_path: str | Path,
    cams: Optional[List[str]] = None,
    max_rows: Optional[int] = 50,
    show_only_nonempty: bool = False,
) -> None:
    """
    Czyta info_conversion_export.json i wypisuje tabelę:

        slot | t_rel_ms | batch | local_idx | CAM1_dt[ms] | CAM2_dt[ms] | ...

    z dodaną informacją:
        - w którym batchu (B_xxx) jest dany slot,
        - jaki ma lokalny indeks w tym batchu.

    Parametry:
        json_path        – ścieżka do info_conversion_export.json
        cams             – lista kamer w kolejności kolumn; jeśli None, używa tych z metadata
        max_rows         – maks. liczba wierszy do wyświetlenia (None = wszystkie)
        show_only_nonempty – jeśli True: pokazuje tylko sloty, w których
                             jest przynajmniej jedna kamera z dt_ms != None
    """
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    if cams is None:
        cams = meta.get("cameras", [])
    if not cams:
        # fallback: wykryj kamery z pierwszego slota
        cams_set = set()
        for batch in data.get("batches", []):
            for slot in batch.get("slots", []):
                cams_set.update(slot.get("frames", {}).keys())
            if cams_set:
                break
        cams = sorted(cams_set)

    rows: List[Dict[str, Any]] = []

    # Zbierz wszystkie wiersze (sloty) ze wszystkich batchy
    for batch in data.get("batches", []):
        batch_id_str = batch["batch_id_str"]  # np. "000"
        batch_label = f"B_{batch_id_str}"
        for slot in batch.get("slots", []):
            global_slot_idx = slot["global_slot_idx"]
            local_slot_idx = slot["local_slot_idx"]
            t_rel_ms = slot["t_rel_ms"]
            frames = slot.get("frames", {})

            # dt_ms dla każdej kamery (None jeśli brak)
            dt_by_cam: Dict[str, Optional[float]] = {}
            for cam in cams:
                frame_info = frames.get(cam)
                if frame_info is None:
                    dt_by_cam[cam] = None
                else:
                    raw = frame_info.get("Raw_set", {})
                    dt_by_cam[cam] = raw.get("dt_ms", None)

            if show_only_nonempty:
                # sprawdź, czy jest jakaś kamera z dt_ms != None
                if all(dt_by_cam[cam] is None for cam in cams):
                    continue

            rows.append(
                {
                    "slot": global_slot_idx,
                    "t_rel_ms": t_rel_ms,
                    "batch": batch_label,
                    "local_idx": local_slot_idx,
                    "dt_by_cam": dt_by_cam,
                }
            )

    # Posortuj po globalnym indeksie slota
    rows.sort(key=lambda r: r["slot"])

    # Drukowanie tabeli
    header = ["slot", "t_rel_ms", "batch", "local_idx"] + [f"{cam}_dt[ms]" for cam in cams]
    print(" | ".join(f"{h:>15}" for h in header))
    print("-" * (len(header) * 18))

    count = 0
    for row in rows:
        if max_rows is not None and count >= max_rows:
            print(f"... (obcięto dalsze wiersze, pokazano {max_rows})")
            break

        base_cols = [
            f"{row['slot']:>15d}",
            f"{row['t_rel_ms']:>15.3f}",
            f"{row['batch']:>15}",
            f"{row['local_idx']:>15d}",
        ]

        dt_cols = []
        for cam in cams:
            dt = row["dt_by_cam"][cam]
            if dt is None:
                cell = "–"
            else:
                cell = f"{dt:.3f}"
            dt_cols.append(f"{cell:>15}")

        print(" | ".join(base_cols + dt_cols))
        count += 1

def seconds_to_hms(sec):
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return hours, minutes, seconds


def prepare_structure_from_json(json_path):
    """
    Wczytuje konfigurację z pliku JSON, buduje strukturę ret
    oraz tworzy katalogi wyjściowe (out) jeśli jeszcze nie istnieją.
    """

    # --- wczytanie pliku JSON ---
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    folders = cfg.get("folders", [])
    out_dir = Path(cfg.get("out_dir", "")).resolve()
    cores = cfg.get("cores", None)
    quality = cfg.get("quality", 1)
    method = cfg.get("method", 1)

    # --- lista folderów wejściowych ---
    list_in = [str(Path(folder).resolve()) for folder in folders]

    # --- lista folderów wyjściowych ---
    list_out = []
    for folder in list_in:
        name = Path(folder).name  # ostatnia część ścieżki
        out_path = out_dir / name
        list_out.append(str(out_path))

        # --- automatyczne tworzenie katalogu wyjściowego ---
        out_path.mkdir(parents=True, exist_ok=True)

    # --- ustawianie liczby workerów ---
    if cores is None:
        workers = None
    else:
        max_cores = multiprocessing.cpu_count()
        workers = max(1, min(int(cores), max_cores))

    # --- wynik ---
    ret = {
        "in": list_in,
        "out": list_out,
        "workers": workers,
        "quality": quality,
        "method": method,
    }

    return ret

# ==========================
#  TESTY RĘCZNE (opcjonalne)
# ==========================

if __name__ == "__main__":
    # Przykład użycia funkcji:
    bin_path = r""
    out_dir = r""
    export_single_bin_to_webp(bin_path, out_dir)

