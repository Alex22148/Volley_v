# VolleyHub — plan wdrożenia po aktualizacji projektu

## Status po szybkim audycie

Sprawdzone na dostarczonej paczce `VolleyHub_enterprice.7z`:

- `python -m py_compile ...` — OK, pliki `.py` kompilują się.
- `python -m pytest -q --import-mode=importlib tests` przed patchami — `21 passed, 11 skipped, 4 failed`.
- Po patchach P0 — `25 passed, 11 skipped`.

Aktualna struktura jest już bliska docelowej, bo istnieją osobne obszary:

```text
src/runtime_benchmark/
src/runtime_gpu/
src/runtime_gui/
src/runtime_inference/
src/runtime_production/
src/runtime_sources/
src/runtime_diagnostics/
```

Największy problem architektoniczny nadal pozostaje w `gui.py`: główne GUI importuje bezpośrednio bardzo dużo funkcji z `ui.dir_gui.benchmark`. To trzeba odciąć w kolejnym kroku, ale nie robić tego przed naprawą P0, bo najpierw warto mieć zielone testy produkcyjnego fastpath.

---

## Kolejność wdrożenia

### P0 — stabilizacja testów i hot-path live

Cel: projekt ma przechodzić testy, a live loop nie może resetować stanu timestampów przy każdej klatce.

Patche:

1. `patches/src_runtime_production_fast_path_config.patch`
2. `patches/src_runtime_production_preview_worker.patch`
3. `patches/live_runtime_live_backend_controller.patch`

Efekt:

- walidacja `FastPathConfig` najpierw sprawdza kontrakt logiczny, dopiero potem plik `.engine`;
- `PreviewWorker` działa z aktualnym `SharedMemoryManager` i prostymi stubami testowymi;
- preview low-quality jest szybszy i nie blokuje testu publikacji ramki;
- `LiveBackendController` nie resetuje `_last_ts_seen_by_role` w każdej iteracji.

Test po P0:

```powershell
python -m py_compile src/runtime_production/fast_path_config.py src/runtime_production/preview_worker.py live_runtime/live_backend_controller.py
python -m pytest -q --import-mode=importlib tests
```

Oczekiwane:

```text
25 passed, 11 skipped
```

---

### P1 — oddzielenie benchmarku od GUI

Cel: benchmark ma być modułem runtime, a GUI tylko launcherem/klientem.

Docelowo:

```text
src/runtime_benchmark/          # logika benchmarku
src/runtime_gui/                # okno/launcher benchmarku
ui/dir_gui/benchmark.py         # docelowo legacy albo adapter
```

Zmiana w `gui.py` powinna iść w stronę usunięcia importów typu:

```python
from ui.dir_gui.benchmark import start_auto_benchmark, stop_auto_benchmark, ...
```

i zastąpienia ich jednym adapterem, np.:

```python
from src.runtime_gui.runtime_benchmark_gui import open_runtime_benchmark_window
```

albo launcherem subprocess:

```python
subprocess.Popen([
    sys.executable,
    "-m",
    "src.runtime_benchmark.run_4cam_raw_inference_benchmark",
    "--config", str(config_path),
])
```

Ważne: główne GUI nie powinno znać szczegółów benchmarku, percentile, sweepów, wariantów batcha, source mode itd. Ono może tylko:

- otworzyć okno benchmarku,
- uruchomić proces benchmarku,
- pokazać ostatni raport,
- porównać ostatnie wyniki.

---

### P2 — wspólny kontrakt GPU image processing

Cel: benchmark i aplikacja produkcyjna mają używać tego samego GPU image processingu.

Proponowany nowy plik:

```text
src/runtime_gpu/gpu_preprocess_pipeline.py
```

Docelowy kontrakt:

```python
@dataclass(slots=True)
class GpuPacket:
    tensor: torch.Tensor          # CUDA, BCHW, RGB, normalized
    roles: tuple[str, ...]
    timestamps_ns: tuple[int, ...]
    stage_ms: dict[str, float]

class GpuPreprocessPipeline:
    def process_raw_packet(self, raw_frames, roles, timestamps_ns) -> GpuPacket:
        ...
```

Zasada:

```text
RAW Bayer -> GPU debayer -> GPU resize -> GPU normalize -> TensorRT
```

Bez powrotu do CPU między debayerem i inferencją.

---

### P3 — produkcyjny fastpath jako tryb aplikacji

Cel: aplikacja może działać w dwóch trybach:

```text
safe path: CPU / Ultralytics / prostszy debug
fast path: RAW Bayer -> GPU processing -> TensorRT batch=4
```

Środowisko uruchomieniowe:

```powershell
$env:VOLLEYHUB_FAST_PATH="1"
$env:VOLLEYHUB_TRT_ENGINE="artifacts/tensorrt_exports/best__fp16_640_b4_static.engine"
$env:VOLLEYHUB_YOLO_BATCH="4"
$env:VOLLEYHUB_YOLO_IMGSZ="640"
$env:VOLLEYHUB_COLOR_BACKEND="native_cuda_npp"
python main.py
```

Aplikacja powinna jasno raportować:

```text
FASTPATH READY
FASTPATH FALLBACK: reason=...
FASTPATH DISABLED
```

---

### P4 — raport diagnostyczny i smoke tests użytkowe

Dodać jeden skrypt wejściowy:

```text
tools/volleyhub_smoke_check.py
```

Powinien sprawdzać:

- importy krytyczne,
- dostępność CUDA,
- dostępność TensorRT,
- istnienie engine,
- zgodność batch/imgsz,
- dostępność pypylon,
- start `FastPathConfig.validate()`,
- start benchmarku synthetic bez kamer.

Wynik powinien być czytelny dla operatora/developera:

```text
[OK] GUI imports
[OK] Runtime benchmark imports
[WARN] pypylon unavailable — camera mode disabled
[OK] CUDA available: RTX ...
[FAIL] TensorRT engine missing: ...
```

---

## Minimalna definicja sukcesu

Po wdrożeniu P0–P2:

```text
1. Projekt przechodzi testy.
2. GUI może startować bez benchmarku w hot-path.
3. Benchmark działa jako oddzielny moduł.
4. GPU processing jest wspólny dla benchmarku i production fastpath.
5. Preview jest osobnym, lekkim workerem i nie blokuje inferencji.
```

Po P3:

```text
4 kamery / synthetic packet -> GPU image processing -> TensorRT batch=4 -> wynik -> GUI
```

