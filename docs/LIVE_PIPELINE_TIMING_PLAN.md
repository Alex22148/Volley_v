# Live Pipeline Timing — Wdrożeniowy plan i procedura aktywacji

Cel: zebrać per-stage timingi z faktycznego, działającego live pipeline'u, żeby zdecydować, którą ze ścieżek A/B/C/D/E uruchomić jako optymalizację. Ten dokument opisuje minimalny patch, który już został wdrożony, oraz krok po kroku jak go aktywować, zinterpretować dane i ostatecznie cofnąć.

## Faza 1 (wdrożona — przedmiot tej iteracji)

Probe diagnostyczny + aggregator, OFF by default. Aktywuje się jednym env varem; produkcja bez tego env vara zachowuje się dokładnie tak jak przed patchem.

### Pliki dodane

- `src/runtime_diagnostics/__init__.py`
- `src/runtime_diagnostics/live_timing_probe.py` — probe + rolling window p95
- `src/runtime_diagnostics/build_live_pipeline_report.py` — generator markdown
- `reports/live_pipeline_timing_report.md` — raport (sekcje benchmark gotowe, live placeholder)
- `docs/LIVE_PIPELINE_TIMING_PLAN.md` — ten dokument

### Pliki produkcyjne zmodyfikowane (minimalny diff)

1. **`live_runtime/live_backend_controller.py`**
   - +13 linii w głowie pliku: defensywny import probe + no-op fallback.
   - +24 linie w `_run_loop` pod blokiem `frames_processed`: `_probe.record_packet(stage_ms={...})` z istniejącymi zmiennymi `grab_ms`, `convert_ms`, `preprocess_ms`, `infer_ms`, `track_ms`, `total_pipeline_ms`. Cały blok w try/except. **Zero zmian w logice pipeline'u.**

2. **`vision/vision_ring_buffer_v3_nonblocking.py`**
   - +9 linii: defensywny import probe + no-op fallback w głowie pliku.
   - +11 linii w `push()`: pomiar `buffer_push_ms` po linii `with self.lock`. Mierzy tylko czas ścieżki gorącej.

Zero zmian w `gui.py`, `vision/yolo_module.py`, `capture/*`, `main.py`. TrainManager, DatasetValidator, QUICK/FULL nietknięte.

### Co probe loguje

Co `VOLLEYHUB_LIVE_PROBE_EVERY` paczek (default 30) probe emituje **jedną** zwięzłą linię na proces, np.:

```
[LIVE_PACKET_TIMING] tag=live_backend frames=30 window=30 role=CENTER_L resolution=2464x2056 color_backend=cpu inference_backend=ultralytics batch=1 yolo_enabled=True median_ms=cpu_debayer_ms=29.40,grab_ms=2.10,inference_ms=12.10,postprocess_ms=0.42,preprocess_ms=0.31,total_packet_ms=46.32 p95_ms=cpu_debayer_ms=49.20,grab_ms=4.30,inference_ms=14.50,postprocess_ms=0.81,preprocess_ms=0.55,total_packet_ms=72.80 drops=0
```

Linia idzie do:
- standard logger (`logging.INFO`, prefiks `[LIVE_PACKET_TIMING]`)
- `logs/live_probe_<tag>_pid<pid>.log` (per proces, append-only)

### Aktywacja

```powershell
# 1) ustaw env var PRZED uruchomieniem produkcji
$env:VOLLEYHUB_LIVE_PROBE = "1"
$env:VOLLEYHUB_LIVE_PROBE_EVERY = "30"          # opcjonalnie: log co N paczek
$env:VOLLEYHUB_LIVE_PROBE_WINDOW = "200"        # opcjonalnie: rozmiar rolling window
$env:VOLLEYHUB_LIVE_PROBE_LOGDIR = "logs"       # opcjonalnie: katalog logów

# 2) uruchom produkcję jak zwykle
python main.py
# pozwól systemowi działać 30-60 sekund pod typowym obciążeniem

# 3) zatrzymaj produkcję
# 4) wygeneruj raport
python -m src.runtime_diagnostics.build_live_pipeline_report
# raport: reports/live_pipeline_timing_report.md
```

### Co przeanalizować po pierwszym runie

Patrz na sekcję `Per-stage breakdown (live data)` w `reports/live_pipeline_timing_report.md`. Pytania kontrolne:

1. **Czy `cpu_debayer_ms` median > 25 ms?** Jeśli tak — produkcja faktycznie idzie torem CPU debayer i to jest dominujący koszt. Decyzja: faza 2.
2. **Czy `inference_ms` rośnie z cyklem czy jest stabilny?** Niestabilność wskazuje na konflikt GPU/CPU lub torch/TRT cold-start.
3. **Czy `total_packet_ms` p95 > 1.5 × median?** Wysokie p95 = długie ogony. Częsta przyczyna: lock contention w VisionRingBuffer, `buffer_push_ms` ogony, GIL podczas debayera CPU.
4. **Czy `grab_ms` jest zaskakująco wysoki?** Jeśli tak — Basler buffer pool / GenICam timeout dominuje, a nie samo przetwarzanie.

### Dezaktywacja

```powershell
$env:VOLLEYHUB_LIVE_PROBE = ""
```

Lub po prostu uruchom produkcję bez tego env vara. Probe wtedy używa singletonu `_NoopProbe`, który ma metody `record_packet`, `record_drop`, `time_block` zamienione na natychmiastowy return.

### Cofnięcie patcha (jeśli kiedykolwiek będzie potrzebne)

Każdy hook jest oznaczony komentarzem `# Diagnostic probe (...)` lub `# --- Diagnostic probe ---`. Wystarczy:

1. Usunąć blok `try: from src.runtime_diagnostics... except: ... def _get_live_probe(...)` w głowach plików.
2. Usunąć wszystkie bloki oznaczone komentarzem `Diagnostic probe`.
3. Usunąć katalog `src/runtime_diagnostics/`.

Zero efektów ubocznych poza brakiem probe.

## Faza 2 (NIE wdrożona — opcjonalna, decyzja po fazie 1)

Jeśli faza 1 pokaże, że dominuje `cpu_debayer_ms` lub `grab_ms`, ALE chcemy zobaczyć dokładnie który podstage ich ucierpi (np. czy debayer CPU dzieli rdzeń z grabberem, czy queue_wait jest zatkany), wtedy dodać:

### 2a. `capture/grabber_module.py`

Po linii `result = camera.RetrieveResult(...)` w pętli grab:
```python
try:
    from src.runtime_diagnostics.live_timing_probe import get_probe as _gp
    _p = _gp("grabber")
    if _p.enabled:
        _p.record_packet(stage_ms={"grab_ms": grab_elapsed_ms, "queue_push_ms": push_elapsed_ms},
                         role=role, camera_idx=cam_idx)
except Exception:
    pass
```

### 2b. `capture/color_convert_process.py`

Wokół `q.get()` (queue_wait) i wokół `cv2.cvtColor(...)` (cpu_debayer_ms) i wokół preview_publish:
```python
try:
    from src.runtime_diagnostics.live_timing_probe import get_probe as _gp
    _p = _gp("color_convert")
except Exception:
    _p = None

# w pętli:
t_qw = time.perf_counter()
item = q.get(timeout=...)
queue_wait_ms = (time.perf_counter() - t_qw) * 1000.0

t_db = time.perf_counter()
bgr = cv2.cvtColor(bayer, code)
cpu_debayer_ms = (time.perf_counter() - t_db) * 1000.0

# preview publish:
t_pp = time.perf_counter()
# ... publish to preview queue ...
preview_publish_ms = (time.perf_counter() - t_pp) * 1000.0

if _p is not None and _p.enabled:
    _p.record_packet(stage_ms={
        "queue_wait_ms": queue_wait_ms,
        "cpu_debayer_ms": cpu_debayer_ms,
        "preview_publish_ms": preview_publish_ms,
    })
```

Patche fazy 2 to po ~10 linii per plik, też w try/except, też off-by-default.

## Faza 3 — decyzja optymalizacyjna

Dopiero **po** zebraniu danych z fazy 1 (i ewentualnie 2):

| Wybór | Warunek wyzwalający                                        | Spodziewana poprawa                          | Koszt wdrożenia |
|-------|------------------------------------------------------------|-----------------------------------------------|-----------------|
| **A)** Reduce resolution (np. 1920×1080) | `total_packet_ms` median > 30 ms i `cpu_debayer_ms` dominuje | -30..-50% kosztu color stage | Niski — config kamery |
| **B)** GPU debayer (CUDA/NPP/C++) | `cpu_debayer_ms` median > 25 ms i resolution musi zostać native | -70..-90% kosztu debayera | Wysoki — natywny adapter |
| **C)** Pełne gpu_zero_copy (BCHW float CUDA → TRT) | `inference_ms` ma duży stały overhead z preprocess Ultralytics | -3..-5 ms na każdej inferencji + eliminacja round-tripu | Średni — przepiąć inferencję, dodać engine batch=1 native |
| **D)** Niższe `publish_interval_ms` lub wyłączenie GUI publish | `preview_publish_ms` lub blokowanie GUI dominuje p95 | Tylko zmniejszenie ogonów p95 | Trywialny |
| **E)** True 4-cam batch w live pipeline | `inference_ms` × 4 pojedyncze calle > 1 batch=4 call | -30..-50% inferencji w sumie | Średni — zmienić `_resolve_key` na grupowanie ts_ns |

Decyzja, **który** z A/B/C/D/E wdrażać, zależy od liczb które wyjdą z fazy 1. Bez tych liczb to byłaby premature optimization.

## Kontekst architektoniczny (dla recenzentów)

Probe respektuje produkcyjne ograniczenia z `LIVE_VISION_ARCHITECTURE.md` / memory architektonicznego:

- Nie wprowadza `cv2.cuda` ani DeepStream/DALI w runtime.
- Nie ingeruje w `latest-frame-only` semantykę, nie modyfikuje `shared_state["bayer_key"]`.
- Nie zmienia kolejności / blokad w `_run_loop`.
- Nie wprowadza `time.sleep` ani synchronizacji blokujących między procesami.
- Działa samodzielnie per proces — nie wymaga `multiprocessing.Manager`, brak shared state pomiędzy probe instancjami.
- Nie loguje danych z obrazów (tylko skalary timingowe i metadane konfiguracji).

## Mapa plików (cheat sheet)

```
src/runtime_diagnostics/
  __init__.py
  live_timing_probe.py            # _Probe, _NoopProbe, get_probe, _StageContext
  build_live_pipeline_report.py   # CLI: python -m src.runtime_diagnostics.build_live_pipeline_report

live_runtime/live_backend_controller.py
  ^^^ +37 linii diagnostycznych w try/except (init + jeden hook w _run_loop)

vision/vision_ring_buffer_v3_nonblocking.py
  ^^^ +20 linii diagnostycznych w try/except (init + hook w push())

logs/live_probe_<tag>_pid<pid>.log     # zapisywane przez probe gdy aktywny

reports/live_pipeline_timing_report.md  # generowany przez build_live_pipeline_report
```
