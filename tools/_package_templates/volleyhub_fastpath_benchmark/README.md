# VolleyHub Fast-Path Benchmark — portable runtime package

Minimalna paczka do mierzenia toru:
**RAW Bayer (1920×1080, 4 cam) → native CUDA debayer → resize/normalize na GPU → zero-copy → TensorRT FP16 batch=4**

Paczka odpowiada wprost na dwa pytania:

1. **Przy inferencji Full HD-like (1088×1920) ile FPS per camera daje dany komputer?**
2. **Dla jakiego największego rozmiaru obrazów wejściowych można uzyskać stabilne 50 FPS per camera?**

Cel: 50 FPS per camera = 1 packet (4 obrazy) na 20 ms. PASS safe = `packet_ms_p95 ≤ 20 ms`.

## Co w środku

- `src/runtime_*` — runtime moduły (sources, gpu, inference, production, diagnostics)
- `scripts/` — sterujące skrypty Python
- `configs/` — config sweepa, baseline RTX 2080 Super, szablon camera_roles
- `artifacts/tensorrt_exports/` — engines TRT FP16 b4 static (jeśli dołączone; inaczej puste)
- `reports/` — tu lądują wyniki
- `logs/` — tu lądują logi probe
- `RUN_ME_FIRST.ps1`, `run_synthetic_sweep.ps1`, `run_real_basler_sweep.ps1` — wrappery PS

## Wymagania

- **Windows 10/11** (skrypty PS), Linux/macOS jest możliwy ale wymaga ręcznego venv
- **NVIDIA GPU + CUDA toolkit** (dla TensorRT i custom CUDA kernel)
- **Visual Studio Build Tools 2019/2022** z C++ workload (do JIT build natywnego debayera)
- **Python 3.10–3.12**
- **TensorRT** w wersji zgodnej z używanym torchem (np. torch 2.5.1+cu121 + TRT 10.x)
- **pypylon + pylon SDK** — TYLKO do trybu basler (synthetic działa bez nich)

Szczegóły instalacji w `requirements_gpu_notes.md`.

## Szybki start na komputerze BEZ kamer

```powershell
# 1) setup venv + sprawdź środowisko
.\RUN_ME_FIRST.ps1

# 2) odpal syntetyczny sweep (60 s × 4 warianty imgsz)
.\run_synthetic_sweep.ps1

# 3) raport otworzy się sam:
# reports/fastpath_comparison_report.md
# reports/fastpath_comparison_report.html
```

## Szybki start na komputerze Z kamerami Basler

```powershell
# 1) setup
.\RUN_ME_FIRST.ps1

# 2) skopiuj swój camera_roles.json do configs/ (template w configs/camera_roles.template.json)
copy camera_roles.json configs\camera_roles.json

# 3) sweep z prawdziwymi kamerami
.\run_real_basler_sweep.ps1
```

## Komendy bez wrapperów PS

```powershell
# synthetic
python scripts\run_fastpath_sweep.py --mode synthetic --duration-s 60

# basler
python scripts\run_fastpath_sweep.py --mode basler --duration-s 60 --camera-roles configs\camera_roles.json

# tylko raport (z istniejących results.json + baseline)
python scripts\build_comparison_report.py --input reports\results.json --baseline configs\baseline_rtx2080super.json
```

## Engine TensorRT — ważne

TensorRT engine **zależy od konkretnego GPU + wersji TRT + sterownika**. Engine zbudowany na innym
komputerze MOŻE nie działać. Paczka rozróżnia trzy sytuacje:

- `engine_reused = true` — engine był na dysku, załadował się czysto
- `engine_built_on_this_machine = true` — engine został świeżo zbudowany tutaj
- `engine_compatibility_warning` — niezerowy gdy zachowuje się dziwnie (np. nie ładuje, mimo że plik jest)
- `status = ENGINE_MISSING` — pliku po prostu nie ma; wariant pomijamy bez crashu

Aby zbudować engines lokalnie:

```powershell
python scripts\export_or_check_engines.py --check          # tylko diagnostyka
python scripts\export_or_check_engines.py --build-missing  # zbuduj brakujące
```

Build wymaga `best.pt` w katalogu paczki (skopiuj swój model). Bez `best.pt` można tylko sprawdzać.

## Jak czytać raport

Raport `reports/fastpath_comparison_report.md` ma sekcje:
- **A. Executive summary** — odpowiedź na 2 pytania + 3 presety (LIVE_50FPS, QUALITY_40FPS, FULLHD_30FPS)
- **B. Hardware comparison** — Twój komputer vs RTX 2080 Super baseline (`speedup_vs_baseline`)
- **C. Resolution sweep** — pełna tabela wszystkich rozmiarów
- **D. Stage breakdown** — gdzie konkretnie ucieka czas (`color_ms`, `inference_ms`, ...)
- **E. Decision answer** — krótkie werdykty
- **F. Environment diagnostics** — OS, CUDA, GPU, VRAM, TRT, pypylon
- **G. Errors / warnings** — co nie poszło, co naprawić

Wykresy w `reports/final_charts/` (PNG, embedowane w HTML i MD).

## Co oznacza wynik

- **640×640 PASS safe** ⇒ komputer nadaje się do **LIVE_50FPS** (4 cam @ 50 FPS, lekka inferencja)
- **960×960 PASS safe** ⇒ **QUALITY_50FPS** (4 cam @ 50 FPS, trochę lepsza jakość)
- **1088×1920 PASS safe** ⇒ **FULLHD_50FPS** (Full HD inference @ 50 FPS — wymaga mocniejszego GPU)
- **1088×1920 daje 30–40 FPS** ⇒ **FULLHD_30FPS** (Full HD ale niższa kadencja)

## Limity tej paczki

- NIE zawiera GUI, treningu, datasetu — to wyłącznie tor benchmarkowy.
- NIE wymaga modeli `.pt` chyba że chcesz przebudować engine.
- NIE wymaga kamer dla synthetic mode.
- W razie braku natywnego CUDA backendu (build extension nie poszedł): wariant kończy się
  `BACKEND_MISSING`, raport pokazuje to czytelnie, nic nie crashuje.
