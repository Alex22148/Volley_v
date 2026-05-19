# Instrukcja dla testera — VolleyHub fast-path benchmark

Cześć! Otrzymałaś/eś tę paczkę żeby zmierzyć **wydajność toru inferencji 4 kamer**
na swoim komputerze. Test trwa ok. **5–10 minut**, potem zwracasz 3 pliki.

Paczka odpowiada na dwa pytania:

1. **Przy inferencji Full HD-like (1088×1920) ile FPS per kamera daje ten komputer?**
2. **Dla jakiego największego rozmiaru obrazu wejściowego utrzymuje 50 FPS per kamera?**

---

## Czego potrzebujesz na komputerze testowym

**Wymagane (bez tego nie ruszy):**
- Windows 10 lub 11
- Karta NVIDIA z CUDA (np. RTX 2070 / 3060 / 4060 lub mocniejsza)
- Sterowniki NVIDIA aktualne (driver ≥ 535, ideally najnowszy)
- **CUDA Toolkit** zainstalowany (12.x preferowane, ale 11.8 też zadziała)
- **Python 3.10, 3.11 lub 3.12** w PATH (sprawdź: `python --version`)
- **Visual Studio Build Tools 2019 lub 2022** z workloadem **"Desktop development with C++"**
  (potrzebne do JIT-build natywnego CUDA debayera)
- ~5 GB wolnego miejsca na dysku (na venv + biblioteki)
- Połączenie z internetem (do pobrania paczek `pip`)

**Opcjonalne (NIE potrzebne dla tego testu):**
- pylon SDK + 4 kamery Basler — tylko jeśli chcemy test z prawdziwymi kamerami.
  Dla tej iteracji robimy tylko **synthetic test**, więc kamery NIE są wymagane.

---

## Co trzeba zrobić — krok po kroku

### Krok 1. Rozpakuj paczkę

Rozpakuj ZIP na dysk, np. do `D:\benchmark\`. Powinieneś dostać folder
`volleyhub_fastpath_benchmark` ze strukturą:

```
volleyhub_fastpath_benchmark\
  README.md                         <-- techniczna dokumentacja
  INSTRUKCJA_TESTERA.md             <-- ten plik
  RUN_ME_FIRST.ps1                  <-- bootstrap
  run_synthetic_sweep.ps1           <-- DOMYŚLNY test
  run_real_basler_sweep.ps1         <-- (pomijamy, wymaga kamer)
  requirements.txt
  configs\
  scripts\
  src\
  artifacts\tensorrt_exports\       <-- engines TensorRT (4 sztuki)
  reports\                          <-- tu wylądują wyniki
  logs\
```

### Krok 2. Otwórz PowerShell jako Administrator

> **WAŻNE**: Otwórz PowerShell **z prawami administratora** (prawym klikiem → "Uruchom jako administrator").
> Pierwszy uruchomienie może wymagać zezwolenia na uruchamianie skryptów.

Jeśli przy uruchomieniu `.ps1` zobaczysz komunikat o policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Zatwierdź "T" (Tak). To ustawia bypass tylko dla tego okna PowerShell.

### Krok 3. Wejdź do folderu paczki

```powershell
cd D:\benchmark\volleyhub_fastpath_benchmark
```

(Jeśli rozpakowałaś gdzie indziej, podaj swoją ścieżkę.)

### Krok 4. Uruchom bootstrap

```powershell
.\RUN_ME_FIRST.ps1
```

Co się stanie:
- skrypt utworzy `.venv` (lokalne wirtualne środowisko Pythona)
- pobierze i zainstaluje wszystkie wymagane biblioteki (~2–5 minut, zależy od łącza)
- uruchomi sprawdzenie środowiska — zobaczysz tabelkę:

```
============================================================
 Environment check — overall: OK / WARN / FAIL
============================================================
  [OK  ] python                 3.12.x on Windows-11-...
  [OK  ] torch                  torch 2.5.x, CUDA 12.x, GPU NVIDIA ...
  [OK  ] tensorrt               tensorrt 10.x
  [WARN] pypylon                PYPYLON_MISSING — synthetic OK, basler nie
  [OK  ] opencv                 opencv 4.x
  [OK  ] native_cuda_debayer    available, detail=custom_cuda_kernel
  [OK  ] engines                all 4 engines present
  [OK  ] directories            all required directories exist
```

Akceptowalny stan to **`overall: OK` lub `WARN`**. Zielone `[OK]` przy:
- python
- torch (musi być cuda_available=True!)
- tensorrt
- opencv
- native_cuda_debayer
- engines

`pypylon WARN` jest OK — nie używamy kamer w tym teście.

> **Co jeśli `[FAIL]`?** Patrz sekcja "Problemy" niżej.

### Krok 5. Uruchom benchmark

```powershell
.\run_synthetic_sweep.ps1
```

Co się stanie:
- skrypt zmierzy 4 warianty (640×640, 960×960, 1280×1280, 1088×1920) po 60 sekund każdy
- razem ~4 minuty + ładowanie engines
- pod koniec automatycznie zbuduje raport i otworzy folder `reports\`

W trakcie zobaczysz logi typu:

```
=== variant 640x640 | imgsz=640 | engine=best__fp16_640_b4_static...
=== variant 640x640 done. status=PASS packet_p95=15.73 ms
```

### Krok 6. Zwróć 3 pliki

Po zakończeniu prześlij **3 pliki** z folderu `reports\`:

1. **`reports\fastpath_comparison_report.md`** — czytelny raport z werdyktem
2. **`reports\results.json`** — surowe dane (potrzebne do dalszej analizy)
3. **`reports\environment_check.json`** — info o sprzęcie i wersjach

Spakuj te 3 pliki do ZIP i odeślij. To wszystko czego potrzebujemy.

(Opcjonalnie możesz przejrzeć sam `fastpath_comparison_report.md` — sekcja
**A. Executive summary** ma odpowiedź wprost.)

---

## Co znajdziesz w raporcie

Otwórz `reports\fastpath_comparison_report.md` w dowolnym edytorze tekstowym
(albo `fastpath_comparison_report.html` w przeglądarce — ładniej wygląda).

Najważniejsze sekcje:

- **A. Executive summary** — odpowiedź wprost: ile FPS przy Full HD i jaki maks rozmiar dla 50 FPS
- **B. Hardware comparison vs baseline** — porównanie z RTX 2080 Super (`speedup_vs_baseline`)
- **C. Resolution sweep** — pełna tabela 4 wariantów
- **D. Stage breakdown** — który etap (color, inference, postprocess...) zjada najwięcej czasu
- **E. Decision answer** — krótkie werdykty + presety LIVE/QUALITY/FULLHD
- **F. Environment** — info o Twoim sprzęcie
- **G. Errors / warnings** — czy coś nie zadziałało

Wykresy są w `reports\final_charts\` — 6 PNG-ów.

---

## Problemy — co robić

### `[FAIL] torch` lub `cuda_available=False`

Twój torch nie widzi karty graficznej. Najczęstsze przyczyny:
1. Sterownik NVIDIA za stary → zaktualizuj z https://www.nvidia.com/Download/index.aspx
2. Zainstalował się torch CPU-only (ten z PyPI jest CPU). Trzeba reinstalować z indeksu CUDA:

```powershell
.\.venv\Scripts\Activate.ps1
pip uninstall torch -y
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python scripts\check_environment.py
```

Dla CUDA 11.8: użyj `--index-url https://download.pytorch.org/whl/cu118`.
Sprawdź wersję swojej CUDA: `nvidia-smi` (kolumna "CUDA Version").

### `[FAIL] tensorrt`

```powershell
.\.venv\Scripts\Activate.ps1
pip install tensorrt
```

### `[WARN] native_cuda_debayer` lub `NATIVE_BACKEND_MISSING`

Ten backend buduje się JIT przy pierwszym imporcie. Wymaga dwóch rzeczy w PATH:
- `nvcc` (z CUDA Toolkit)
- `cl.exe` (z Visual Studio Build Tools)

Najprostsze rozwiązanie:
1. Otwórz "x64 Native Tools Command Prompt for VS 2019/2022" (z menu Start)
2. W tym oknie:
   ```
   cd D:\benchmark\volleyhub_fastpath_benchmark
   .\.venv\Scripts\activate.bat
   python scripts\check_environment.py
   ```

Jeśli to pierwszy raz — kompilacja zajmie ~30 s, zobaczysz pasek `[1/3]`, `[2/3]`, `[3/3]`.

### `[WARN] engines` lub `ENGINE_MISSING` w wynikach

Engine TensorRT może nie zadziałać na Twoim sprzęcie (różne GPU / wersje TRT).
Wtedy dany wariant w raporcie ma `status=ENGINE_MISSING` i jest pomijany — nie jest błędem,
po prostu odpowiedni engine trzeba przebudować lokalnie. Nie blokuje testu.

Jeśli chcesz przebudować lokalnie (wymaga `best.pt` w folderze paczki — pobierz od nas osobno):

```powershell
python scripts\export_or_check_engines.py --check          # diagnostyka
python scripts\export_or_check_engines.py --build-missing  # rebuild brakujących
```

Każdy engine ~3 minuty buildu.

### Inne błędy w trakcie sweepa

Skrypty są napisane defensywnie — wyłapują wyjątki i wpisują je do `results.json` jako pole
`errors` per wariant. Nawet jeśli któryś wariant umrze, pozostałe się dokończą i raport powstanie.

Jeśli zobaczysz traceback po którym test się **zatrzymuje** całkowicie — prześlij output konsoli
tak jak jest, łącznie z tracebackiem.

### Jak zacząć od nowa

```powershell
# usuń venv i wyniki
Remove-Item -Recurse -Force .venv, reports
.\RUN_ME_FIRST.ps1
.\run_synthetic_sweep.ps1
```

---

## FAQ

**Czy potrzebuję modelu YOLO (`best.pt`)?**
Nie. Engines TensorRT są w paczce. `best.pt` jest potrzebny TYLKO jeśli engines nie ładują się
i trzeba je przebudować.

**Czy test obciąża dysk / sieć?**
Nie. Wszystko liczy się na GPU + RAM. Sieć potrzebna tylko raz, do `pip install`.

**Czy mogę używać komputera w trakcie testu?**
Lepiej nie. Test mierzy FPS, więc inne procesy obciążające GPU/CPU mogą zaniżyć wynik.
Zamknij gry, OBS, edytory wideo, przeglądarki z YouTube/Netflix.

**Jak długo to trwa?**
- Pobranie i instalacja: 2–5 minut (zależy od łącza)
- Sprawdzenie środowiska: ~10 sekund
- Sweep 4 wariantów × 60 s = ~4–5 minut + ~30 s ładowania engines
- **Łącznie ~10 minut** od rozpakowania ZIP-a do wyniku.

**Jak duży jest raport?**
- `fastpath_comparison_report.md` — 5–10 KB, czysty tekst
- `results.json` — 5–20 KB, JSON
- `environment_check.json` — 1–2 KB
- (`final_charts/*.png` — 6 plików × ~75 KB, opcjonalnie)

ZIP z odpowiedzią ~50 KB. Spokojnie wszędzie się zmieści.

**Mogę uruchomić to na laptopie / na komputerze bez NVIDII?**
Bez karty NVIDII: nie. Test używa CUDA.
Na laptopie z NVIDIĄ: tak, byle laptop był wpięty do zasilania (battery saver tnie GPU).

---

## Co dalej

Po otrzymaniu Twoich 3 plików (`fastpath_comparison_report.md`, `results.json`,
`environment_check.json`) wiemy:

1. Ile FPS Twój komputer wyciąga przy Full HD inference dla 4 kamer
2. Jaki największy rozmiar inferencji utrzymuje stabilne 50 FPS
3. Gdzie jest bottleneck (color / inference / postprocess)
4. Czy Twój komputer jest szybszy / wolniejszy od baseline (RTX 2080 Super)

Na podstawie tego decydujemy konfigurację produkcyjną dla 4-kamerowego systemu.

Dzięki za pomoc! 🙏

---

## Krótka ściąga

```powershell
cd <folder z paczki>
.\RUN_ME_FIRST.ps1
.\run_synthetic_sweep.ps1
# odeślij: reports\fastpath_comparison_report.md
#         reports\results.json
#         reports\environment_check.json
```
