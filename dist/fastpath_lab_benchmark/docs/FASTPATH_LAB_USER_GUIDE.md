# FASTPATH LAB BENCHMARK â€” instrukcja obsĹ‚ugi

> Wersja dla testera zewnÄ™trznego. Nie zakĹ‚ada znajomoĹ›ci architektury
> systemu VolleyHub_K. Po przeczytaniu tej instrukcji potrafisz
> uruchomiÄ‡ paczkÄ™ krok po kroku i odczytaÄ‡ z raportu odpowiedzi na
> dwa kluczowe pytania:
>
> 1. *Ile FPS moĹĽna uzyskaÄ‡ przy inferencji Full HD-like (1088Ă—1920)?*
> 2. *Jaki najwiÄ™kszy rozmiar inferencji speĹ‚nia stabilne 50 FPS?*

---

## 1. Cel paczki

`fastpath_lab_benchmark.zip` to **proste, jednoznaczne narzÄ™dzie
eksperymentalne** do pomiaru wydajnoĹ›ci GPU. **Nie jest to** miniaturowa
aplikacja produkcyjna â€” jest to celowo zminimalizowany zestaw
skryptĂłw, w ktĂłrym kaĹĽdy etap moĹĽna uruchomiÄ‡ osobno i zobaczyÄ‡ jego
koszt.

Paczka umoĹĽliwia pomiar:

- samej inferencji TensorRT,
- samego toru color (debayer + resize + normalize) na GPU,
- peĹ‚nego syntetycznego toru RAW â†’ GPU color â†’ TensorRT,
- sweepu rozdzielczoĹ›ci inferencji (640, 960, 1280, 1088Ă—1920),
- pojedynczego, wĹ‚asnego eksperymentu sterowanego plikiem JSON.

**Ten benchmark NIE mierzy:**

- realnego grabowania z kamer Basler / pypylon,
- GUI (preview, podglÄ…d, panele sterowania),
- ring buffera / shared memory / kolejek produkcyjnych,
- synchronizacji sprzÄ™towej miÄ™dzy kamerami,
- zapisu na dysk, kompresji wideo,
- NMS ani postprocessu detekcji,
- peĹ‚nego dziaĹ‚ania aplikacji VolleyHub.

**Co wiÄ™c wĹ‚aĹ›ciwie mierzy?** **GĂłrnÄ… granicÄ™ wydajnoĹ›ci toru GPU**
oraz **koszt poszczegĂłlnych etapĂłw** (debayer, inferencja). JeĹ›li ten
benchmark daje PASS dla 50 FPS, to znaczy ĹĽe **GPU ma potencjaĹ‚** â€”
ale to nadal nie dowĂłd, ĹĽe caĹ‚a aplikacja produkcyjna osiÄ…gnie
50 FPS na koĹ„cu (kamery, IPC, GUI dokĹ‚adajÄ… swoje).

---

## 2. NajwaĹĽniejsze pojÄ™cia

### Packet

`1 packet` = jedna paczka 4 obrazĂłw, **po jednym z kaĹĽdej kamery**.
Engine TensorRT z batch=4 wykonuje inferencjÄ™ dla caĹ‚ego packetu
za jednym wywoĹ‚aniem.

### FPS per camera

`FPS per camera` = liczba packetĂłw na sekundÄ™.

> PrzykĹ‚ad: 50 packetĂłw / s = 50 FPS na kamerÄ™ = 200 obrazĂłw / s
> Ĺ‚Ä…cznie dla 4 kamer.

### Batch

`batch=4` oznacza, ĹĽe engine TensorRT przetwarza **4 obrazy naraz**
w jednym wywoĹ‚aniu. To jest wĹ‚aĹ›ciwoĹ›Ä‡ pliku `.engine` (statyczna),
nie da siÄ™ jej zmieniÄ‡ bez przebudowy.

### active_cameras

Ile slotĂłw w batchu jest **wypeĹ‚nionych realnÄ… treĹ›ciÄ…** (piĹ‚ka,
tĹ‚o). PozostaĹ‚e sloty zawierajÄ… ciemne, puste ramki. Ten parametr
sĹ‚uĹĽy **do interpretacji scenariusza** ("a co jeĹ›li mam tylko
3 kamery podĹ‚Ä…czone?"), **a nie do skrĂłcenia GPU**. Engine static
batch=4 nadal liczy 4 obrazy.

### target_fps

- `target_fps > 0`: skrypt 06 **pacingowo** prĂłbuje utrzymaÄ‡ taki rytm (np. 50 packetĂłw / s â†’ jeden packet co 20 ms).
- `target_fps = 0`: skrypt dziaĹ‚a w **free-run** â€” mierzy maksymalnÄ… przepustowoĹ›Ä‡ bez sztucznego ograniczenia.

### p95

`p95` to wartoĹ›Ä‡ bezpieczna (pesymistyczna). MĂłwi: w 95% przypadkĂłw
czas byĹ‚ **lepszy lub rĂłwny** tej liczbie. Zwykle `p95 > median`.
JeĹ›li `packet_ms_p95 <= 20 ms`, moĹĽna mĂłwiÄ‡ o **stabilnych 50 FPS**.

### BudĹĽet 50 FPS

```
packet_budget_ms = 1000 ms / 50 FPS = 20 ms
```

Stabilne 50 FPS = `packet_ms_p95 <= 20 ms`. Margines stabilnoĹ›ci to
rĂłĹĽnica `20 - packet_ms_p95` (im wiÄ™cej tym lepiej).

---

## 3. Struktura paczki

```
fastpath_lab_benchmark/
â”śâ”€â”€ README.md                       (krĂłtkie wprowadzenie)
â”śâ”€â”€ requirements.txt
â”śâ”€â”€ RUN_ALL.ps1                     (peĹ‚ny sweep + raport)
â”śâ”€â”€ CHECK_ENV.ps1                   (sprawdzenie Ĺ›rodowiska)
â”‚
â”śâ”€â”€ configs/                        (pliki konfiguracyjne)
â”‚   â”śâ”€â”€ benchmark_config.yaml       (domyĹ›lne wartoĹ›ci dla RUN_ALL)
â”‚   â”śâ”€â”€ baseline_rtx2080super.json  (liczby referencyjne)
â”‚   â””â”€â”€ single_test.json            (config 1 eksperymentu)
â”‚
â”śâ”€â”€ engines/                        (pliki TensorRT .engine)
â”‚   â”śâ”€â”€ README_ENGINES.md
â”‚   â””â”€â”€ best__fp16_*.engine
â”‚
â”śâ”€â”€ scripts/                        (skrypty uruchamiane przez testera)
â”‚   â”śâ”€â”€ 00_check_env.py
â”‚   â”śâ”€â”€ 01_benchmark_inference_only.py
â”‚   â”śâ”€â”€ 02_benchmark_color_only.py
â”‚   â”śâ”€â”€ 03_benchmark_full_synthetic_path.py
â”‚   â”śâ”€â”€ 04_resolution_sweep.py
â”‚   â”śâ”€â”€ 05_make_report.py
â”‚   â””â”€â”€ 06_run_single_test.py
â”‚
â”śâ”€â”€ src/                            (maĹ‚e moduĹ‚y pomocnicze, czytelne)
â”‚   â”śâ”€â”€ trt_runner.py
â”‚   â”śâ”€â”€ cuda_debayer.py
â”‚   â”śâ”€â”€ synthetic_raw.py
â”‚   â”śâ”€â”€ metrics.py
â”‚   â”śâ”€â”€ report_builder.py
â”‚   â””â”€â”€ native_debayer/             (custom CUDA kernel + adapter)
â”‚
â”śâ”€â”€ results/                        (wyniki JSON + CSV â€” wypeĹ‚ni siÄ™ po RUN)
â”śâ”€â”€ reports/                        (raport HTML/MD + wykresy)
â””â”€â”€ docs/                           (TA instrukcja)
```

| folder | rola |
|---|---|
| `configs/` | pliki konfiguracyjne eksperymentĂłw (YAML/JSON) |
| `engines/` | pliki TensorRT `.engine` |
| `scripts/` | skrypty CLI uruchamiane przez testera |
| `src/` | maĹ‚e moduĹ‚y (~150â€“250 linii kaĹĽdy) uĹĽywane przez skrypty |
| `results/` | surowe wyniki JSON / CSV z kaĹĽdego uruchomienia |
| `reports/` | raporty HTML / Markdown + wykresy PNG |
| `docs/` | dokumentacja uĹĽytkownika (ten plik) |

---

## 4. Szybki start

### Krok 1 â€” rozpakuj ZIP

```
C:\benchmark\fastpath_lab_benchmark\
```

### Krok 2 â€” zainstaluj zaleĹĽnoĹ›ci

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install tensorrt==10.16.*
pip install -r requirements.txt
```

### Krok 3 â€” uruchom sprawdzenie Ĺ›rodowiska

```powershell
.\CHECK_ENV.ps1
```

(rĂłwnowaĹĽne: `python scripts\00_check_env.py`)

### Krok 4 â€” uruchom peĹ‚ny zestaw testĂłw

```powershell
.\RUN_ALL.ps1
```

(`-DurationS 15 -Warmup 5` dla szybszego przebiegu)

### Krok 5 â€” otwĂłrz raport

```
reports\fastpath_lab_report.html
```

---

## 5. KolejnoĹ›Ä‡ analizy â€” ogĂłlny przebieg

Skrypty sÄ… ponumerowane celowo. Logiczny przepĹ‚yw:

| # | skrypt | rola |
|---|---|---|
| 1 | `00_check_env.py` | sprawdza, czy maszyna ma wszystko czego trzeba |
| 2 | `01_benchmark_inference_only.py` | mierzy sam TensorRT, bez color, bez RAW |
| 3 | `02_benchmark_color_only.py` | mierzy sam GPU debayer + resize + normalize |
| 4 | `03_benchmark_full_synthetic_path.py` | mierzy peĹ‚ny tor: RAW â†’ color â†’ TRT |
| 5 | `04_resolution_sweep.py` | powtarza (4) dla 640, 960, 1280, 1088Ă—1920 |
| 6 | `05_make_report.py` | skĹ‚ada wszystko w raport HTML + Markdown + wykresy |
| 7 | `06_run_single_test.py` | jeden wĹ‚asny eksperyment z `single_test.json` |

Najszybsza Ĺ›cieĹĽka do odpowiedzi: **(1) â†’ (5) â†’ (6) raportu**.
Skrypty 1â€“4 sÄ… dla gĹ‚Ä™bszej analizy (gdzie konkretnie jest
bottleneck).

---

## 6. Skrypt `00_check_env.py`

### Do czego sĹ‚uĹĽy

Sprawdza Ĺ›rodowisko maszyny testera. Bez tego skryptu nie wiadomo,
czy w ogĂłle ma sens uruchamiaÄ‡ dalsze benchmarki.

### Co sprawdza

- wersjÄ™ Pythona,
- OS,
- `torch` + dostÄ™pnoĹ›Ä‡ CUDA + nazwÄ™ GPU + VRAM,
- `tensorrt`,
- `matplotlib` (opcjonalnie â€” do wykresĂłw),
- native CUDA debayer (custom CUDA kernel z `src/native_debayer`),
- obecnoĹ›Ä‡ plikĂłw `.engine` w `engines/`.

### Czego NIE mierzy

- Nie mierzy FPS.
- Nie uruchamia inferencji.
- Nie uruchamia GPU benchmarku.
- Nie kompiluje engine (jeĹ›li brakuje â€” tylko raportuje brak).

### Komenda

```powershell
python scripts\00_check_env.py
```

### Pliki wynikowe

```
results\environment.json
results\environment.md
```

### Jak interpretowaÄ‡ wynik

| status | znaczenie |
|---|---|
| **OK** | komponent dostÄ™pny |
| **WARNING** | komponent opcjonalny (np. matplotlib â€” bez niego raport powstanie, ale bez wykresĂłw) |
| **FAIL** | komponent krytyczny â€” benchmark GPU moĹĽe nie dziaĹ‚aÄ‡ |

Typowe usterki:

- `CUDA_UNAVAILABLE` â†’ benchmark GPU nie ruszy. Sterownik NVIDIA / `torch` GPU build.
- `TENSORRT_MISSING` â†’ nie bÄ™dzie inferencji TensorRT.
- `ENGINE_MISSING` â†’ brakuje konkretnego pliku `.engine`.
- `NATIVE_BACKEND_MISSING` â†’ custom CUDA debayer nie buduje siÄ™ (brak Visual C++ Build Tools lub CUDA Toolkit). Skrypty kolorowe i peĹ‚ny tor przeĹ‚Ä…czÄ… siÄ™ na wolniejszy `torch_gpu_fallback`.

---

## 7. Skrypt `01_benchmark_inference_only.py`

### Do czego sĹ‚uĹĽy

Mierzy **samÄ… inferencjÄ™ TensorRT**. Daje odpowiedĹş na pytanie:
"czy sam engine jest wystarczajÄ…co szybki dla danego input size?".

### Co dokĹ‚adnie mierzy

```
CUDA tensor [batch, 3, H, W]   (losowy, alokowany raz)
   â†’ TensorRT forward (execute_async_v3)
   â†’ CUDA synchronize
   â†’ pomiar czasu (CUDA event)
```

### Czego NIE mierzy

- nie mierzy kamer,
- nie mierzy RAW / Bayer,
- nie mierzy debayera,
- nie mierzy resize / normalize,
- nie mierzy GUI / ring buffera / kolejek,
- nie mierzy generacji syntetycznego obrazu (tensor jest losowy, zaalokowany raz).

### Kiedy go uĹĽywaÄ‡

Gdy chcesz wiedzieÄ‡: "czy *sam* TensorRT engine wyrabia siÄ™ dla
danego input size na tym GPU?" JeĹ›li ten test daje FAIL dla
1088Ă—1920, to nie warto dalej walczyÄ‡ â€” caĹ‚oĹ›Ä‡ bÄ™dzie wolniejsza.

### PrzykĹ‚ad komendy

```powershell
python scripts\01_benchmark_inference_only.py `
    --engine engines\best__fp16_640_b4_static.engine `
    --input-shape 640 640 --batch 4 --duration-s 30
```

### NajwaĹĽniejsze parametry

| flaga | znaczenie |
|---|---|
| `--engine` | Ĺ›cieĹĽka do pliku TensorRT `.engine` |
| `--input-shape H W` | rozmiar wejĹ›cia (musi pasowaÄ‡ do engine) |
| `--batch` | batch (musi pasowaÄ‡ do engine) |
| `--duration-s` | czas pomiaru po warmupie |
| `--warmup` | iteracje rozgrzewkowe (odrzucane ze statystyk) |

### Pliki wynikowe

```
results\inference_only_<HxW>.json
results\inference_only_<HxW>.csv
```

### NajwaĹĽniejsze metryki

- `inference_ms_median`, `inference_ms_p95`
- `packet_ms_median`, `packet_ms_p95`
- `fps_per_camera_median`, `fps_per_camera_safe_p95`
- `pass_50fps_safe_p95`

### Interpretacja

| warunek | wniosek |
|---|---|
| `inference_ms_p95 > 20 ms` | sama inferencja jest zbyt wolna dla 50 FPS, niezaleĹĽnie od reszty |
| `inference_ms_p95 < 20 ms` | sama inferencja siÄ™ mieĹ›ci, ale trzeba doliczyÄ‡ koszt color/resize i overhead |

---

## 8. Skrypt `02_benchmark_color_only.py`

### Do czego sĹ‚uĹĽy

Mierzy **sam tor GPU color**: debayer + resize + normalize.

### Co dokĹ‚adnie mierzy

```
synthetic RAW/Bayer batch [4, capH, capW]
   â†’ upload na CUDA
   â†’ native CUDA debayer (custom_cuda_kernel)
   â†’ bilinear resize do (outH, outW)
   â†’ normalize do float [0,1]
   â†’ float16
   â†’ output: CUDA tensor [4, 3, outH, outW]
```

### Czego NIE mierzy

- nie mierzy TensorRT,
- nie mierzy detekcji,
- nie mierzy kamer Basler,
- nie mierzy GUI,
- nie mierzy kolejek produkcyjnych.

### PrzykĹ‚ad komendy

```powershell
python scripts\02_benchmark_color_only.py `
    --capture-shape 1080 1920 --output-shape 640 640 `
    --batch 4 --duration-s 30
```

### Parametry

| flaga | znaczenie |
|---|---|
| `--capture-shape H W` | rozdzielczoĹ›Ä‡ syntetycznego RAW Bayer |
| `--output-shape H W` | docelowy rozmiar po resize |
| `--batch` | batch |
| `--bayer-pattern` | RG / BG / GR / GB |
| `--backend` | `auto` / `native` / `torch` |

### Pliki wynikowe

```
results\color_only_<HxW>.json
results\color_only_<HxW>.csv
```

### NajwaĹĽniejsze metryki

- `color_ms_median`, `color_ms_p95`
- `output_location` (powinno byÄ‡ `"cuda"`)
- `gpu_roundtrip` (powinno byÄ‡ `false`)
- `backend_used` (`native_cuda_kernel` lub `torch_gpu_fallback`)

### Interpretacja

| warunek | wniosek |
|---|---|
| `color_ms_p95` maĹ‚e (np. < 3 ms) | color nie jest bottleneckiem |
| `color_ms_p95` duĹĽe | bottleneck w debayer / resize |
| `output_location = "cuda"` i `gpu_roundtrip = false` | tor jest poprawny GPU-only, bez CPU bounce |
| `backend_used = "torch_gpu_fallback"` | native kernel nie zbudowaĹ‚ siÄ™ â€” wynik jest 2Ă— wolniejszy niĹĽ mĂłgĹ‚by byÄ‡ |

---

## 9. Skrypt `03_benchmark_full_synthetic_path.py`

### Do czego sĹ‚uĹĽy

Mierzy **peĹ‚ny syntetyczny tor GPU** â€” bez produkcyjnej aplikacji,
ale z wszystkimi etapami GPU jednoczeĹ›nie.

### Co dokĹ‚adnie mierzy

```
synthetic Bayer batch [4, capH, capW] (numpy)
   â†’ CUDA upload
   â†’ native CUDA debayer
   â†’ bilinear resize + normalize
   â†’ TensorRT inference (zero-copy z color output)
   â†’ minimalny postprocess (tylko shape check, brak NMS)
```

### Czego NIE mierzy

- nie mierzy Basler RetrieveResult,
- nie mierzy realnych timestampĂłw,
- nie mierzy GUI,
- nie mierzy ring buffera,
- nie mierzy kolejek produkcyjnych,
- nie mierzy zapisu na dysk,
- nie mierzy realnej synchronizacji sprzÄ™towej.

### PrzykĹ‚ad komendy

```powershell
python scripts\03_benchmark_full_synthetic_path.py `
    --engine engines\best__fp16_640_b4_static.engine `
    --capture-shape 1080 1920 --input-shape 640 640 `
    --batch 4 --duration-s 30
```

### Pliki wynikowe

```
results\full_synthetic_<HxW>.json
results\full_synthetic_<HxW>.csv
```

### NajwaĹĽniejsze metryki

- `color_ms_median` / `_p95`
- `inference_ms_median` / `_p95`
- `postprocess_ms_median` / `_p95`
- `packet_ms_median` / `_p95` (to jest "caĹ‚oĹ›Ä‡" pakietu na GPU)
- `fps_per_camera_median`, `fps_per_camera_safe_p95`
- `pass_50fps_safe_p95`
- `zero_copy_to_inference` (powinno byÄ‡ `true`)
- `fallback_used` (powinno byÄ‡ `false` w zdrowej konfiguracji)

### Interpretacja

**To jest najwaĹĽniejszy test dla pytania**:
> Czy tor GPU ma potencjaĹ‚ osiÄ…gnÄ…Ä‡ 50 FPS?

| warunek | wniosek |
|---|---|
| `packet_ms_p95 <= 20 ms` | tor GPU speĹ‚nia stabilne 50 FPS |
| `packet_ms_p95 > 20 ms` | dany rozmiar inferencji jest za ciÄ™ĹĽki dla stabilnych 50 FPS |
| `fallback_used = true` | wynik **NIE** odpowiada prawdziwemu fast path â€” coĹ› jest nie tak ze Ĺ›rodowiskiem |
| `zero_copy_to_inference = false` | miÄ™dzy color a inferencjÄ… jest CPU bounce â€” wynik gorszy niĹĽ mĂłgĹ‚by byÄ‡ |

---

## 10. Skrypt `04_resolution_sweep.py`

### Do czego sĹ‚uĹĽy

Automatycznie uruchamia peĹ‚ny tor (`03_*`) dla zestawu rozmiarĂłw
inferencji. To jest **gĹ‚Ăłwny skrypt** dla odpowiedzi na dwa
kluczowe pytania uĹĽytkownika.

DomyĹ›lnie testuje:

- 640 Ă— 640
- 960 Ă— 960
- 1280 Ă— 1280
- 1088 Ă— 1920 (Full HD-like)

### Co mierzy

Dla kaĹĽdego rozmiaru: dokĹ‚adnie to co `03_benchmark_full_synthetic_path.py`.

### PrzykĹ‚ad komendy

```powershell
python scripts\04_resolution_sweep.py --duration-s 30
```

### Pliki wynikowe

```
results\resolution_sweep.json
results\resolution_sweep.csv
results\full_synthetic_<HxW>.{json,csv}   (po jednym pliku na rozmiar)
```

### NajwaĹĽniejsze metryki (per wiersz)

- `inference_input_shape`
- `pixels_mpx`
- `color_ms_p95`
- `inference_ms_p95`
- `packet_ms_p95`
- `fps_per_camera_median`, `fps_per_camera_safe_p95`
- `pass_50fps_safe_p95`
- `stability_margin_ms`

### Interpretacja

Ten skrypt **bezpoĹ›rednio odpowiada** na:

1. **Ile FPS daje Full HD-like inference?** â†’ wiersz `1088x1920`, pole `fps_per_camera_safe_p95`.
2. **Jaki najwiÄ™kszy input speĹ‚nia 50 FPS?** â†’ najwiÄ™kszy wiersz (po `pixels_mpx`) z `pass_50fps_safe_p95 = true`.

PrzykĹ‚ad:

> JeĹĽeli 1088Ă—1920 ma `fps_per_camera_safe_p95 = 32` â†’ Full HD inference
> to tryb ~30 FPS, nie 50.
>
> JeĹĽeli 640Ă—640 ma `pass_50fps_safe_p95 = true` â†’ 640Ă—640 speĹ‚nia
> stabilne 50 FPS.

---

## 11. Skrypt `05_make_report.py`

### Do czego sĹ‚uĹĽy

SkĹ‚ada wyniki z `results/` w **raport HTML + Markdown + wykresy**.
Nie uruchamia ĹĽadnych nowych benchmarkĂłw â€” tylko agreguje to co juĹĽ
jest na dysku.

### Co robi

Czyta:

- `results/full_synthetic_*.json` (wyniki sweepu),
- `results/environment.json` (info o maszynie),
- `configs/baseline_rtx2080super.json` (liczby referencyjne).

Generuje:

- raport HTML,
- raport Markdown,
- 4 wykresy PNG (FPS vs rozmiar, packet_ms vs 20 ms budget, inference vs Mpx, stage breakdown).

### Komenda

```powershell
python scripts\05_make_report.py
```

### Pliki wynikowe

```
reports\fastpath_lab_report.html
reports\fastpath_lab_report.md
reports\charts\fps_per_camera.png
reports\charts\packet_ms_vs_budget.png
reports\charts\inference_vs_mpx.png
reports\charts\stage_breakdown.png
```

### Sekcje raportu

- **Machine** â€” co za maszyna (GPU, CUDA, TRT).
- **Direct answers** â€” Q1 (Full HD-like FPS) i Q2 (najwiÄ™kszy 50 FPS).
- **Suggested presets** â€” LIVE_50FPS / QUALITY_40FPS / FULLHD_30FPS.
- **Resolution sweep table** â€” peĹ‚na tabela wynikĂłw.
- **Baseline comparison** â€” porĂłwnanie z referencyjnym RTX 2080 Super.
- **Charts** â€” 4 wykresy.
- **What this benchmark does NOT measure** â€” disclaimer.

---

## 12. Skrypt `06_run_single_test.py`

### Do czego sĹ‚uĹĽy

Pozwala uruchomiÄ‡ **jeden, wĹ‚asny eksperyment** â€” bez sweepu, z
parametrami pobranymi z jednego pliku JSON. Najprostszy tryb dla
testera, ktĂłry chce powiedzieÄ‡: "uruchom dokĹ‚adnie ten scenariusz".

### Plik konfiguracyjny

```
configs\single_test.json
```

### Komenda

```powershell
python scripts\06_run_single_test.py --config configs\single_test.json
```

### Co moĹĽna zmieniaÄ‡ w `single_test.json`

| pole | znaczenie |
|---|---|
| `engine` | Ĺ›cieĹĽka do `.engine` |
| `capture_shape_hw` | rozmiar syntetycznego RAW Bayer `[H, W]` |
| `inference_shape_hw` | rozmiar wejĹ›cia do inferencji `[H, W]` â€” musi pasowaÄ‡ do engine |
| `batch` | batch (musi pasowaÄ‡ do engine) |
| `active_cameras` | 1..batch â€” ile slotĂłw ma realnÄ… treĹ›Ä‡; reszta to ciemne ramki |
| `target_fps` | docelowy FPS; 0 = free-run |
| `duration_s` | czas pomiaru po warmupie |
| `warmup` | iteracje rozgrzewkowe |
| `bayer_pattern` | RG / BG / GR / GB |
| `synthetic_source.dtype` | uint8 / uint16 |
| `synthetic_source.noise_sigma` | szum Gaussa (sigma w 0..255) |
| `synthetic_source.ball_enabled` | piĹ‚ka tak/nie |
| `synthetic_source.ball_radius_px` | promieĹ„ piĹ‚ki |
| `synthetic_source.ball_speed_px_per_frame` | prÄ™dkoĹ›Ä‡ piĹ‚ki |
| `synthetic_source.base_intensity` | jasnoĹ›Ä‡ tĹ‚a |
| `synthetic_source.background_amplitude` | amplituda gradientu |
| `synthetic_source.ball_intensity` | jasnoĹ›Ä‡ piĹ‚ki |
| `synthetic_source.seed` | seed RNG |

### Co mierzy

GPU fast path dla jednego scenariusza:
synthetic RAW batch â†’ GPU color â†’ TensorRT.

### Czego NIE mierzy

- nie mierzy kosztu generowania syntetycznej ramki w pÄ™tli,
- nie mierzy realnej kamery,
- nie mierzy GUI,
- nie mierzy kolejek,
- nie mierzy ring buffera.

> **Uwaga waĹĽna:** Batch syntetyczny jest **generowany raz** i
> reusowany w pÄ™tli, **aby mierzyÄ‡ tor GPU**, a nie CPU-side
> rendering. Bez tego przy 1080Ă—1920Ă—4 generacja w numpy
> zdominowaĹ‚aby pÄ™tlÄ™ i ukryĹ‚a prawdziwÄ… wydajnoĹ›Ä‡ GPU.

### Jak interpretowaÄ‡ output

| pole | znaczenie |
|---|---|
| `actual sustained FPS` | rzeczywiĹ›cie utrzymany FPS (iterations / elapsed) â€” Ĺ‚Ä…cznie z pacing sleeps |
| `overruns` | liczba iteracji, w ktĂłrych GPU/tor nie zmieĹ›ciĹ‚ siÄ™ w okresie |
| `target_hit` | YES gdy actual FPS â‰Ą 98% target **i** overrun_ratio < 5% |
| `FPS per camera (uncapped median/p95)` | maksymalna przepustowoĹ›Ä‡ bez pacingu |

`target_hit = NO` â†’ konfiguracja **nie utrzymuje** zadanego FPS.

### PrzykĹ‚ad interpretacji

**Scenariusz**: `batch=4, active_cameras=4, target_fps=50`
- `sustained FPS = 49.99`, `overruns = 0/150`, `target_hit = YES`
- **Wniosek:** konfiguracja utrzymuje 50 FPS dla 4 kamer.

**Scenariusz**: `batch=4, active_cameras=1, target_fps=0` (free-run)
- pokazuje **maksymalnÄ… przepustowoĹ›Ä‡**,
- ale engine static batch=4 nadal wykonuje pracÄ™ dla 4 obrazĂłw â€”
  `active_cameras=1` w tym scenariuszu **nie zmniejsza** GPU.

---

## 13. ModuĹ‚y w `src/`

KaĹĽdy moduĹ‚ jest maĹ‚y (~150â€“250 linii) i moĹĽna przeczytaÄ‡ go w 5
minut. Nie ma ukrytej magii.

### `src/trt_runner.py`

**Rola:**
- Ĺ‚aduje TensorRT `.engine`,
- pomija nagĹ‚Ăłwek Ultralytics jeĹ›li jest,
- alokuje bufory I/O (torch tensors na CUDA),
- wykonuje `execute_async_v3`,
- mierzy inferencjÄ™ CUDA event'em.

**Input:** CUDA tensor `[B, 3, H, W]`.
**Output:** dict wyjĹ›ciowych tensorĂłw + `inference_ms`.

### `src/cuda_debayer.py`

**Rola:**
- owija native CUDA debayer (custom kernel),
- fallback na torch jeĹ›li native niedostÄ™pny,
- robi resize + normalize w jednym kroku,
- pilnuje, ĹĽeby output byĹ‚ na CUDA.

**Input:** RAW Bayer batch `[B, H, W]` uint8 (numpy lub torch).
**Output:** CUDA tensor `[B, 3, outH, outW]` float16/float32 + metadane backendu.

### `src/synthetic_raw.py`

**Rola:**
- generuje syntetyczny obraz RAW/Bayer,
- piĹ‚ka + gradientowe tĹ‚o + opcjonalny szum Gaussa,
- powtarzalne testy (`seed`).

**Pola konfiguracji** dokĹ‚adnie pasujÄ… do produkcyjnego
`SyntheticSourceConfig` (mapowanie 1:1).

> W `06_run_single_test.py` batch jest generowany raz i reusowany.

### `src/metrics.py`

**Rola:**
- liczy medianÄ™ i p95 dla prĂłbek czasu,
- liczy `fps_per_camera`,
- liczy verdict `pass_50fps_safe_p95`,
- liczy `stability_margin_ms`.

### `src/report_builder.py`

**Rola:**
- czyta `results/*.json`,
- porĂłwnuje z baseline RTX 2080 Super,
- buduje raport Markdown + HTML,
- generuje wykresy matplotlib.

---

## 14. Typowy przebieg analizy

### Scenariusz A â€” szybka odpowiedĹş na dwa pytania

```powershell
python scripts\00_check_env.py
python scripts\04_resolution_sweep.py --duration-s 30
python scripts\05_make_report.py
start reports\fastpath_lab_report.html
```

W raporcie sprawdĹş sekcjÄ™ **Direct answers**:

- Q1: Full HD-like inference FPS (median, safe p95, PASS/FAIL).
- Q2: Largest input at safe 50 FPS.

### Scenariusz B â€” sprawdzenie samej inferencji

```powershell
python scripts\01_benchmark_inference_only.py `
    --engine engines\best__fp16_1088x1920_b4_static.engine `
    --input-shape 1088 1920 --batch 4 --duration-s 30
```

JeĹ›li `inference_ms_p95 > 20 ms` â†’ Full HD inference **nie** speĹ‚ni
50 FPS niezaleĹĽnie od debayera. Dalsza walka nie ma sensu.

### Scenariusz C â€” sprawdzenie, czy bottleneckiem jest color

```powershell
python scripts\02_benchmark_color_only.py `
    --capture-shape 1080 1920 --output-shape 640 640 `
    --batch 4 --duration-s 30
```

JeĹ›li `color_ms_p95` jest **maĹ‚e** (np. < 3 ms), bottleneck jest
raczej w TensorRT. JeĹ›li **duĹĽe** â€” debayer/resize jest sprawcÄ….

### Scenariusz D â€” pojedynczy eksperyment z JSON

```powershell
# 1. Edytuj configs\single_test.json (engine, capture, inference, batch, target_fps).
# 2. Uruchom:
python scripts\06_run_single_test.py --config configs\single_test.json
```

SprawdĹş w outputie: `actual sustained FPS`, `overruns`, `target_hit`.

---

## 15. Interpretacja koĹ„cowych wynikĂłw

| Wynik | Znaczenie |
|---|---|
| `packet_ms_p95 <= 20` | stabilne 50 FPS |
| `packet_ms_median <= 20`, ale `p95 > 20` | Ĺ›rednio dziaĹ‚a, ale **niestabilnie** â€” drops |
| `inference_ms_p95 > 20` | **sama** inferencja za wolna |
| `color_ms_p95` duĹĽe | bottleneck w debayer / resize |
| `zero_copy_to_inference = false` | wynik **NIE** odpowiada fast path (jest CPU bounce) |
| `fallback_used = true` | test **NIE** dziaĹ‚a na wĹ‚aĹ›ciwym backendzie (native kernel padĹ‚) |
| `ENGINE_MISSING` | brak engine dla danego input shape |
| `NATIVE_BACKEND_MISSING` | brak custom CUDA debayera |

---

## 16. Jak odpowiedzieÄ‡ na pytania uĹĽytkownika na podstawie raportu

### Pytanie 1: Przy inferencji Full HD-like ile FPS moĹĽna uzyskaÄ‡?

1. OtwĂłrz `reports\fastpath_lab_report.html`.
2. W tabeli **Resolution sweep** znajdĹş wiersz `1088x1920`.
3. Odczytaj:
   - `fps_per_camera_median`,
   - `fps_per_camera_safe_p95`.

**Format odpowiedzi:**

> Przy inferencji Full HD-like 1088Ă—1920 ta maszyna osiÄ…ga okoĹ‚o
> **X FPS median** oraz **Y FPS safe p95** per camera.
> (Stabilne 50 FPS: **PASS/FAIL**.)

### Pytanie 2: Jaki najwiÄ™kszy rozmiar inferencji speĹ‚nia stabilne 50 FPS?

1. W tabeli **Resolution sweep** znajdĹş **najwiÄ™kszy wiersz** (po
   `pixels_mpx`), dla ktĂłrego `pass_50fps_safe_p95 = PASS`.
2. Odczytaj `label` tego wiersza.

**Format odpowiedzi:**

> NajwiÄ™kszy rozmiar inferencji speĹ‚niajÄ…cy stabilne 50 FPS
> (safe p95) na tej maszynie to **X Ă— Y**, z marginesem
> `stability_margin_ms = Z ms`.

---

## 17. Ograniczenia benchmarku

Ten benchmark **nie zastÄ™puje** testu produkcyjnego z kamerami.

JeĹ›li benchmark pokazuje PASS:

> "GPU fast path **ma potencjaĹ‚** osiÄ…gnÄ…Ä‡ 50 FPS dla danego input
> size na tej maszynie."

To NIE jest dowĂłd, ĹĽe caĹ‚a aplikacja produkcyjna osiÄ…gnie 50 FPS.
Aby to potwierdziÄ‡, trzeba pĂłĹşniej wykonaÄ‡ realny test z:

- kamerami Basler / pypylon (grabowanie),
- GUI,
- kolejkami / ring bufferem,
- synchronizacjÄ… sprzÄ™towÄ…,
- shared memory,
- zapisem.

KaĹĽdy z tych elementĂłw dokĹ‚ada swoje milisekundy.

---

## 18. Troubleshooting

### Brak TensorRT

**Objaw:** `TENSORRT_MISSING` lub `import tensorrt` rzuca bĹ‚Ä…d.
**RozwiÄ…zanie:** `pip install tensorrt==10.16.*` w wersji zgodnej z
CUDA Toolkit. MoĹĽe teĹĽ trzeba `tensorrt-libs`, `tensorrt-bindings`
na niektĂłrych dystrybucjach.

### Brak engine

**Objaw:** `ENGINE_MISSING` w `00_check_env.py`.
**RozwiÄ…zanie:** SkopiowaÄ‡ odpowiedni plik `.engine` do `engines/`
(zgodnie z `engines/README_ENGINES.md`). MoĹĽna teĹĽ dostaÄ‡ je z
maszyny, na ktĂłrej engine'y byĹ‚y eksportowane.

### Engine incompatible

**Objaw:** `TensorRT engine load failed` /
`magicTag != kEXPECTED_MAGIC_TAG` (TRT 10+).
**RozwiÄ…zanie:** Engine zostaĹ‚ zbudowany innÄ… wersjÄ… TRT lub na
innej architekturze GPU. **Przebuduj engine** na tej maszynie
(zgodnie z procedurÄ… eksportu w gĹ‚Ăłwnym repo, poza paczkÄ…).

### CUDA unavailable

**Objaw:** `torch.cuda.is_available() = False`.
**RozwiÄ…zanie:** SprawdziÄ‡:

1. czy zainstalowany sterownik NVIDIA (`nvidia-smi`),
2. czy `torch` jest w wersji **CUDA build** (nie CPU),
3. czy CUDA Toolkit pasuje do wersji `torch` (cu121, cu124, itp.).

### Native backend missing

**Objaw:** `NATIVE_BACKEND_MISSING`.
**RozwiÄ…zanie:** Custom CUDA debayer wymaga:

1. CUDA Toolkit (`nvcc`),
2. hosta C++ kompatybilnego (na Windows: Visual C++ Build Tools).

JeĹ›li to nie chodzi â€” paczka i tak dziaĹ‚a, tylko zamiast natywnego
kernela uĹĽywa wolniejszego `torch_gpu_fallback`. SprawdĹş w
`results/environment.json` pole `native_cuda_debayer.error`.

### Wynik FPS jest dziwnie niski

SprawdĹş kolejno:

- czy `fallback_used = false` (jeĹ›li `true` â€” naprawiamy native kernel),
- czy `zero_copy_to_inference = true` (jeĹ›li `false` â€” CPU bounce),
- czy engine ma wĹ‚aĹ›ciwy `batch` i `imgsz`,
- czy nie dziaĹ‚a inny proces na GPU (`nvidia-smi`),
- czy laptop nie jest w trybie oszczÄ™dzania energii (Battery / Quiet),
- czy temperatura GPU nie powoduje throttlingu,
- czy `--warmup` nie jest za maĹ‚y (dla 1088Ă—1920 warto 10+).

---

## 19. Minimalna checklista dla testera

```
[ ] RozpakowaĹ‚em ZIP do C:\benchmark\fastpath_lab_benchmark\.
[ ] ZainstalowaĹ‚em zaleĹĽnoĹ›ci (torch CUDA + tensorrt + requirements.txt).
[ ] UruchomiĹ‚em CHECK_ENV.ps1.
[ ] CUDA = OK.
[ ] TensorRT = OK.
[ ] Engine files = OK (cztery pliki .engine).
[ ] Native CUDA backend = OK (lub fallback Ĺ›wiadomie zaakceptowany).
[ ] UruchomiĹ‚em RUN_ALL.ps1 lub scripts\04_resolution_sweep.py.
[ ] UruchomiĹ‚em scripts\05_make_report.py.
[ ] OtworzyĹ‚em reports\fastpath_lab_report.html.
[ ] OdczytaĹ‚em FPS dla 1088Ă—1920.
[ ] OdczytaĹ‚em najwiÄ™kszy input dla stabilnych 50 FPS.
[ ] SprawdziĹ‚em, czy fallback_used = false.
[ ] SprawdziĹ‚em, czy zero_copy_to_inference = true.
[ ] ZachowaĹ‚em results/ i reports/ do dalszej analizy.
```

---

## 20. Finalny mini-przewodnik w 5 liniach

JeĹ›li chcesz **tylko odpowiedĹş na dwa pytania** i nic wiÄ™cej:

1. Uruchom `.\CHECK_ENV.ps1`.
2. Uruchom `.\RUN_ALL.ps1`.
3. OtwĂłrz `reports\fastpath_lab_report.html`.
4. PrzejdĹş do sekcji **Direct answers**.
5. Odczytaj:
   - Full HD-like FPS (Q1),
   - largest resolution for stable 50 FPS (Q2).

Koniec.


