// /static/app.js
const sel = document.getElementById("roleSelect");
const img = document.getElementById("preview");
// 🔒 Zablokuj natywne przeciąganie obrazka
img.setAttribute("draggable", "false");
img.style.webkitUserDrag = "none";   // Safari/Chrome
img.style.userSelect = "none";       // na wszelki wypadek

// Nieważne "dragstart" (Chrome/Firefox) — nie ciągnij ghost-image
img.addEventListener("dragstart", (e) => e.preventDefault());

const selLabel = document.getElementById("selLabel");
const statusBox = document.getElementById("status");
const exp = document.getElementById("exp");
const gain = document.getElementById("gain");
const expVal = document.getElementById("expVal");
const gainVal = document.getElementById("gainVal");
let previewInterval = null;
let frameInFlight = false;

let roles = [];
let selected = null;
let tick = 0;

// ==== ROI zoom state ====
let mode = "full"; // "full" | "roi"
let roi = { x: 0, y: 0, w: 1, h: 1 }; // współrzędne znormalizowane (0..1)
let zoomFactor = 2.5; // domyślny mnożnik przy kliknięciu (ustawiany też sliderem)
let drag = null; // {start:{nx,ny}, end:{nx,ny}} podczas zaznaczania
let suppressClickOnce = false;
let roiOverlay = null; // podgląd prostokąta podczas drag


// ✅ Nowa wersja z retry (czeka, aż backend zgłosi dostępne role)
async function fetchRoles(retries = 8, delayMs = 1000) {
  for (let i = 0; i < retries; i++) {
    try {
      const r = await fetch("/api/roles");
      const data = await r.json();
      if (data.roles && data.roles.length > 0) {
        roles = data.roles;
        selected = data.selected || roles[0];

        sel.innerHTML = "";
        roles.forEach((role) => {
          const opt = document.createElement("option");
          opt.value = role;
          opt.textContent = role;
          if (role === selected) opt.selected = true;
          sel.appendChild(opt);
        });
        selLabel.textContent = selected || "—";

        if (selected) {
          await selectRole(selected);
        }

        console.log("[UI] ✅ Roles fetched:", roles);
        return;
      }
    } catch (e) {
      console.warn("[UI] ⚠️ fetchRoles attempt failed:", e);
    }
    console.log(`[UI] ⏳ Waiting for roles (attempt ${i + 1}/${retries})...`);
    await new Promise((r) => setTimeout(r, delayMs));
  }
  console.error("[UI] ❌ Nie udało się pobrać listy ról z serwera.");
}



function forceRefresh() {
  if (selected) {
    const { outW, outH } = getOutputSize();
    img.src = `/api/frame?role=${encodeURIComponent(selected)}&out_w=${outW}&out_h=${outH}&x=${Date.now()}&t=${tick++}`;

  }
}

function getOutputSize() {
  const r = img.getBoundingClientRect();
  let outW = Math.max(1, Math.round(r.width));
  let outH = Math.max(1, Math.round(r.height));
  const maxW = 960;
  if (outW > maxW) {
    const s = maxW / outW;
    outW = Math.max(1, Math.round(outW * s));
    outH = Math.max(1, Math.round(outH * s));
  }
  return { outW, outH };
}

async function selectRole(role) {
  await fetch("/api/select_camera", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role }),
  });
  selected = role;
  selLabel.textContent = selected || "—";
  tick = 0;
  frameInFlight = false;
  startPreviewLoop(); // 👈 restart podglądu tylko dla nowej kamery
}



async function sendExposure(value) {
  if (!selected) return;
  await fetch("/api/exposure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role: selected, value }),
  });
}

async function sendGain(value) {
  if (!selected) return;
  await fetch("/api/gain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role: selected, value }),
  });
}

function startPreviewLoop(fps = 12) {
  if (previewInterval !== null) {
    clearInterval(previewInterval); // zatrzymaj poprzednią pętlę
  }

  const interval = 1000 / fps;
   previewInterval = setInterval(() => {
    if (!selected) return;
    if (frameInFlight) return;
    frameInFlight = true;
    const { outW, outH } = getOutputSize();
    const cacheBust = `t=${tick++}&x=${Date.now()}`;
    const url =
      mode === "full"
        ? `/api/frame?role=${encodeURIComponent(selected)}&out_w=${outW}&out_h=${outH}&${cacheBust}`
        : `/api/roi?role=${encodeURIComponent(selected)}&x=${roi.x}&y=${roi.y}&w=${roi.w}&h=${roi.h}&out_w=${outW}&out_h=${outH}&${cacheBust}`;
    const onDone = () => {
      frameInFlight = false;
    };
    img.onload = onDone;
    img.onerror = onDone;
    img.src = url;
  }, interval);
}



sel.addEventListener("change", (e) => {
  selectRole(e.target.value);
});

exp.addEventListener("input", (e) => {
  expVal.textContent = `${parseFloat(e.target.value).toFixed(1)} µs`;
});
exp.addEventListener("change", (e) => {
  sendExposure(parseFloat(e.target.value));
});

gain.addEventListener("input", (e) => {
  gainVal.textContent = `${parseFloat(e.target.value).toFixed(1)} dB`;
});
gain.addEventListener("change", (e) => {
  sendGain(parseFloat(e.target.value));
});

(async function init() {
  try {
    await fetchRoles(); // 👈 odpala pętlę przez selectRole()
    await loadYoloStats(); // 👈 load YOLO state
    statusBox.textContent = "Połączono z serwerem.";
  } catch (e) {
    statusBox.textContent = "Błąd połączenia z serwerem.";
  }
})();


// === 🔍 ROI zoom (klik albo zaznacz prostokątem) ===
// mapowanie kliknięcia na współrzędne znormalizowane 0..1, uwzględnia "letterbox"
function pointToNormalized(e) {
  const rect = img.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;
  // aktualnie wyświetlana bitmapa (z /api/frame lub /api/roi) ma proporcje:
  const ar = (img.naturalWidth || cw) / (img.naturalHeight || ch);
  const drawnW = Math.min(cw, ch * ar);
  const drawnH = Math.min(ch, cw / ar);
  const offX = (cw - drawnW) / 2;
  const offY = (ch - drawnH) / 2;
  const px = e.clientX - rect.left - offX;
  const py = e.clientY - rect.top - offY;
  if (px < 0 || py < 0 || px > drawnW || py > drawnH) return null; // poza obrazem
  return { nx: px / drawnW, ny: py / drawnH };
}

function getDrawnMetrics() {
  const rect = img.getBoundingClientRect();
  const cw = rect.width, ch = rect.height;
  const nw = img.naturalWidth || cw, nh = img.naturalHeight || ch;
  const scale = Math.min(cw / nw, ch / nh);
  const drawnW = nw * scale;
  const drawnH = nh * scale;
  const offX = (cw - drawnW) / 2;
  const offY = (ch - drawnH) / 2;
  return { drawnW, drawnH, offX, offY };
}

function currentBase() {
  // jeśli już jesteś w ROI – kolejne przybliżenia liczymy wewnątrz niego
  return (mode === "roi") ? roi : { x: 0, y: 0, w: 1, h: 1 };
}

// przybliż wokół punktu ekranu (p.nx,p.ny ∈ [0..1] względem wyświetlanej bitmapy)
function zoomAroundPoint(p, factor) {
  const base = currentBase();
  const rect = img.getBoundingClientRect();
  const ar = rect.width / rect.height; // zachowujemy proporcje okna

  // rozmiar nowego ROI RELATYWNIE do bieżącej bazy
  let wRel = 1 / factor;
  let hRel = wRel / ar;
  if (hRel > 1) { hRel = 1; wRel = hRel * ar; } // guard

  // zamiana na współrzędne względem pełnej klatki
  const wFull = wRel * base.w;
  const hFull = hRel * base.h;
  let xFull = base.x + p.nx * base.w - wFull / 2;
  let yFull = base.y + p.ny * base.h - hFull / 2;

  // clamp wewnątrz base (i 0..1)
  xFull = Math.max(base.x, Math.min(xFull, base.x + base.w - wFull));
  yFull = Math.max(base.y, Math.min(yFull, base.y + base.h - hFull));

  roi = { x: xFull, y: yFull, w: wFull, h: hFull };
  mode = "roi";
}


// pomocniczo: pokaż prostokąt podczas drag
function ensureOverlay() {
  if (roiOverlay) return roiOverlay;
  roiOverlay = document.createElement("div");
  Object.assign(roiOverlay.style, {
    position: "absolute",
    border: "2px solid rgba(52,152,219,0.9)",
    background: "rgba(52,152,219,0.15)",
    pointerEvents: "none",
    display: "none",
    zIndex: 5
  });
  // kontenerem jest rodzic <img> (ma position: relative z Tailwinda/aspect-video)
  img.parentElement.style.position = "relative";
  img.parentElement.appendChild(roiOverlay);
  return roiOverlay;
}

// klik = powiększ wokół punktu wg zoomFactor (suwak)
img.addEventListener("click", (e) => {
  // jeśli chwilę wcześniej był drag, to klik kończący drag ignorujemy
  if (suppressClickOnce) { suppressClickOnce = false; return; }
  const p = pointToNormalized(e);
  if (!p) return;
  zoomAroundPoint(p, zoomFactor);
  const slider = document.getElementById("zoom-slider");
  if (slider) slider.value = Math.max(1, 1 / Math.min(roi.w, roi.h)).toFixed(1);
});

// drag = dowolny prostokąt
img.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  e.preventDefault(); // << stopuj natywne przeciąganie obrazka
  const p = pointToNormalized(e);
  if (!p) return;
  drag = { start: p, end: p };
  const overlay = ensureOverlay();
  overlay.style.display = "block";
});
document.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const p = pointToNormalized(e);
  if (!p) return;
  drag.end = p;
  // rysuj overlay
  const { drawnW, drawnH, offX, offY } = getDrawnMetrics();
  const x0 = Math.min(drag.start.nx, drag.end.nx);
  const y0 = Math.min(drag.start.ny, drag.end.ny);
  const x1 = Math.max(drag.start.nx, drag.end.nx);
  const y1 = Math.max(drag.start.ny, drag.end.ny);
  const overlay = ensureOverlay();
  overlay.style.left   = `${offX + x0 * drawnW}px`;
  overlay.style.top    = `${offY + y0 * drawnH}px`;
  overlay.style.width  = `${(x1 - x0) * drawnW}px`;
  overlay.style.height = `${(y1 - y0) * drawnH}px`;
});

document.addEventListener("mouseup", () => {
  if (!drag) return;
  const x0 = Math.min(drag.start.nx, drag.end.nx);
  const y0 = Math.min(drag.start.ny, drag.end.ny);
  const x1 = Math.max(drag.start.nx, drag.end.nx);
  const y1 = Math.max(drag.start.ny, drag.end.ny);
  const w = Math.max(0.01, x1 - x0);
  const h = Math.max(0.01, y1 - y0);
  const base = currentBase();
 // z ekranu (0..1) -> do pełnej klatki przez „base”
  const xFull = base.x + x0 * base.w;
  const yFull = base.y + y0 * base.h;
  const wFull = w * base.w;
  const hFull = h * base.h;
  roi = { x: xFull, y: yFull, w: wFull, h: hFull };
  mode = "roi";
  const overlay = ensureOverlay();
  overlay.style.display = "none";

  drag = null;
  suppressClickOnce = true; // zablokuj „klik” generowany po drag
  // podbij slider wg „mocy” zoomu
  const slider = document.getElementById("zoom-slider");
  if (slider) {
    const approx = Math.max(1, 1 / Math.min(roi.w, roi.h));  // ← z finalnego ROI

    slider.value = approx.toFixed(1);
    zoomFactor = approx;
  }
});


img.addEventListener("wheel", (e) => {
  e.preventDefault();
  const p = pointToNormalized(e);
  if (!p) return;
  const factor = e.deltaY < 0 ? 1.25 : (1 / 1.25); // w górę = przybliż
  zoomAroundPoint(p, factor);
  const slider = document.getElementById("zoom-slider");
  if (slider) slider.value = Math.max(1, 1 / Math.min(roi.w, roi.h)).toFixed(1);
}, { passive: false });


// Slider steruje „mocą” zoomu przy KLIKU (nie skaluje obrazu po kliencie)
document.getElementById("zoom-slider")?.addEventListener("input", (e) => {
  zoomFactor = Math.max(1, parseFloat(e.target.value) || 1);
});
// Reset przyciskiem – wróć do pełnej klatki
document.getElementById("reset-zoom")?.addEventListener("click", () => {
  mode = "full";
  roi = { x: 0, y: 0, w: 1, h: 1 };
  const slider = document.getElementById("zoom-slider");
  if (slider) slider.value = "1";
});
// Podwójny klik = też reset
img.addEventListener("dblclick", () => {
  mode = "full";
  roi = { x: 0, y: 0, w: 1, h: 1 };
  const slider = document.getElementById("zoom-slider");
  if (slider) slider.value = "1";
});

// ===== 🎯 YOLO FUNCTIONALITY =====
const yoloToggle = document.getElementById("yolo-toggle");
const yoloModelPath = document.getElementById("yolo-model-path");
const yoloLoadBtn = document.getElementById("yolo-load-btn");
const yoloClassId = document.getElementById("yolo-class-id");
const yoloConfidence = document.getElementById("yolo-confidence");
const yoloConfidenceVal = document.getElementById("yolo-confidence-val");
const yoloStatus = document.getElementById("yolo-status");

// Update confidence display
yoloConfidence?.addEventListener("input", (e) => {
  yoloConfidenceVal.textContent = parseFloat(e.target.value).toFixed(2);
});

// Toggle YOLO on/off
yoloToggle?.addEventListener("change", async (e) => {
  const enabled = e.target.checked;
  try {
    const response = await fetch("/api/yolo/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled })
    });
    
    const result = await response.json();
    if (result.ok) {
      yoloStatus.textContent = enabled ? "YOLO enabled ✅" : "YOLO disabled";
      yoloStatus.style.color = enabled ? "#22c55e" : "#6b7280";
    } else {
      yoloStatus.textContent = `Error: ${result.error}`;
      yoloStatus.style.color = "#ef4444";
      e.target.checked = !enabled; // revert
    }
  } catch (error) {
    yoloStatus.textContent = `Network error: ${error.message}`;
    yoloStatus.style.color = "#ef4444";
    e.target.checked = !enabled; // revert
  }
});

// Load custom YOLO model
yoloLoadBtn?.addEventListener("click", async () => {
  const modelPath = yoloModelPath.value.trim();
  if (!modelPath) {
    yoloStatus.textContent = "Please enter model path";
    yoloStatus.style.color = "#ef4444";
    return;
  }
  
  yoloLoadBtn.disabled = true;
  yoloLoadBtn.textContent = "Loading...";
  yoloStatus.textContent = "Loading model...";
  yoloStatus.style.color = "#3b82f6";
  
  try {
    const response = await fetch("/api/yolo/load_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_path: modelPath })
    });
    
    const result = await response.json();
    if (result.ok) {
      yoloStatus.textContent = `Model loaded: ${result.model_path} ✅`;
      yoloStatus.style.color = "#22c55e";
    } else {
      yoloStatus.textContent = `Load failed: ${result.error}`;
      yoloStatus.style.color = "#ef4444";
    }
  } catch (error) {
    yoloStatus.textContent = `Network error: ${error.message}`;
    yoloStatus.style.color = "#ef4444";
  } finally {
    yoloLoadBtn.disabled = false;
    yoloLoadBtn.textContent = "Load";
  }
});

// Set ball class ID
yoloClassId?.addEventListener("change", async (e) => {
  const classId = parseInt(e.target.value);
  try {
    const response = await fetch("/api/yolo/set_class_id", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_id: classId })
    });
    
    const result = await response.json();
    if (result.ok) {
      yoloStatus.textContent = `Ball class ID: ${result.class_id}`;
      yoloStatus.style.color = "#22c55e";
    } else {
      yoloStatus.textContent = `Error: ${result.error}`;
      yoloStatus.style.color = "#ef4444";
    }
  } catch (error) {
    yoloStatus.textContent = `Network error: ${error.message}`;
    yoloStatus.style.color = "#ef4444";
  }
});

// Set confidence threshold
yoloConfidence?.addEventListener("change", async (e) => {
  const confidence = parseFloat(e.target.value);
  try {
    const response = await fetch("/api/yolo/set_confidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence })
    });
    
    const result = await response.json();
    if (result.ok) {
      yoloStatus.textContent = `Confidence: ${result.confidence.toFixed(2)}`;
      yoloStatus.style.color = "#22c55e";
    } else {
      yoloStatus.textContent = `Error: ${result.error}`;
      yoloStatus.style.color = "#ef4444";
    }
  } catch (error) {
    yoloStatus.textContent = `Network error: ${error.message}`;
    yoloStatus.style.color = "#ef4444";
  }
});

// Load YOLO stats on page load
async function loadYoloStats() {
  try {
    const response = await fetch("/api/yolo/stats");
    const result = await response.json();
    
    if (result.ok && result.stats) {
      const stats = result.stats;
      yoloToggle.checked = stats.enabled || false;
      yoloClassId.value = stats.ball_class_id || 0;
      yoloConfidence.value = stats.confidence_threshold || 0.5;
      yoloConfidenceVal.textContent = (stats.confidence_threshold || 0.5).toFixed(2);
      
      if (stats.model_path) {
        yoloModelPath.value = stats.model_path;
      }
      
      yoloStatus.textContent = stats.enabled ? "YOLO enabled ✅" : "YOLO disabled";
      yoloStatus.style.color = stats.enabled ? "#22c55e" : "#6b7280";
    }
  } catch (error) {
    console.warn("Could not load YOLO stats:", error);
  }
}
