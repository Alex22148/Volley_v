# VolleyHub.spec
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

project_dir = Path(".").resolve()


def D(name: str):
    """Dołącz plik z katalogu projektu (trafi do exe / _MEIPASS)."""
    return (str(project_dir / name), ".")


# -----------------------------
#  DANE STATYCZNE (wewnątrz exe)
# -----------------------------
datas = [
    D("no_signal1.png"),          # <-- upewnij się, że nazwa pliku się zgadza
    D("index.html"),
    D("app.js"),
    D("layout.png"),
    D("camera_roles.json"),
    D("camera_basic_parameters.json"),
    D("app_config.json"),

]

binaries = []
hiddenimports = []

# -----------------------------
#  ZBIERANIE PAKIETÓW
# -----------------------------
for pkg in ["customtkinter", "pypylon", "aiortc", "aiohttp", "av", "cv2"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# dynamiczne biblioteki (DLL/so) od OpenCV i PyAV
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("av")

block_cipher = None

# -----------------------------
#  ANALIZA
# -----------------------------
a = Analysis(
    ["main.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# -----------------------------
#  EXE (ONEFILE)
# -----------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,   # <-- WAŻNE: dokładamy binarki
    a.zipfiles,
    a.datas,      # <-- WAŻNE: dokładamy datas
    [],
    name="VolleyHub_v3.5",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    console=True,   # lub False jeśli nie chcesz konsoli
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
