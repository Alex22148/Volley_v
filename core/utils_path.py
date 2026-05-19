# utils_path.py
import sys
from pathlib import Path

def resource_path(rel: str) -> str:
    """
    Zwraca absolutną ścieżkę zarówno w trybie dev, jak i w exe (PyInstaller onefile/onedir).
    rel: np. "index.html" albo "no_signal1.png"
    """
    if getattr(sys, "frozen", False):
        # onefile → dane w sys._MEIPASS
        # onedir  → też tu działa, PyInstaller też ustawia _MEIPASS
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parent.parent
    return str((base / rel).resolve())
