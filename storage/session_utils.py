
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys

try:
    # lokalny import â€“ nie crashuj jeśli podczas budowania exe nie ma konfiguracji
    from core.config_menager import load_config
except Exception:
    def load_config() -> dict:
        return {}

# Folder aplikacji (dziaĹ‚a z .py i z .exe/Onefile dziÄ™ki _MEIPASS)
APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
_DEFAULT_SESSIONS = APP_DIR / "sessions"

_active_session: Path | None = None

def get_sessions_root() -> Path:
    """
    Zwraca katalog 'sessions' zgodny z konfiguracjÄ…:
    - jeśli config['path2save'] koĹ„czy się już na 'sessions' â†’ użyj go,
    - jeśli nie â†’ doĹ‚ĂłĹĽ 'sessions',
    - jeśli brak configu â†’ APP_DIR/sessions
    """
    cfg = load_config() or {}
    raw = str(cfg.get("path2save") or "").strip()
    if not raw:
        return _DEFAULT_SESSIONS
    p = Path(raw)
    return p if p.name.lower() == "sessions" else (p / "sessions")

def ensure_active_session() -> Path:
    """
    Gwarantuje, ĹĽe istnieje aktywny katalog sesji wraz z podfolderem snapshots.
    """
    global _active_session
    if _active_session is None:
        root = get_sessions_root()
        root.mkdir(parents=True, exist_ok=True)
        _active_session = root / f"session_{datetime.now():%Y%m%d_%H%M%S}"
        _active_session.mkdir(parents=True, exist_ok=True)
        (_active_session / "snapshots").mkdir(parents=True, exist_ok=True)
    return _active_session

def set_active_session(path: str | Path) -> Path:
    """
    Ustaw zewnÄ™trznie wybrany katalog sesji (np. z GUI). Katalog zostanie utworzony.
    """
    global _active_session
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (_active_session or p)  # dla mypy
    _active_session = p
    (_active_session / "snapshots").mkdir(parents=True, exist_ok=True)
    return _active_session

def get_active_session() -> Path | None:
    return _active_session

