# # log_setup.py
#
# import os, sys, logging, logging.handlers, tempfile, datetime
#
# def _is_writable_dir(path: str) -> bool:
#     try:
#         os.makedirs(path, exist_ok=True)
#         test = os.path.join(path, ".write_test")
#         with open(test, "w", encoding="utf-8") as f:
#             f.write("ok")
#         os.remove(test)
#         return True
#     except Exception:
#         return False
#
# def get_app_dir():
#     # katalog z .exe (PyInstaller) albo z .py (dev) — NIE _MEIPASS
#     if getattr(sys, "frozen", False):
#         return os.path.dirname(sys.executable)
#     return os.path.dirname(os.path.abspath(__file__))
#
# def get_log_dir(app_name="VolleyHub"):
#     # 1) spróbuj obok .exe w podkatalogu logs/
#     d1 = os.path.join(get_app_dir(), "logs")
#     if _is_writable_dir(d1):
#         return d1
#     # 2) fallback: %LOCALAPPDATA%\VolleyHub\logs
#     local = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
#     d2 = os.path.join(local, app_name, "logs")
#     if _is_writable_dir(d2):
#         return d2
#     # 3) ostatecznie: temp
#     d3 = os.path.join(tempfile.gettempdir(), app_name, "logs")
#     os.makedirs(d3, exist_ok=True)
#     return d3

# log_setup.py
import logging, os, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_dir: str | os.PathLike, level=logging.INFO):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logfile = Path(log_dir) / "volleyhub.log"

    fmt = "%(asctime)s | %(processName)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    ]

    # Konsola tylko, gdy naprawdę istnieje strumień i jawnie chcemy (zmienna env)
    if os.environ.get("VOLLEYHUB_CONSOLE", "0") == "1" and (sys.stderr is not None or sys.stdout is not None):
        handlers.append(logging.StreamHandler(sys.stderr or sys.stdout))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)

    # Przytnij hałas bibliotek
    for noisy in ("asyncio", "aiortc", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("LOG start → %s", logfile)
    return str(logfile)

