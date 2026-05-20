# config_menager.py
import json, threading
_CFG_PATH = None
_LOCK = threading.Lock()

def set_config_path(p):  # wołane z main.py
    global _CFG_PATH; _CFG_PATH = p

def _defaults():
    return {"path2save":"", "per_role":{}, "per_camera":{}, "exposure_val":None, "gain_val":None}

def load_config(path=None):
    path = path or _CFG_PATH or "app_config.json"
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except FileNotFoundError:
        data = _defaults()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data

def _deep_merge(dst, src):
    for k,v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict): _deep_merge(dst[k], v)
        else: dst[k] = v
    return dst

def save_config(update: dict, path=None):
    path = path or _CFG_PATH or "app_config.json"
    with _LOCK:
        cfg = load_config(path)
        _deep_merge(cfg, update)
        with open(path, "w", encoding="utf-8") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
    return True
