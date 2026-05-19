from pathlib import Path
import struct
import json
from typing import Generator, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# ====== FORMAT BFRM (jak w saver_worker_bin) ======
HEADER_FMT = "<4sIIIQI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
HEADER_TAG = b"BFRM"

# ====== PYLON DEMOSAIK (BayerRG8 -> BGR) ======
try:
    from pypylon import pylon  # type: ignore
except Exception:
    pylon = None  # type: ignore

_conv = None
_pylon_img = None
_pylon_last_shape = (0, 0)


def bayer_to_bgr(bayer_rg8: np.ndarray) -> np.ndarray:
    global _conv, _pylon_img, _pylon_last_shape, pylon

    H, W = bayer_rg8.shape[:2]

    if pylon is None:
        return np.repeat(bayer_rg8[..., None], 3, axis=2)

    try:
        if _conv is None:
            _conv = pylon.ImageFormatConverter()
            _conv.OutputPixelFormat = pylon.PixelType_BGR8packed
            _conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        if _pylon_img is None or _pylon_last_shape != (H, W):
            _pylon_img = pylon.PylonImage()
            _pylon_img.Create(W, H, pylon.PixelType_BayerRG8)
            _pylon_last_shape = (H, W)

        dst = _pylon_img.GetArray()
        if not bayer_rg8.flags.get("C_CONTIGUOUS", False):
            bayer_rg8 = np.ascontiguousarray(bayer_rg8)
        dst[:, :] = bayer_rg8

        col = _conv.Convert(_pylon_img)
        bgr = col.GetArray()
        return np.asarray(bgr).copy()
    except Exception:
        return np.repeat(bayer_rg8[..., None], 3, axis=2)


def iter_bin_records(bin_path: str | Path) -> Generator[Tuple[int, int, int, int, int, bytes], None, None]:
    path = Path(bin_path)
    idx = 0
    with path.open("rb") as f:
        while True:
            header = f.read(HEADER_SIZE)
            if not header:
                break
            if len(header) < HEADER_SIZE:
                raise ValueError(f"{path}: ucięty nagłówek w rekordzie {idx}")

            tag, w, h, c, ts_ns, payload_nbytes = struct.unpack(HEADER_FMT, header)
            if tag != HEADER_TAG:
                raise ValueError(f"{path}: zły TAG nagłówka (oczekiwano BFRM, mamy {tag!r})")

            payload = f.read(payload_nbytes)
            if len(payload) < payload_nbytes:
                raise ValueError(f"{path}: ucięty payload w rekordzie {idx}")

            yield idx, w, h, c, ts_ns, payload
            idx += 1


def payload_to_bgr(payload: bytes, w: int, h: int, c: int) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)

    if c == 1:
        bayer = arr.reshape((h, w))
        return bayer_to_bgr(bayer)

    img = arr.reshape((h, w, c))
    if c == 3:
        return img
    if c > 3:
        return img[:, :, :3]
    pad = [img] + [img[:, :, :1]] * (3 - c)
    return np.concatenate(pad, axis=2)


def extract_timestamps_from_bin(bin_path: str | Path) -> List[Dict]:
    timestamps = []
    for idx, w, h, c, ts_ns, _ in iter_bin_records(bin_path):
        timestamps.append({
            "frame_idx": idx,
            "ts_ns": ts_ns,
            "ts_s": ts_ns / 1e9
        })
    return timestamps


def load_session_timestamps(session_dir: str | Path) -> Dict[str, List[Dict]]:
    session_dir = Path(session_dir)
    cameras = {}

    for role_dir in session_dir.iterdir():
        if not role_dir.is_dir() or role_dir.name in ("snapshots", "webp"):
            continue

        role = role_dir.name
        all_timestamps = []

        for bin_file in sorted(role_dir.glob("*.bin")):
            all_timestamps.extend(extract_timestamps_from_bin(bin_file))

        if all_timestamps:
            cameras[role] = all_timestamps

    return cameras


def compute_sync_metrics(cameras: Dict[str, List[Dict]], ref_role: str | None = None) -> Dict:
    if not cameras:
        raise ValueError("Brak danych kamer")

    roles = sorted(cameras.keys())
    if ref_role is None:
        ref_role = roles[0]
    if ref_role not in cameras:
        raise ValueError(f"Brak kamery referencyjnej: {ref_role}")

    result = {
        "reference_role": ref_role,
        "per_camera": {},
        "pairwise_offsets": {},
    }

    # metryki per kamera
    for role, ts_list in cameras.items():
        ts_ns = np.array([x["ts_ns"] for x in ts_list], dtype=np.int64)

        intervals = np.diff(ts_ns) if len(ts_ns) > 1 else np.array([], dtype=np.int64)
        avg_interval_ns = float(np.mean(intervals)) if len(intervals) else None
        std_interval_ns = float(np.std(intervals)) if len(intervals) else None
        fps = float(1e9 / avg_interval_ns) if avg_interval_ns and avg_interval_ns > 0 else None

        result["per_camera"][role] = {
            "n_frames": int(len(ts_ns)),
            "first_ts_ns": int(ts_ns[0]) if len(ts_ns) else None,
            "last_ts_ns": int(ts_ns[-1]) if len(ts_ns) else None,
            "avg_interval_ns": avg_interval_ns,
            "std_interval_ns": std_interval_ns,
            "avg_fps": fps,
        }

    ref_ts = np.array([x["ts_ns"] for x in cameras[ref_role]], dtype=np.int64)

    # offsety względem referencji
    for role in roles:
        role_ts = np.array([x["ts_ns"] for x in cameras[role]], dtype=np.int64)
        n = min(len(ref_ts), len(role_ts))
        if n == 0:
            continue

        offsets_ns = role_ts[:n] - ref_ts[:n]

        mean_ns = float(np.mean(offsets_ns))
        std_ns = float(np.std(offsets_ns))
        min_ns = int(np.min(offsets_ns))
        max_ns = int(np.max(offsets_ns))
        drift_ns = int(max_ns - min_ns)

        result["pairwise_offsets"][role] = {
            "n_compared": int(n),
            "mean_offset_ns": mean_ns,
            "mean_offset_ms": mean_ns / 1e6,
            "std_offset_ns": std_ns,
            "std_offset_ms": std_ns / 1e6,
            "min_offset_ns": min_ns,
            "max_offset_ns": max_ns,
            "drift_ns": drift_ns,
            "drift_ms": drift_ns / 1e6,
            "is_good": std_ns / 1e6 < 0.5,
            "is_ok": std_ns / 1e6 < 2.0,
        }

    return result


def plot_sync_report(cameras: Dict[str, List[Dict]], out_dir: str | Path, ref_role: str | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roles = sorted(cameras.keys())
    if not roles:
        return

    if ref_role is None:
        ref_role = roles[0]

    ref_ts = np.array([x["ts_ns"] for x in cameras[ref_role]], dtype=np.int64)
    # 1. offset vs czas
    plt.figure(figsize=(12, 6))
    for role in roles:
        role_ts = np.array([x["ts_ns"] for x in cameras[role]], dtype=np.int64)
        n = min(len(ref_ts), len(role_ts))
        if n == 0:
            continue
        offsets_ms = (role_ts[:n] - ref_ts[:n]) / 1e6
        t_s = (ref_ts[:n] - ref_ts[0]) / 1e9
        plt.plot(t_s, offsets_ms, label=role, linewidth=1)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Czas [s]")
    plt.ylabel("Offset względem referencji [ms]")
    plt.title(f"Offset czasowy względem {ref_role}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "offset_vs_time.png", dpi=160)
    plt.close()

    # 2. histogram offsetów
    plt.figure(figsize=(12, 6))
    for role in roles:
        role_ts = np.array([x["ts_ns"] for x in cameras[role]], dtype=np.int64)
        n = min(len(ref_ts), len(role_ts))
        if n == 0:
            continue
        offsets_ms = (role_ts[:n] - ref_ts[:n]) / 1e6
        plt.hist(offsets_ms, bins=80, alpha=0.5, label=role)
    plt.xlabel("Offset względem referencji [ms]")
    plt.ylabel("Liczba próbek")
    plt.title(f"Histogram offsetów względem {ref_role}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "offset_histogram.png", dpi=160)
    plt.close()

    # 3. interwały między klatkami
    plt.figure(figsize=(12, 6))
    for role in roles:
        role_ts = np.array([x["ts_ns"] for x in cameras[role]], dtype=np.int64)
        if len(role_ts) < 2:
            continue
        intervals_ms = np.diff(role_ts) / 1e6
        t_s = (role_ts[1:] - role_ts[0]) / 1e9
        plt.plot(t_s, intervals_ms, label=role, linewidth=1)
    plt.xlabel("Czas [s]")
    plt.ylabel("Interwał między klatkami [ms]")
    plt.title("Stabilność FPS / interwałów")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "frame_intervals.png", dpi=160)
    plt.close()

    # 4. drift po odjęciu średniego offsetu
    plt.figure(figsize=(12, 6))
    for role in roles:
        role_ts = np.array([x["ts_ns"] for x in cameras[role]], dtype=np.int64)
        n = min(len(ref_ts), len(role_ts))
        if n == 0:
            continue
        offsets_ms = (role_ts[:n] - ref_ts[:n]) / 1e6
        centered_ms = offsets_ms - np.mean(offsets_ms)
        t_s = (ref_ts[:n] - ref_ts[0]) / 1e9
        plt.plot(t_s, centered_ms, label=role, linewidth=1)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Czas [s]")
    plt.ylabel("Drift po odjęciu średniego offsetu [ms]")
    plt.title("Dryft synchronizacji")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "drift_vs_time.png", dpi=160)
    plt.close()


def export_session_timestamps(session_dir: str | Path, out_json: str | Path = None) -> Dict:
    session_dir = Path(session_dir)
    if out_json is None:
        out_json = session_dir / "timestamps.json"

    cameras = load_session_timestamps(session_dir)

    result = {
        "session": session_dir.name,
        "cameras": cameras,
    }

    sync_analysis = compute_sync_metrics(cameras)
    result["sync_analysis"] = sync_analysis

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    plots_dir = session_dir / "sync_plots"
    plot_sync_report(cameras, plots_dir, ref_role=sync_analysis["reference_role"])

    print(f"[TIMESTAMPS] ✅ Zapisano JSON: {out_json}")
    print(f"[TIMESTAMPS] ✅ Zapisano wykresy: {plots_dir}")

    print("\n[SYNC] Podsumowanie:")
    for role, metrics in sync_analysis["pairwise_offsets"].items():
        print(
            f"  {role}: "
            f"mean={metrics['mean_offset_ms']:+.3f} ms, "
            f"std={metrics['std_offset_ms']:.3f} ms, "
            f"drift={metrics['drift_ms']:.3f} ms"
        )

    return result


if __name__ == "__main__":
    export_session_timestamps(
        session_dir=r"E:\sessions\session_20260311_131509",
        out_json=r"E:\sessions\session_20260311_131509\sync_report.json"
    )
