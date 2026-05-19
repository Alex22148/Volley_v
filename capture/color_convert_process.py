import multiprocessing as mp
import numpy as np
_color_procs_started = False
_color_procs_lock = mp.Lock()

# --- FPS governor: stabilne tempo bez driftu ---
class FpsLimiter:
    def __init__(self, fps):
        self.set_fps(fps)
        self._next = None

    def set_fps(self, fps):
        # nie schodzimy poniżej 0.5 FPS
        self.fps = max(0.5, float(fps))
        self.dt = 1.0 / self.fps

    def allow(self, now):
        """Zwraca True jeśli wolno wypuĹ›ciÄ‡ nastÄ™pnÄ… klatkÄ™.
           Ustawia nastÄ™pne 'okno czasowe' z kompensacjÄ… zalegĹ‚oĹ›ci."""
        if self._next is None:
            self._next = now
        if now + 1e-9 >= self._next:
            # jeśli jesteĹ›my spĂłĹşnieni, doskocz do najbliĹĽszego slotu w przyszĹ‚oĹ›ci
            missed = int((now - self._next) // self.dt)
            self._next = (self._next + (missed + 1) * self.dt)
            return True
        return False

    def sleep_until_next(self, now, max_sleep=0.01):
        """Opcjonalny, lekki sen do nastÄ™pnego slotu."""
        t = self._next - now if self._next is not None else 0.0
        if t > 0:
            # na Windows 10 sensownie trzymaÄ‡ max ~10ms
            import time
            time.sleep(min(t, max_sleep))

def downscale_by_stride(img: np.ndarray, factor: int) -> np.ndarray:
    return img[::factor, ::factor, :]




import multiprocessing as mp
import queue
import time
import numpy as np


def universal_color_processor(raw_bayer_q,
                              output_queues,
                              processing_config,
                              stop_evt,
                              stats_q=None,
                              remote_evt=None,
                              shared_state=None):
    """
    WejĹ›cie:
        raw_bayer_q: kolejka z krotkami (role, bayer_key, ts_ns)
                     bayer_key to klucz do ramki BayerRG8 w SharedMemoryManager.

    WyjĹ›cia (poprzez output_queues):
        {
            "preview": {role: Queue} LUB Queue z krotkami (role, shm_key, ts_ns),
            "zoom":    Queue z krotkami (role, frame_bgr, ts_ns),
            # remote uĹĽywa shared_state["current_key"] / ["full_key"]
        }

    Konfiguracja:
        processing_config = {
            "global": {"fps_cap": 20},
            "preview": {"fps":12,"scale":0.35,"color_format":"RGB","send_every_n":1},
            "remote":  {"fps":10,"scale":0.5, "color_format":"BGR","send_every_n":1},
        }
    """
    import platform
    from storage.shared_memory_manager import get_shared_memory_manager
    import cv2
    from pypylon import pylon
    
    # Initialize SharedMemoryManager
    smm = get_shared_memory_manager()

    # ---------- DEMOSAIK: Pylon -> fallback OpenCV -> szary ----------
    _conv = pylon.ImageFormatConverter()
    _conv.OutputPixelFormat = pylon.PixelType_BGR8packed
    _conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    _pylon_img = None
    _last_shape = (0, 0)  # (H, W)

    def bayer_to_bgr(bayer_rg8: np.ndarray) -> np.ndarray:
        """
        Demosaik BayerRG8 -> BGR8.
        1) prĂłbujemy PYLON
        2) fallback: OpenCV (cv2.COLOR_BayerRG2BGR)
        3) ostatecznie: 3x szary
        """
        nonlocal _pylon_img, _last_shape
        H, W = bayer_rg8.shape[:2]

        # --- próba przez Pylon ---
        try:
            if _pylon_img is None or _last_shape != (H, W):
                _pylon_img = pylon.PylonImage()
                _pylon_img.Create(W, H, pylon.PixelType_BayerRG8)
                _last_shape = (H, W)

            if not bayer_rg8.flags["C_CONTIGUOUS"]:
                bayer_rg8 = np.ascontiguousarray(bayer_rg8)

            dst = _pylon_img.GetArray()
            dst[:, :] = bayer_rg8

            col = _conv.Convert(_pylon_img)
            bgr = col.GetArray()
            return bgr
        except Exception as e:
            # --- fallback: OpenCV bez pylon ---
            try:
                if bayer_rg8.ndim == 3 and bayer_rg8.shape[2] == 1:
                    bayer_rg8 = bayer_rg8[:, :, 0]
                return cv2.cvtColor(bayer_rg8, cv2.COLOR_BayerRG2BGR)
            except Exception as e2:
                # ostatecznie: 3x szary, ĹĽeby GUI nie padĹ‚o
                return np.repeat(bayer_rg8[..., None], 3, axis=2)


    # ---------- lokalny resize (jeśli chcesz bez cv2, moĹĽesz uĹĽyÄ‡ NN) ----------
    def resize_rgb_nn(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        """Nearest-neighbour dla RGB/BGR; szybki i prosty."""
        h, w = img.shape[:2]
        if new_w <= 0 or new_h <= 0:
            raise ValueError("new_w/new_h must be > 0")
        if new_w == w and new_h == h:
            return img
        y_idx = np.linspace(0, h - 1, new_h).astype(np.int32)
        x_idx = np.linspace(0, w - 1, new_w).astype(np.int32)
        return img[np.ix_(y_idx, x_idx)]

    # ---------- konfiguracja z processing_config ----------
    cfg_prev   = processing_config.get("preview", {"scale": 1.0, "fps": 12, "color_format": "RGB", "send_every_n": 1})
    cfg_rem    = processing_config.get("remote",  {"scale": 0.5, "fps": 10, "color_format": "BGR", "send_every_n": 1})
    cfg_global = processing_config.get("global", {})
    global_cap = float(cfg_global.get("fps_cap", 0) or 0.0)

    is_win10 = platform.system().lower().startswith("win") and platform.release() == "10"

    prev_fps = float(cfg_prev.get("fps", 12))
    rem_fps  = float(cfg_rem.get("fps", 8))
    if global_cap > 0:
        prev_fps = min(prev_fps, global_cap)
        rem_fps = min(rem_fps, global_cap)
    if is_win10:
        # lekko przycinamy na Windows 10
        prev_fps = min(prev_fps, 15.0)
        rem_fps  = min(rem_fps,  10.0)

    class FpsLimiter:
        def __init__(self, fps: float):
            self._next = None
            self.set_fps(fps)

        def set_fps(self, fps: float):
            self.fps = max(0.5, float(fps) or 0.5)
            self.dt = 1.0 / self.fps

        def allow(self, now: float) -> bool:
            if self._next is None:
                self._next = now
                return True
            if now + 1e-9 >= self._next:
                missed = int((now - self._next) // self.dt)
                self._next = self._next + (missed + 1) * self.dt
                return True
            return False

        def sleep_until_next(self, now: float, max_sleep: float = 0.01):
            if self._next is None:
                return
            t = self._next - now
            if t > 0:
                time.sleep(min(t, max_sleep))

    prev_limiters, rem_limiters,  = {}, {}
    cnt_prev, cnt_rem = {}, {}

    # PrzekaĹĽ stats_q do YOLO dla raportĂłw wydajnoĹ›ci
    try:
        import core.utils_config as utils_config
        utils_config.stats_q = stats_q
    except Exception:
        pass
    
    # shared_state: minimalne odĹ›wieĹĽanie kluczy (WebRTC)
    shared_min_dt = 0.05
    last_shared_ts = 0.0
    last_current_key, last_full_key = {}, {}

    # statystyki
    frames_in = frames_demosaic = frames_prev = frames_rem = frames_zoom  = 0
    raw_dropped = 0
    stats_last_t = time.time()

    while not stop_evt.is_set():
        try:
            role, bayer_key, ts_ns = raw_bayer_q.get(timeout=0.02)
            frames_in += 1
            try:
                while True:
                    if raw_bayer_q.qsize() <= 2:
                        break
                    _ = raw_bayer_q.get_nowait()
                    raw_dropped += 1
            except (queue.Empty, queue.Full):
                pass
        except queue.Empty:
            # szansa na statystyki
            now_stats = time.time()
            if stats_q is not None and (now_stats - stats_last_t) >= 1.0:
                msg = {
                    "type": "color_stats",
                    "frames_in": frames_in,
                    "frames_demosaic": frames_demosaic,
                    "frames_prev": frames_prev,
                    "frames_rem": frames_rem,
                    "frames_zoom": frames_zoom,
                    "raw_dropped": raw_dropped,
                }
                try:
                    stats_q.put_nowait(msg)
                except (queue.Full, AttributeError):
                    pass
                stats_last_t = now_stats
            time.sleep(0.005)
            continue

        data = smm.read_frame(bayer_key)
        if data is None:
            continue
        bayer = data[0] if isinstance(data, tuple) else data
        H, W = bayer.shape[:2]

        allow_prev = False
        allow_rem = False
        zoom_active = False
        now_prev = time.perf_counter()
        now_rem = None

        # --- PREVIEW: na podstawie cfg_prev ---
        try:
            cnt_prev[role] = cnt_prev.get(role, 0) + 1
            every_n_prev = int(max(1, cfg_prev.get("send_every_n", 1)))
            allow_by_n_prev = (cnt_prev[role] % every_n_prev) == 0
            plim = prev_limiters.get(role)
            if plim is None:
                plim = prev_limiters.setdefault(role, FpsLimiter(prev_fps))
            else:
                plim.set_fps(prev_fps)
            if allow_by_n_prev and plim.allow(now_prev):
                allow_prev = True
        except Exception:
            allow_prev = False

        # --- ZOOM: aktywny tylko dla wybranej roli ---
        try:
            zoom_role = shared_state.get("zoom_role", "") if shared_state else ""
            zoom_active = bool(output_queues.get("zoom") and zoom_role and zoom_role == role)
        except Exception:
            zoom_active = False

        # --- REMOTE: globalny przeĹ‚Ä…cznik + selected_role ---
        try:
            if remote_evt is not None:
                remote_active = remote_evt.is_set()
            else:
                from core.utils_config import remote_stream_enabled
                remote_active = remote_stream_enabled.is_set()
        except Exception:
            remote_active = False

        selected_role = None
        if shared_state is not None:
            try:
                selected_role = shared_state.get("selected_role")
            except Exception:
                selected_role = None

        # --- REMOTE: globalny przeĹ‚Ä…cznik + selected_role ---
        try:
            if remote_evt is not None:
                remote_active = remote_evt.is_set()
            else:
                from core.utils_config import remote_stream_enabled
                remote_active = remote_stream_enabled.is_set()
        except Exception:
            remote_active = False

        selected_role = None
        if shared_state is not None:
            try:
                selected_role = shared_state.get("selected_role")
            except Exception:
                selected_role = None

        if remote_active and selected_role and role == selected_role:
            try:
                cnt_rem[role] = cnt_rem.get(role, 0) + 1
                every_n_rem = int(max(1, cfg_rem.get("send_every_n", 1)))
                allow_by_n_rem = (cnt_rem[role] % every_n_rem) == 0
                now_rem = time.perf_counter()
                rlim = rem_limiters.get(role)
                if rlim is None:
                    rlim = rem_limiters.setdefault(role, FpsLimiter(rem_fps))
                else:
                    rlim.set_fps(rem_fps)
                if allow_by_n_rem and rlim.allow(now_rem):
                    allow_rem = True
            except Exception:
                allow_rem = False





        if not (allow_prev or zoom_active or allow_rem):
            continue

        # 3) DEMOSAIK (bayer -> BGR) - JEDEN RAZ dla wszystkich
        col_bgr = bayer_to_bgr(bayer)
        frames_demosaic += 1
        col_rgb = None  # leniwe BGR->RGB tylko jeśli potrzeba

        # Cache dla zoom/remote (unikamy wielokrotnego demosaiku)
        cached_bgr = col_bgr

        try:
            if allow_prev:
                # UĹĽyj klatki z adnotacjami YOLO jeśli dostÄ™pna, inaczej oryginalna
                base_prev = col_bgr
                
                # # Convert to RGB if needed
                # needs_rgb = str(cfg_prev.get("color_format", "BGR")).upper() == "RGB"
                # if needs_rgb:
                #     base_prev = base_prev[..., ::-1].copy()
                #     col_rgb = base_prev

                scale_p = float(cfg_prev.get("scale", 1.0))
                if scale_p != 1.0:
                    new_w = max(1, int(W * scale_p))
                    new_h = max(1, int(H * scale_p))
                    prev_frame = resize_rgb_nn(base_prev, new_w, new_h)
                else:
                    prev_frame = base_prev

                out_key_prev = f"{role}_preview_{ts_ns}"
                ok_prev = smm.write_frame(out_key_prev, prev_frame, ts_ns=ts_ns)

                q_prev = output_queues.get("preview")
                if ok_prev and q_prev is not None:
                    try:
                        if isinstance(q_prev, dict):
                            q_role = q_prev.get(role)
                            if q_role is not None:
                                q_role.put_nowait((role, out_key_prev, ts_ns))
                        else:
                            q_prev.put_nowait((role, out_key_prev, ts_ns))
                        frames_prev += 1
                    except queue.Full:
                        try:
                            if isinstance(q_prev, dict):
                                q_role = q_prev.get(role)
                                if q_role is not None:
                                    _ = q_role.get_nowait()
                                    q_role.put_nowait((role, out_key_prev, ts_ns))
                            else:
                                _ = q_prev.get_nowait()
                                q_prev.put_nowait((role, out_key_prev, ts_ns))
                            frames_prev += 1
                        except Exception:
                            pass
        except Exception as e:
            print(f"[UNIVERSAL-PROC] preview error for {role}: {e}")

        # 5) ZOOM â€“ uĹĽywa cached_bgr (bez ponownego demosaiku)
        try:
            if zoom_active and output_queues.get("zoom"):
                qz = output_queues["zoom"]
                try:
                    while not qz.empty():
                        qz.get_nowait()
                except Exception:
                    pass
                try:
                    qz.put_nowait((role, cached_bgr, ts_ns))
                    frames_zoom += 1
                except Exception:
                    pass
        except Exception:
            pass

        # 6) REMOTE (WebRTC)
        try:
            if allow_rem:
                # respektuj color_format z cfg_rem
                rem_color_format = str(cfg_rem.get("color_format", "BGR")).upper()

                base_rem = col_bgr  # domyĹ›lnie BGR
                if rem_color_format == "RGB":
                    # leniwe BGR->RGB tylko jeśli faktycznie potrzebne
                    if col_rgb is None:
                        col_rgb = col_bgr[..., ::-1].copy()
                    base_rem = col_rgb

                full_key = f"{role}_full_latest"
                smm.write_frame(full_key, base_rem, ts_ns=ts_ns)

                now_t = time.time()
                should_update = (now_t - last_shared_ts) >= shared_min_dt and last_full_key.get(role) != full_key
                if shared_state is not None and should_update:
                    try:
                        ck_full = shared_state.get("full_key")
                        if ck_full is not None:
                            ck_full[role] = full_key
                            last_full_key[role] = full_key
                            last_shared_ts = now_t
                    except Exception:
                        pass

                scale_r = float(cfg_rem.get("scale", 1.0))
                if scale_r != 1.0:
                    new_wr = max(1, int(W * scale_r))
                    new_hr = max(1, int(H * scale_r))
                    rem_frame = resize_rgb_nn(base_rem, new_wr, new_hr)
                else:
                    rem_frame = base_rem

                out_key_rem = f"{role}_remote_latest"
                ok_rem = smm.write_frame(out_key_rem, rem_frame, ts_ns=ts_ns)
                should_update_current = (now_t - last_shared_ts) >= shared_min_dt and last_current_key.get(role) != out_key_rem
                if ok_rem and shared_state is not None and should_update_current:
                    try:
                        ck = shared_state.get("current_key")
                        if ck is not None:
                            ck[role] = out_key_rem
                            last_current_key[role] = out_key_rem
                            last_shared_ts = now_t
                    except Exception as e:
                        print(f"[UNIVERSAL-PROC] shared_state write error: {e}")

                frames_rem += 1

                rlim = rem_limiters.get(role)
                if rlim is not None and now_rem is not None:
                    rlim.sleep_until_next(now_rem)
        except Exception as e:
            print(f"[UNIVERSAL-PROC] remote error for {role}: {e}")


        # 7) statystyki co ~1s
        now_stats = time.time()
        if stats_q is not None and (now_stats - stats_last_t) >= 3.0:
            msg = {
                "type": "color_stats",
                "frames_in": frames_in,
                "frames_demosaic": frames_demosaic,
                "frames_prev": frames_prev,
                "frames_rem": frames_rem,
                "frames_zoom": frames_zoom,
                "raw_dropped": raw_dropped,


            }
            try:
                stats_q.put_nowait(msg)
            except Exception:
                pass
            stats_last_t = now_stats

    print("[UNIVERSAL-PROC] ... STOP")


def start_universal_color_processor(universal_bayer_q,
                                    output_queues,
                                    processing_config,
                                    stop_evt,
                                    stats_q=None,
                                    remote_evt=None,
                                    shared_state=None):
    p = mp.Process(
        target=universal_color_processor,
        args=(
            universal_bayer_q,
            output_queues,
            processing_config,
            stop_evt,
            stats_q,
            remote_evt,
            shared_state,
        ),
        daemon=True,
        name="universal_color_proc",
    )
    p.start()
    return {"universal": p}





