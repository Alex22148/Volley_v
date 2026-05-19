"""
Auto-generated module extracted from ui/gui.py.
Do not edit manually unless you know what you are doing.
"""

import time

import queue


import customtkinter as ctk
import numpy as np

from PIL import Image


from ui.style_volleyhub import (
    init_ctk_theme,
    BG_LIGHT,
    PANEL_BG,
    CARD_BG,
    CARD_INNER_BG,
    BORDER_COLOR,
    TEXT_MAIN,
    TEXT_DIM,
    ACCENT,
    ACCENT_HOVER,
    SUCCESS,
    DANGER,
    LIVE_BG,
    FONT_FAMILY,
)


def set_ctk_image(widget, image_like, size=None):
    """Ustawia obraz na CTkLabel/CTkButton gwarantując CTkImage + referencję."""
    if isinstance(image_like, ctk.CTkImage):
        cimg = image_like
    elif isinstance(image_like, Image.Image):
        cimg = ctk.CTkImage(light_image=image_like, size=size or image_like.size)
    elif isinstance(image_like, np.ndarray):
        pil = Image.fromarray(image_like)
        cimg = ctk.CTkImage(light_image=pil, size=size or pil.size)
    else:
        return
    widget.configure(image=cimg, text="")
    widget.image = cimg  # trzymamy referencję

def toggle_preview(self):
    self.preview_on = not self.preview_on

    self._sync_operator_state_ui()

    if self.preview_on:
        self._set_status("🎥 Podgląd aktywny", "success")
        # Uruchom pętlę aktualizacji klatek
        if not hasattr(self, "_update_frames_running") or not self._update_frames_running:
            self._update_frames_running = True
            self._after(50, self.update_frames)
    else:
        self._set_status("🛑 Podgląd wyłączony", "danger")
        self._update_frames_running = False

    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("stream", self.preview_on))
    except Exception as e:
        print(f"[GUI] ⚠️ Error toggling preview: {e}")

def update_frames(self):
    if self._closing or not self.winfo_exists():
        return
    
    # Zatrzymaj pętlę jeśli podgląd jest wyłączony
    if not self.preview_on:
        self._update_frames_running = False
        return
    
    if self.live_q is None:
        if not hasattr(self, "_live_q_missing_warned"):
            self._live_q_missing_warned = False
        if not self._live_q_missing_warned:
            self._live_q_missing_warned = True
            print("[GUI] live_q is None - preview disabled")
        self._after(500, self.update_frames)
        return

    frames_for_role = {}
    try:
        while True:
            item = self.live_q.get_nowait()
            if not (isinstance(item, (tuple, list)) and len(item) >= 2):
                continue
            role = item[0]
            payload = item[1]
            ts_ns = item[2] if len(item) >= 3 else 0
            frames_for_role[role] = (payload, ts_ns)
    except queue.Empty:
        pass
    except Exception as e:
        print(f"[GUI] live_q error: {e}")

    for role, (payload, ts_ns) in frames_for_role.items():
        frame = None
        try:
            if isinstance(payload, str):
                got = self.smm.read_frame(payload)
                if got is None:
                    continue
                if isinstance(got, tuple):
                    frame, _meta = got
                else:
                    frame = got
            elif isinstance(payload, np.ndarray):
                frame = payload
            else:
                continue
        except Exception as e:
            print(f"[GUI] decode error for {role}: {e}")
            continue

        if frame is not None:
            self.display_frame(role, frame)

    now = time.time()
    timeout = 0.8
    for role, lbl in self.camera_labels.items():
        last = self._last_frame_time.get(role, 0.0)
        if last == 0.0:
            continue
        if (now - last) > timeout:
            try:
                lbl.configure(
                    text=f"{role}\nOczekiwanie na obraz...",
                    image=None,
                    fg_color=LIVE_BG,
                )
                lbl.image = None
            except Exception:
                pass
            self.fullres_frames[role] = None
            self._last_frame_time[role] = 0.0

    target_fps = 12.0
    interval_ms = int(1000 / target_fps)
    self._after(interval_ms, self.update_frames)

def display_frame(self, role, frame):
    try:
        if frame is None or not isinstance(frame, np.ndarray):
            return
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return

        label = self.camera_labels.get(role) or self.img_labels.get(role)
        if not label:
            print(f"[GUI] ❌ No label for role {role}")
            return

        render_frame = self._apply_yolo_point_overlay(role, frame)

        color_order = str(getattr(self, "preview_input_color_order", "bgr")).strip().lower()
        if color_order == "rgb":
            render_frame_rgb = np.ascontiguousarray(render_frame)
        else:
            render_frame_rgb = np.ascontiguousarray(render_frame[:, :, ::-1])

        self._render_to_label(label, render_frame_rgb)
        self._last_frame_time[role] = time.time()
        self.fullres_frames[role] = frame

    except Exception as e:
        print(f"[GUI] ❌ Error displaying frame for {role}: {e}")

def toggle_preview_color_order(self):
    cur = str(getattr(self, "preview_input_color_order", "bgr")).lower()
    self.preview_input_color_order = "rgb" if cur == "bgr" else "bgr"
    print(f"[GUI] preview_input_color_order = {self.preview_input_color_order}")
    try:
        self._set_status(f"Preview color order: {self.preview_input_color_order.upper()}", "accent")
    except Exception:
        pass

def on_preview_double_click(self, role):
    if self.zoom_q is None:
        self._set_status("Brak kolejki zoom_q.", "danger")
        return
    try:
        self.control_q.put(("set_zoom_role", role))
    except Exception:
        pass

    import cv2
    import queue as _queue
    if self.zoom_q is None:
        self.zoom_q = _queue.Queue(maxsize=8)

    screen_w = self.winfo_screenwidth()
    screen_h = self.winfo_screenheight()
    max_w = int(screen_w * 0.8); max_h = int(screen_h * 0.8)

    if hasattr(self, "_zoom_windows") and role in getattr(self, "_zoom_windows", {}):
        win = self._zoom_windows[role]
        if win.winfo_exists():
            win.lift(); win.focus_force(); return
    else:
        self._zoom_windows = {}

    zoom_win = ctk.CTkToplevel(self)
    zoom_win.title(f"{role} — Zoom Preview")
    zoom_win.geometry(f"{max_w}x{max_h}")
    zoom_win.lift(); zoom_win.attributes("-topmost", True); zoom_win.focus_force()

    lbl = ctk.CTkLabel(zoom_win, text="⏳ czekam na klatkę...")
    lbl.pack(fill="both", expand=True)
    zoom_win.update_idletasks()
    self._zoom_windows[role] = zoom_win

    zoom_factor = 1.0
    zoom_center = [0.5, 0.5]

    def apply_zoom(new_zoom, cx, cy):
        nonlocal zoom_factor, zoom_center
        new_zoom = max(1.0, min(float(new_zoom), 12.0))
        if new_zoom == zoom_factor:
            return
        scale = new_zoom / zoom_factor
        zoom_center[0] = cx + (zoom_center[0] - cx) / scale
        zoom_center[1] = cy + (zoom_center[1] - cy) / scale
        zoom_center[0] = min(max(zoom_center[0], 0.0), 1.0)
        zoom_center[1] = min(max(zoom_center[1], 0.0), 1.0)
        zoom_factor = new_zoom

    def on_mouse_click(event):
        if lbl.winfo_width() <= 0 or lbl.winfo_height() <= 0:
            return
        zoom_center[0] = event.x / max(1, lbl.winfo_width())
        zoom_center[1] = event.y / max(1, lbl.winfo_height())

    def on_wheel_win(event):
        if lbl.winfo_width() <= 0 or lbl.winfo_height() <= 0:
            return
        cx = event.x / max(1, lbl.winfo_width())
        cy = event.y / max(1, lbl.winfo_height())
        step = 0.12 if getattr(event, "delta", 0) > 0 else -0.12
        apply_zoom(zoom_factor + step, cx, cy)

    def on_wheel_up(event):
        cx = event.x / max(1, lbl.winfo_width())
        cy = event.y / max(1, lbl.winfo_height())
        apply_zoom(zoom_factor + 0.12, cx, cy)

    def on_wheel_down(event):
        cx = event.x / max(1, lbl.winfo_width())
        cy = event.y / max(1, lbl.winfo_height())
        apply_zoom(zoom_factor - 0.12, cx, cy)

    lbl.bind("<Button-1>", on_mouse_click)
    lbl.bind("<MouseWheel>", on_wheel_win)  # Win/macOS
    lbl.bind("<Button-4>", on_wheel_up)    # Linux
    lbl.bind("<Button-5>", on_wheel_down)  # Linux

    def reset_zoom(_e=None):
        nonlocal zoom_factor, zoom_center
        zoom_factor = 1.0; zoom_center[:] = [0.5, 0.5]
    lbl.bind("<Double-Button-1>", reset_zoom)

    def render_zoom(frame_rgb):
        try:
            import cv2
            h, w = frame_rgb.shape[:2]
            cx = int(zoom_center[0] * w); cy = int(zoom_center[1] * h)
            half_w = max(1, int(w / (2.0 * zoom_factor)))
            half_h = max(1, int(h / (2.0 * zoom_factor)))
            x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
            x2, y2 = min(w, cx + half_w), min(h, cy + half_h)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame_rgb[y1:y2, x1:x2]
            tw, th = max(2, lbl.winfo_width()), max(2, lbl.winfo_height())
            if crop.shape[1] != tw or crop.shape[0] != th:
                crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
            im = Image.fromarray(crop)
            return ctk.CTkImage(light_image=im, size=(tw, th))
        except Exception as e:
            print(f"[ZOOM] render_zoom error: {e}")
            return None

    zoom_after = {"id": None}

    def update_loop():
        if not zoom_win.winfo_exists() or not lbl.winfo_exists():
            return
        frame_rgb = None
        try:
            q = self.zoom_q; item = None
            while q is not None and not q.empty():
                item = q.get_nowait()
            if item is not None:
                try:
                    role2, payload, _ts = item
                except Exception:
                    role2, _ts, payload = item
                if role2 == role and payload is not None:
                    if isinstance(payload, str):
                        got = self.smm.read_frame(payload)
                        if got is not None:
                            frame_rgb, _ = got if isinstance(got, tuple) else (got, 0)
                    elif isinstance(payload, np.ndarray):
                        frame_rgb = payload
        except Exception as e:
            print(f"[ZOOM] queue read error: {e}")
        if frame_rgb is None:
            frame_rgb = self.fullres_frames.get(role)
        if frame_rgb is not None:
            img_ctk = render_zoom(frame_rgb)

            if img_ctk:
                set_ctk_image(lbl, img_ctk)

        else:
            lbl.configure(text="⏳ brak klatek do zoomu (kolejka pusta)")
        zoom_after["id"] = zoom_win.after(16, update_loop)

    def on_zoom_close():
        aid = zoom_after.get("id")
        if aid:
            try:
                zoom_win.after_cancel(aid)
            except Exception:
                pass
            zoom_after["id"] = None
        try:
            if hasattr(self, "_zoom_windows"):
                self._zoom_windows.pop(role, None)
        except Exception:
            pass
        try:
            if zoom_win.winfo_exists():
                zoom_win.destroy()
        except Exception:
            pass

    zoom_win.protocol("WM_DELETE_WINDOW", on_zoom_close)
    update_loop()

def _render_to_label(self, label, frame_rgb, keep_aspect: bool = True):
    if frame_rgb is None:
        return
    now = time.time()
    if not hasattr(label, "_last_query_ts"):
        label._last_query_ts = 0.0
    if (now - label._last_query_ts) > 0.3 or not hasattr(label, "_target_size"):
        tw = max(1, label.winfo_width())
        th = max(1, label.winfo_height())
        label._target_size = (tw, th)
        label._last_query_ts = now

    target_w, target_h = label._target_size
    fh, fw = frame_rgb.shape[:2]
    if keep_aspect:
        scale = min(target_w / max(1, fw), target_h / max(1, fh))
        new_w = max(1, int(fw * scale)); new_h = max(1, int(fh * scale))
    else:
        new_w, new_h = target_w, target_h

    if fw == new_w and fh == new_h:
        pil_img = Image.fromarray(frame_rgb)
    else:
        pil_img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.BILINEAR)


    img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
    set_ctk_image(label, img)

def _apply_yolo_point_overlay(self, role: str, frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb is None or not isinstance(frame_rgb, np.ndarray):
        return frame_rgb

    snap = self._get_shared_state_value("yolo_last_inference_snapshot", None)
    if not hasattr(snap, "get"):
        return frame_rgb
    det_flat = snap.get("detections_flat", [])
    if not isinstance(det_flat, list) or not det_flat:
        return frame_rgb

    role_norm = str(role).strip().upper()
    latest_step = None
    for d in det_flat:
        if str(d.get("role", "")).strip().upper() != role_norm:
            continue
        sidx = int(d.get("step_idx", -1) or -1)
        if latest_step is None or sidx > latest_step:
            latest_step = sidx
    if latest_step is None:
        return frame_rgb

    try:
        import cv2
        out = frame_rgb.copy()
        h, w = out.shape[:2]
        class_id = int(self._get_shared_state_value("yolo_ball_class_id", 0) or 0)

        for d in det_flat:
            if str(d.get("role", "")).strip().upper() != role_norm:
                continue
            if int(d.get("step_idx", -1) or -1) != int(latest_step):
                continue

            fw = max(1, int(d.get("frame_w", w) or w))
            fh = max(1, int(d.get("frame_h", h) or h))
            bx = float(d.get("x", 0.0) or 0.0)
            by = float(d.get("y", 0.0) or 0.0)
            bw = float(d.get("width", 0.0) or 0.0)
            bh = float(d.get("height", 0.0) or 0.0)
            cx = float(d.get("center_x", bx + bw * 0.5) or (bx + bw * 0.5))
            cy = float(d.get("center_y", by + bh * 0.5) or (by + bh * 0.5))
            conf = float(d.get("confidence", 0.0) or 0.0)

            x1 = int(round(bx * w / fw))
            y1 = int(round(by * h / fh))
            x2 = int(round((bx + bw) * w / fw))
            y2 = int(round((by + bh) * h / fh))
            px = int(round(cx * w / fw))
            py = int(round(cy * h / fh))

            if x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)

            if 0 <= px < w and 0 <= py < h:
                cv2.circle(out, (px, py), 8, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (px, py), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            label_txt = f"{class_id} {conf:.2f}"
            label_size = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            lx1 = int(x1 if x2 > x1 else max(0, px - 10))
            ly1 = int(max(0, (y1 if y2 > y1 else py) - label_size[1] - 10))
            lx2 = int(lx1 + label_size[0])
            ly2 = int(y1 if y2 > y1 else py)
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                out,
                label_txt,
                (lx1, max(0, ly2 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return out
    except Exception:
        return frame_rgb
