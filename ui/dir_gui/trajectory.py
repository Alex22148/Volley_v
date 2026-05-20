import  time,queue
import os
import webbrowser
from collections import deque
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from storage.shared_memory_manager import get_shared_memory_manager
from ui.style_volleyhub import (
    BG_LIGHT)




class TrajectoryWindow(ctk.CTkToplevel):
    def __init__(self, parent, roles, yolo_vis_q, history_len=10, stats_q=None):
        self.stats_q = stats_q
        super().__init__(parent)

        self.title("YOLO trajectories")
        self.geometry("900x700")
        self.configure(fg_color=BG_LIGHT)

        self.roles = list(roles)
        self.yolo_vis_q = yolo_vis_q
        self.history_len = history_len
        self._closing = False
        self._after_id = None

        self.history = {
            role: deque(maxlen=history_len) for role in self.roles
        }

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.axes = self.fig.subplots(2, 2)
        self.axes = self.axes.flatten()


        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_update()

    def _schedule_update(self, delay_ms=1000):
        if self._closing or not self.winfo_exists():
            return
        self._after_id = self.after(delay_ms, self._update_plot)

    def _on_close(self):
        self._closing = True
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        try:
            self.destroy()
        except Exception:
            pass

    def _consume_queue(self):
        if self.yolo_vis_q is None:
            return

        while True:
            try:
                item = self.yolo_vis_q.get_nowait()
            except Exception:
                break

            if not isinstance(item, dict):
                continue

            batch_id = item.get("batch_id")
            per_role_points = item.get("per_role_points", {})

            for role in self.roles:
                pts = per_role_points.get(role, [])
                self.history[role].append({
                    "batch_id": batch_id,
                    "points": pts,
                })

    def _update_plot(self):
        if self._closing or not self.winfo_exists():
            return

        self._consume_queue()

        cmap = [
            "#CBD5E1", "#94A3B8", "#64748B", "#475569", "#334155",
            "#1E293B", "#0F172A", "#2563EB", "#16A34A", "#DC2626"
        ]

        default_w = 4096
        default_h = 3000
        print(f'GUI UPDATE PLOT przed {time.time()}')

        for ax, role in zip(self.axes, self.roles):
            ax.clear()
            ax.set_title(role, fontsize=10)
            ax.grid(True, alpha=0.2)

            batches = list(self.history[role])
            if not batches:
                ax.set_xlim(0, default_w)
                ax.set_ylim(default_h, 0)   # odwrócona Y
                continue

            all_x = []
            all_y = []

            for i, batch in enumerate(batches):
                color = cmap[min(i, len(cmap) - 1)]
                pts = batch.get("points", [])

                if not pts:
                    continue

                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]

                all_x.extend(xs)
                all_y.extend(ys)

                # starsze batch'e cieńsze, nowsze wyraźniejsze
                lw = 0.8 if i < len(batches) - 1 else 1.8

                ax.plot(xs, ys, linewidth=lw, color=color)

                for j, p in enumerate(pts):
                    size = 12 + 8 * j
                    ax.scatter(p["x"], p["y"], s=size, color=color)

            if all_x and all_y:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)

                pad_x = max(10, int((xmax - xmin) * 0.2) if xmax > xmin else 20)
                pad_y = max(10, int((ymax - ymin) * 0.2) if ymax > ymin else 20)

                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_ylim(ymax + pad_y, ymin - pad_y)  # odwrócona Y
            else:
                ax.set_xlim(0, default_w)
                ax.set_ylim(default_h, 0)

            # podpis najnowszego batcha
            try:
                last_batch_id = batches[-1].get("batch_id")
                ax.text(
                    0.02, 0.98, f"batch {last_batch_id}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top"
                )
            except Exception:
                pass

        self.fig.tight_layout()
        self.canvas.draw_idle()
        print(f'GUI UPDATE PLOT po {time.time()}')
        self._schedule_update(1000)


def open_trajectory_window(self):
    if getattr(self, "trajectory_window", None) is not None:
        try:
            if self.trajectory_window.winfo_exists():
                self.trajectory_window.lift()
                self.trajectory_window.focus_force()
                return
        except Exception:
            pass

    self.trajectory_window = TrajectoryWindow(
        parent=self,
        roles=self.roles,
        yolo_vis_q=getattr(self, "yolo_vis_q", None),
        stats_q=self.stats_q,
    )
