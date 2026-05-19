"""
Auto-generated module extracted from ui/gui.py.
Do not edit manually unless you know what you are doing.
"""



import customtkinter as ctk


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

def _ensure_auto_benchmark_vars(self):
    if hasattr(self, "ab_model_source_var"):
        return
    self.ab_model_source_var = ctk.StringVar(value="Wiele modeli (folder)")
    self.ab_models_dir_var = ctk.StringVar(value="")
    self.ab_imgsz_var = ctk.StringVar(value="640,960")
    self.ab_batch_var = ctk.StringVar(value="4,8,12")
    self.ab_batch_mode_var = ctk.StringVar(value="Zakres")
    self.ab_single_model_var = ctk.StringVar(value="")
    self.ab_batch_from_var = ctk.StringVar(value="4")
    self.ab_batch_to_var = ctk.StringVar(value="16")
    self.ab_batch_step_var = ctk.StringVar(value="4")
    self.ab_modes_var = ctk.StringVar(value="Wszystkie kombinacje")
    self.ab_backend_mode_var = ctk.StringVar(value="Auto (wg modelu)")
    self.ab_measure_s_var = ctk.StringVar(value="10")
    self.ab_warmup_s_var = ctk.StringVar(value="4")
    self.ab_bin_s_var = ctk.StringVar(value="10")
    self.ab_buffer_count_var = ctk.StringVar(value="2")
    self.ab_buffer_pause_s_var = ctk.StringVar(value="3")
    self.ab_out_var = ctk.StringVar(value="")
    self.ab_camera_fps_var = ctk.StringVar(value="50")
    self.ab_camera_period_var = ctk.StringVar(value="20000.0 us")
    self.ab_status_var = ctk.StringVar(value="Auto benchmark: gotowy")
    self.ab_plan_var = ctk.StringVar(value="Plan: -")

def _build_main_ui_operator(self):
    pad = 10
    inner = 8
    self._ensure_auto_benchmark_vars()

    # =========================================================
    # ROOT LAYOUT
    # =========================================================
    self.main_frame.grid_rowconfigure(0, weight=0)  # header
    self.main_frame.grid_rowconfigure(1, weight=1)  # content
    self.main_frame.grid_rowconfigure(2, weight=0)  # footer
    self.main_frame.grid_columnconfigure(0, weight=1)

    # =========================================================
    # HEADER — status + quick actions
    # =========================================================
    header = ctk.CTkFrame(self.main_frame, fg_color=CARD_BG, corner_radius=12)
    header.grid(row=0, column=0, sticky="ew", padx=pad, pady=(pad, 6))
    header.grid_columnconfigure(0, weight=1)
    header.grid_columnconfigure(1, weight=0)

    self.header_status_var = ctk.StringVar(value="Kamery: --/-- | CPU: -- | GPU: -- | Remote: OFF")
    self.header_session_var = ctk.StringVar(value="Folder sesji: --")
    self.header_disk_var = ctk.StringVar(value="Wolne miejsce: --")
    self.header_hint_var = ctk.StringVar(value="Gotowy do startu sesji")

    header_left = ctk.CTkFrame(header, fg_color="transparent")
    header_left.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
    from ui.dir_gui.trajectory import TrajectoryWindow
    self.trajectory_window = TrajectoryWindow(
        parent=self,
        roles=self.roles,
        yolo_vis_q=getattr(self, "yolo_vis_q", None),
        stats_q=self.stats_q,
    )
    ctk.CTkLabel(
        header_left,
        text="VolleyHub — Operator",
        text_color=TEXT_MAIN,
        font=(FONT_FAMILY, 17, "bold"),
    ).pack(anchor="w")

    ctk.CTkLabel(
        header_left,
        textvariable=self.header_status_var,
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 12),
    ).pack(anchor="w")

    ctk.CTkLabel(
        header_left,
        textvariable=self.header_session_var,
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 12),
    ).pack(anchor="w")

    ctk.CTkLabel(
        header_left,
        textvariable=self.header_disk_var,
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 12),
    ).pack(anchor="w")

    header_right = ctk.CTkFrame(header, fg_color="transparent")
    header_right.grid(row=0, column=1, sticky="e", padx=10, pady=8)

    self.quick_start_btn = ctk.CTkButton(
        header_right,
        text="🚀 Start Session",
        fg_color="#0ea5e9",
        hover_color="#0284c7",
        command=self._quick_start_session,
        height=34,
        corner_radius=10,
        width=150,
    )
    self.quick_start_btn.pack(fill="x", pady=(0, 6))

    self.preview_btn = ctk.CTkButton(
        header_right,
        text="▶ Włącz podgląd",
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        command=self.toggle_preview,
        height=32,
        corner_radius=10,
        width=150,
    )
    self.preview_btn.pack(fill="x", pady=(0, 6))

    self.yolo_btn = ctk.CTkButton(
        header_right,
        text="🎯 Detekcja OFF",
        fg_color="#FF6B35",
        hover_color="#E55A2B",
        command=self.toggle_yolo,
        height=32,
        corner_radius=10,
        width=150,
    )
    self.yolo_btn.pack(fill="x", pady=(0, 4))

    ctk.CTkLabel(
        header_right,
        textvariable=self.header_hint_var,
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 11),
    ).pack(anchor="e", pady=(4, 0))

    # =========================================================
    # CONTENT — left preview / right contextual tabs
    # =========================================================
    body = ctk.CTkFrame(self.main_frame, fg_color=BG_LIGHT)
    body.grid(row=1, column=0, sticky="nsew", padx=pad, pady=(0, 6))
    body.grid_rowconfigure(0, weight=1)
    body.grid_columnconfigure(0, weight=68)  # PREVIEW
    body.grid_columnconfigure(1, weight=32)  # RIGHT PANEL

    # ---------------------------------------------------------
    # PREVIEW AREA
    # ---------------------------------------------------------
    preview_area = ctk.CTkFrame(
        body,
        fg_color=PANEL_BG,
        corner_radius=12,
        border_width=1,
        border_color=BORDER_COLOR,
    )
    preview_area.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    preview_area.grid_rowconfigure(1, weight=1)
    preview_area.grid_columnconfigure(0, weight=1)

    preview_top = ctk.CTkFrame(preview_area, fg_color=CARD_BG, corner_radius=10)
    preview_top.grid(row=0, column=0, sticky="ew", padx=inner, pady=(inner, 6))
    preview_top.grid_columnconfigure(0, weight=1)
    preview_top.grid_columnconfigure(1, weight=0)

    self.preview_status = ctk.CTkLabel(
        preview_top,
        text="Podgląd wyłączony",
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 13, "bold"),
        anchor="w",
    )
    self.preview_status.grid(row=0, column=0, sticky="w", padx=10, pady=8)

    self.preview_hint_var = ctk.StringVar(value="Double click na kamerze = powiększenie")
    ctk.CTkLabel(
        preview_top,
        textvariable=self.preview_hint_var,
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 11),
    ).grid(row=0, column=1, sticky="e", padx=10, pady=8)

    preview_wall = ctk.CTkFrame(preview_area, fg_color=CARD_BG, corner_radius=10)
    preview_wall.grid(row=1, column=0, sticky="nsew", padx=inner, pady=(0, inner))
    preview_wall.grid_columnconfigure(0, weight=1)
    preview_wall.grid_columnconfigure(1, weight=1)
    preview_wall.grid_rowconfigure(0, weight=1)
    preview_wall.grid_rowconfigure(1, weight=1)

    self.camera_labels = {}
    self.img_labels = {}
    self.preview_frames = {}
    self.sliders = {}
    self.eff_bars = {}
    self.eff_labels = {}

    for i, role in enumerate(self.roles):
        row, col = divmod(i, 2)

        cam_card = ctk.CTkFrame(
            preview_wall,
            fg_color=PANEL_BG,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        cam_card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
        cam_card.grid_rowconfigure(1, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)

        # top bar kamery
        cam_top = ctk.CTkFrame(cam_card, fg_color="transparent")
        cam_top.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        cam_top.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            cam_top,
            text=role.upper(),
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")

        eff_wrap = ctk.CTkFrame(cam_top, fg_color="transparent")
        eff_wrap.grid(row=0, column=1, sticky="e")
        eff_wrap.grid_columnconfigure(1, weight=1)

        eff_bar = ctk.CTkProgressBar(
            eff_wrap,
            width=80,
            fg_color="#334155",
            progress_color=ACCENT,
        )
        eff_bar.set(0.0)
        eff_bar.grid(row=0, column=0, sticky="e", padx=(0, 6))

        eff_lbl = ctk.CTkLabel(
            eff_wrap,
            text="0% (0/0)",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 10),
        )
        eff_lbl.grid(row=0, column=1, sticky="e")

        self.eff_bars[role] = eff_bar
        self.eff_labels[role] = eff_lbl

        # box preview — maksymalnie duży
        preview_box = ctk.CTkFrame(
            cam_card,
            fg_color=CARD_INNER_BG,
            corner_radius=8,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        preview_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.preview_frames[role] = preview_box

        lbl = ctk.CTkLabel(
            preview_box,
            text=f"{role}\nOczekiwanie na obraz...",
            fg_color="transparent",
            text_color=TEXT_DIM,
            font=(FONT_FAMILY, 12),
        )
        lbl.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.97, relheight=0.97)
        lbl.bind(
            "<Configure>",
            lambda e, r=role: (
                    self.fullres_frames.get(r) is not None and self.display_frame(r, self.fullres_frames[r])
            ),
        )
        lbl.bind("<Double-Button-1>", lambda e, r=role: self.on_preview_double_click(r))

        self.camera_labels[role] = lbl
        self.img_labels[role] = lbl

    # ---------------------------------------------------------
    # RIGHT CONTEXT PANEL
    # ---------------------------------------------------------
    right = ctk.CTkFrame(
        body,
        fg_color=PANEL_BG,
        corner_radius=12,
        border_width=1,
        border_color=BORDER_COLOR,
    )
    right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    right.grid_rowconfigure(0, weight=1)
    right.grid_columnconfigure(0, weight=1)

    self.operator_tabs = ctk.CTkTabview(right)
    self.operator_tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    tab_session = self.operator_tabs.add("Session")
    tab_ai = self.operator_tabs.add("AI / YOLO")
    tab_cameras = self.operator_tabs.add("Cameras")
    tab_diag = self.operator_tabs.add("Diagnostics")

    # =========================================================
    # TAB: SESSION
    # =========================================================
    session_card = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
    session_card.pack(fill="x", padx=6, pady=(6, 6))

    ctk.CTkLabel(
        session_card,
        text="Session i Recording",
        text_color=TEXT_MAIN,
        font=(FONT_FAMILY, 14, "bold"),
    ).pack(anchor="w", padx=10, pady=(8, 2))

    self.path_label = ctk.CTkLabel(
        session_card,
        text=self.raw_dir if self.raw_dir else "– nie wybrano –",
        text_color=TEXT_DIM,
        font=("Consolas", 10),
        justify="left",
        anchor="w",
        wraplength=340,
    )
    self.path_label.pack(fill="x", padx=10, pady=(2, 4))

    self.btn_choose_folder = ctk.CTkButton(
        session_card,
        text="📁 Zmień folder",
        command=self.on_choose_raw,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        height=30,
    )
    self.btn_choose_folder.pack(fill="x", padx=10, pady=(0, 8))

    self.rec_led = ctk.CTkLabel(
        session_card,
        text="●",
        text_color="#B0B0B0",
        font=(FONT_FAMILY, 18, "bold"),
    )
    self.rec_led.pack(anchor="w", padx=10)

    self.rec_text = ctk.CTkLabel(
        session_card,
        text="Nagrywanie wyłączone",
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 12, "bold"),
        anchor="w",
    )
    self.rec_text.pack(fill="x", padx=10, pady=(0, 6))

    self.session_elapsed_var = ctk.StringVar(value="Czas sesji: 00:00:00")
    self.buffer_fill_var = ctk.StringVar(value="Bufor: -- / -- s")
    self.left_disk_var = ctk.StringVar(value="Wolne miejsce: --")

    ctk.CTkLabel(session_card, textvariable=self.session_elapsed_var, text_color=TEXT_DIM).pack(anchor="w", padx=10)
    ctk.CTkLabel(session_card, textvariable=self.buffer_fill_var, text_color=TEXT_DIM).pack(anchor="w", padx=10)
    ctk.CTkLabel(session_card, textvariable=self.left_disk_var, text_color=TEXT_DIM).pack(anchor="w", padx=10,
                                                                                          pady=(0, 8))

    rec_controls = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
    rec_controls.pack(fill="x", padx=6, pady=(0, 6))
    rec_controls.grid_columnconfigure((0, 1), weight=1)

    self.btn_start = self._make_btn(rec_controls, "● Start recording", self.on_start_record, "#f97316", height=32,
                                    font_size=12)
    self.btn_pause = self._make_btn(rec_controls, "⏸ Pause", self.on_pause_record, "#f59e0b", height=32,
                                    font_size=12)
    self.btn_stop = self._make_btn(rec_controls, "■ Stop", self.on_stop_record, DANGER, height=32, font_size=12)
    self.btn_save_buffer = self._make_btn(rec_controls, "💾 Save buffer (5s)", self.on_save_buffer, "#7c3aed",
                                          height=32, font_size=12)
    self.btn_shot = self._make_btn(rec_controls, "📸 Snapshot", self.on_take_shot, SUCCESS, height=32, font_size=12)

    self.btn_start.grid(row=0, column=0, padx=(8, 4), pady=(8, 4), sticky="ew")
    self.btn_pause.grid(row=0, column=1, padx=(4, 8), pady=(8, 4), sticky="ew")
    self.btn_stop.grid(row=1, column=0, padx=(8, 4), pady=4, sticky="ew")
    self.btn_save_buffer.grid(row=1, column=1, padx=(4, 8), pady=4, sticky="ew")
    self.btn_shot.grid(row=2, column=0, columnspan=2, padx=8, pady=(4, 8), sticky="ew")

    buffer_card = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
    buffer_card.pack(fill="x", padx=6, pady=(0, 6))

    ctk.CTkLabel(
        buffer_card,
        text="Długość bufora [s]",
        text_color=TEXT_DIM,
    ).pack(anchor="w", padx=10, pady=(8, 0))

    self.buffer_sec_value = ctk.CTkLabel(
        buffer_card,
        text="5",
        text_color=ACCENT,
        font=(FONT_FAMILY, 12, "bold"),
    )
    self.buffer_sec_value.pack(anchor="e", padx=10)

    self.buffer_sec_slider = ctk.CTkSlider(
        buffer_card,
        from_=1,
        to=30,
        number_of_steps=29,
        command=self.on_buffer_seconds_change,
    )
    self.buffer_sec_slider.pack(fill="x", padx=10, pady=(2, 8))
    self.buffer_sec_slider.set(5)

    remote_card = ctk.CTkFrame(tab_session, fg_color=CARD_BG, corner_radius=10)
    remote_card.pack(fill="x", padx=6, pady=(0, 6))

    self.remote_status = ctk.CTkLabel(
        remote_card,
        text="Server wyłączony",
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 12, "bold"),
    )
    self.remote_status.pack(anchor="w", padx=10, pady=(8, 2))

    self.remote_btn = ctk.CTkButton(
        remote_card,
        text="🛰 Remote access",
        command=self.toggle_remote_server,
        height=30,
    )
    self.remote_btn.pack(fill="x", padx=10, pady=(0, 6))

    self.remote_url_label = ctk.CTkLabel(
        remote_card,
        textvariable=self.remote_url_var,
        text_color="#0077CC",
        font=(FONT_FAMILY, 11, "bold"),
    )
    self.remote_url_label.pack(anchor="w", padx=10, pady=(0, 4))
    self.remote_url_label.bind("<Button-1>", lambda e: self._open_remote_url())
    self.remote_url_label.bind("<Enter>", lambda e: self.remote_url_label.configure(cursor="hand2"))
    self.remote_url_label.bind("<Leave>", lambda e: self.remote_url_label.configure(cursor=""))

    self.remote_qr_label = ctk.CTkLabel(remote_card, text="")
    self.remote_qr_label.pack(anchor="w", padx=10, pady=(0, 8))

    # =========================================================
    # TAB: AI / YOLO
    # =========================================================
    ai_card = ctk.CTkFrame(tab_ai, fg_color=CARD_BG, corner_radius=10)
    ai_card.pack(fill="x", padx=6, pady=(6, 6))

    ctk.CTkLabel(
        ai_card,
        text="Detekcja AI",
        text_color=TEXT_MAIN,
        font=(FONT_FAMILY, 14, "bold"),
    ).pack(anchor="w", padx=10, pady=(8, 4))

    self.ai_tabs = ctk.CTkTabview(ai_card)
    self.ai_tabs.pack(fill="x", padx=8, pady=(0, 8))

    tab_std = self.ai_tabs.add("Standard")
    tab_pro = self.ai_tabs.add("Pro")

    self.yolo_status = ctk.CTkLabel(
        tab_std,
        text="Detekcja wyłączona",
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 12, "bold"),
        anchor="w",
    )
    self.yolo_status.pack(fill="x", padx=8, pady=(8, 2))

    self.yolo_perf_label = ctk.CTkLabel(
        tab_std,
        text="Szybkość: -- ms | -- img/s",
        text_color=TEXT_DIM,
        anchor="w",
    )
    self.yolo_perf_label.pack(fill="x", padx=8, pady=(0, 2))
    self.yolo_perf_canvas = None

    self.yolo_model_path = ""
    self.yolo_model_label = ctk.CTkLabel(
        tab_std,
        text="Model: domyślny",
        text_color=TEXT_DIM,
        anchor="w",
    )
    self.yolo_model_label.pack(fill="x", padx=8, pady=(0, 4))

    model_row = ctk.CTkFrame(tab_std, fg_color="transparent")
    model_row.pack(fill="x", padx=8, pady=(0, 4))
    self.yolo_model_btn = ctk.CTkButton(
        model_row,
        text="📁 Wybierz model",
        command=self.choose_yolo_model,
        height=28,
    )
    self.yolo_model_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

    self.yolo_load_btn = ctk.CTkButton(
        model_row,
        text="⚡ Wybierz i załaduj",
        command=self._pick_and_load_yolo_model,
        height=28,
        fg_color="#0f766e",
        hover_color="#0d5f5a",
    )
    self.yolo_load_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

    conf_row = ctk.CTkFrame(tab_std, fg_color="transparent")
    conf_row.pack(fill="x", padx=8, pady=(2, 2))
    ctk.CTkLabel(conf_row, text="Czułość detekcji", text_color=TEXT_DIM).pack(side="left")
    self.yolo_conf_value = ctk.CTkLabel(
        conf_row,
        text="0.25",
        text_color=ACCENT,
        font=(FONT_FAMILY, 11, "bold"),
    )
    self.yolo_conf_value.pack(side="right")

    self.yolo_conf_slider = ctk.CTkSlider(
        tab_std,
        from_=0,
        to=1,
        number_of_steps=18,
        command=self.on_yolo_conf_change,
    )
    self.yolo_conf_slider.pack(fill="x", padx=8, pady=(0, 4))
    self.yolo_conf_slider.set(0.25)

    size_row = ctk.CTkFrame(tab_std, fg_color="transparent")
    size_row.pack(fill="x", padx=8, pady=(2, 2))
    ctk.CTkLabel(size_row, text="Rozmiar analizy", text_color=TEXT_DIM).pack(side="left")
    self.yolo_size_value = ctk.CTkLabel(
        size_row,
        text="640px",
        text_color=ACCENT,
        font=(FONT_FAMILY, 11, "bold"),
    )
    self.yolo_size_value.pack(side="right")

    self.yolo_size_slider = ctk.CTkSlider(
        tab_std,
        from_=320,
        to=1280,
        number_of_steps=12,
        command=self.on_yolo_size_change,
    )
    self.yolo_size_slider.pack(fill="x", padx=8, pady=(0, 4))
    self.yolo_size_slider.set(640)

    batch_row = ctk.CTkFrame(tab_std, fg_color="transparent")
    batch_row.pack(fill="x", padx=8, pady=(2, 2))
    ctk.CTkLabel(batch_row, text="Images processed at once", text_color=TEXT_DIM).pack(side="left")
    self.yolo_batch_value = ctk.CTkLabel(
        batch_row,
        text="4",
        text_color=ACCENT,
        font=(FONT_FAMILY, 11, "bold"),
    )
    self.yolo_batch_value.pack(side="right")

    self.yolo_batch_slider = ctk.CTkSlider(
        tab_std,
        from_=4,
        to=32,
        number_of_steps=7,
        command=self.on_yolo_batch_change,
    )
    self.yolo_batch_slider.pack(fill="x", padx=8, pady=(0, 6))
    self.yolo_batch_slider.set(4)

    std_btns = ctk.CTkFrame(tab_std, fg_color="transparent")
    std_btns.pack(fill="x", padx=8, pady=(2, 8))
    std_btns.grid_columnconfigure((0, 1), weight=1)

    self.yolo_btn.configure(text="🎯 Enable detection")
    self.yolo_btn.master = std_btns  # nie wpływa funkcjonalnie, tylko porządek referencji

    yolo_local_btn = self._make_btn(
        std_btns,
        "🎯 Enable detection",
        self.toggle_yolo,
        "#FF6B35",
        height=30,
        font_size=11,
    )
    yolo_local_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

    self.yolo_traj_btn = ctk.CTkButton(std_btns, text="📈 Show ball trajectory", command=self.open_trajectory_window,
                                       height=30, fg_color="#334155", hover_color="#1e293b", )


    self.yolo_traj_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    self.live_backend_label = ctk.CTkLabel(
        tab_pro,
        textvariable=self.live_backend_var,
        text_color=TEXT_DIM,
        anchor="w",
    )
    self.live_backend_label.pack(fill="x", padx=8, pady=(8, 4))

    pro_backend = ctk.CTkFrame(tab_pro, fg_color="transparent")
    pro_backend.pack(fill="x", padx=8, pady=(0, 4))

    self.live_backend_btn = ctk.CTkButton(
        pro_backend,
        text="Backend: Ultralytics",
        command=self.toggle_live_infer_backend,
        height=28,
    )
    self.live_backend_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

    self.live_engine_btn = ctk.CTkButton(
        pro_backend,
        text="Wybierz .engine",
        command=self.choose_live_trt_engine,
        height=28,
    )
    self.live_engine_btn.pack(side="right")

    self.live_track_status_label = ctk.CTkLabel(
        tab_pro,
        textvariable=self.live_track_status_var,
        text_color=TEXT_DIM,
        anchor="w",
    )
    self.live_track_status_label.pack(fill="x", padx=8, pady=(0, 2))

    self.live_track_metrics_label = ctk.CTkLabel(
        tab_pro,
        textvariable=self.live_track_metrics_var,
        text_color=TEXT_DIM,
        anchor="w",
    )
    self.live_track_metrics_label.pack(fill="x", padx=8, pady=(0, 2))

    self.live_compare_label = ctk.CTkLabel(
        tab_pro,
        textvariable=self.live_compare_var,
        text_color=TEXT_DIM,
        anchor="w",
    )
    self.live_compare_label.pack(fill="x", padx=8, pady=(0, 4))

    pro_btns = ctk.CTkFrame(tab_pro, fg_color="transparent")
    pro_btns.pack(fill="x", padx=8, pady=(2, 8))
    pro_btns.grid_columnconfigure((0, 1), weight=1)

    self.live_track_btn = ctk.CTkButton(
        pro_btns,
        text="Start LIVE_TRACK",
        command=self.toggle_live_track,
        fg_color="#0ea5e9",
        hover_color="#0284c7",
        height=30,
    )
    self.live_track_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

    self.live_save_stats_btn = ctk.CTkButton(
        pro_btns,
        text="Zapisz live stats",
        command=self.save_live_track_stats_snapshot,
        fg_color="#475569",
        hover_color="#334155",
        height=30,
    )
    self.live_save_stats_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    pro_expert = ctk.CTkFrame(tab_pro, fg_color="transparent")
    pro_expert.pack(fill="x", padx=8, pady=(0, 8))

    self.yolo_preprocess_label = ctk.CTkLabel(
        pro_expert,
        text="Preprocess: CPU",
        text_color=TEXT_DIM,
        anchor="w",
    )
    self.yolo_preprocess_label.pack(side="left", fill="x", expand=True)

    self.yolo_preprocess_btn = ctk.CTkButton(
        pro_expert,
        text="CPU/GPU",
        command=self.toggle_yolo_preprocess_backend,
        height=26,
        width=96,
    )
    self.yolo_preprocess_btn.pack(side="left", padx=(6, 6))

    self.yolo_debug_btn = ctk.CTkButton(
        pro_expert,
        text="Debug",
        command=self.debug_yolo,
        height=26,
        width=84,
    )
    self.yolo_debug_btn.pack(side="left")

    self._apply_live_backend_ui()
    self._apply_yolo_preprocess_backend_ui()

    # =========================================================
    # TAB: CAMERAS
    # =========================================================
    cameras_info = ctk.CTkFrame(tab_cameras, fg_color=CARD_BG, corner_radius=10)
    cameras_info.pack(fill="both", expand=True, padx=6, pady=(6, 6))

    ctk.CTkLabel(
        cameras_info,
        text="Ustawienia kamer",
        text_color=TEXT_MAIN,
        font=(FONT_FAMILY, 14, "bold"),
    ).pack(anchor="w", padx=10, pady=(8, 6))

    ctk.CTkLabel(
        cameras_info,
        text="Tutaj sterowanie ekspozycją i gain dla każdej kamery. "
             "Przeniesienie tych kontrolek z miniatur daje dużo większy preview.",
        text_color=TEXT_DIM,
        justify="left",
        wraplength=360,
    ).pack(anchor="w", padx=10, pady=(0, 8))

    cameras_scroll = ctk.CTkScrollableFrame(cameras_info, fg_color="transparent")
    cameras_scroll.pack(fill="both", expand=True, padx=6, pady=(0, 8))

    for role in self.roles:
        cam_ctrl = ctk.CTkFrame(cameras_scroll, fg_color=PANEL_BG, corner_radius=10)
        cam_ctrl.pack(fill="x", padx=4, pady=4)

        ctk.CTkLabel(
            cam_ctrl,
            text=role.upper(),
            text_color=TEXT_MAIN,
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 4))

        cam_ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cam_ctrl, text="EXP", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=1, column=0,
                                                                                             sticky="w",
                                                                                             padx=(8, 6))
        exp_slider = ctk.CTkSlider(
            cam_ctrl,
            from_=100,
            to=20000,
            command=lambda v, r=role: self.on_expo_change(v, r),
            fg_color="#334155",
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
        )
        exp_slider.grid(row=1, column=1, sticky="ew", padx=(0, 6))
        exp_val = ctk.CTkLabel(cam_ctrl, text="1000 µs", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
        exp_val.grid(row=1, column=2, sticky="e", padx=(0, 6))
        exp_entry = ctk.CTkEntry(cam_ctrl, width=72, placeholder_text="µs")
        exp_entry.grid(row=1, column=3, sticky="e", padx=(0, 8))
        exp_entry.bind("<Return>", lambda e, r=role, w=exp_entry: self._apply_from_entry(r, "exp", w))
        exp_entry.bind("<KP_Enter>", lambda e, r=role, w=exp_entry: self._apply_from_entry(r, "exp", w))

        ctk.CTkLabel(cam_ctrl, text="GAIN", text_color=TEXT_DIM, font=(FONT_FAMILY, 10)).grid(row=2, column=0,
                                                                                              sticky="w",
                                                                                              padx=(8, 6),
                                                                                              pady=(4, 8))
        gain_slider = ctk.CTkSlider(
            cam_ctrl,
            from_=0,
            to=24,
            command=lambda v, r=role: self.on_gain_change(v, r),
            fg_color="#334155",
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
        )
        gain_slider.grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=(4, 8))
        gain_val = ctk.CTkLabel(cam_ctrl, text="0.0 dB", text_color=TEXT_DIM, font=(FONT_FAMILY, 10))
        gain_val.grid(row=2, column=2, sticky="e", padx=(0, 6), pady=(4, 8))
        gain_entry = ctk.CTkEntry(cam_ctrl, width=72, placeholder_text="dB")
        gain_entry.grid(row=2, column=3, sticky="e", padx=(0, 8), pady=(4, 8))
        gain_entry.bind("<Return>", lambda e, r=role, w=gain_entry: self._apply_from_entry(r, "gain", w))
        gain_entry.bind("<KP_Enter>", lambda e, r=role, w=gain_entry: self._apply_from_entry(r, "gain", w))

        self.sliders[role] = {
            "exp_slider": exp_slider,
            "exp_val": exp_val,
            "exp_entry": exp_entry,
            "gain_slider": gain_slider,
            "gain_val": gain_val,
            "gain_entry": gain_entry,
        }

    # =========================================================
    # TAB: DIAGNOSTICS
    # =========================================================
    health_card = ctk.CTkFrame(tab_diag, fg_color=CARD_BG, corner_radius=10)
    health_card.pack(fill="x", padx=6, pady=(6, 6))

    ctk.CTkLabel(
        health_card,
        text="Sygnał wydolności systemu",
        text_color=TEXT_MAIN,
        font=(FONT_FAMILY, 13, "bold"),
    ).pack(anchor="w", padx=10, pady=(10, 2))

    self.system_health_var = ctk.StringVar(value="⏳ Oczekiwanie na pomiary...")
    self.system_latency_var = ctk.StringVar(value="Opóźnienie: -- ms | trend: --")

    self.system_health_label = ctk.CTkLabel(
        health_card,
        textvariable=self.system_health_var,
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 12, "bold"),
        anchor="w",
    )
    self.system_health_label.pack(fill="x", padx=10, pady=(0, 2))

    self.system_latency_label = ctk.CTkLabel(
        health_card,
        textvariable=self.system_latency_var,
        text_color=TEXT_DIM,
        font=(FONT_FAMILY, 11),
        anchor="w",
    )
    self.system_latency_label.pack(fill="x", padx=10, pady=(0, 6))

    self.system_health_bar = ctk.CTkProgressBar(
        health_card,
        fg_color="#334155",
        progress_color="#22c55e",
        height=14,
    )
    self.system_health_bar.pack(fill="x", padx=10, pady=(0, 10))
    self.system_health_bar.set(0.0)

    diag_logs = ctk.CTkFrame(tab_diag, fg_color=CARD_BG, corner_radius=10)
    diag_logs.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    ctk.CTkLabel(
        diag_logs,
        text="Status i feedback",
        text_color=TEXT_MAIN,
        font=(FONT_FAMILY, 13, "bold"),
    ).pack(anchor="w", padx=10, pady=(10, 6))

    self.stats_box = ctk.CTkTextbox(
        diag_logs,
        wrap="word",
        fg_color=PANEL_BG,
        text_color=TEXT_MAIN,
    )
    self.stats_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))
    self.stats_box.insert("end", "✅ System gotowy\n")

    self.session_summary_var = ctk.StringVar(value="Podsumowanie sesji: brak danych")
    ctk.CTkLabel(
        diag_logs,
        textvariable=self.session_summary_var,
        text_color=TEXT_DIM,
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=10, pady=(0, 10))

    # =========================================================
    # FOOTER — secondary actions only
    # =========================================================
    bottom = ctk.CTkFrame(self.main_frame, fg_color=CARD_BG, corner_radius=12)
    bottom.grid(row=2, column=0, sticky="ew", padx=pad, pady=(0, pad))
    bottom.grid_columnconfigure((0, 1, 2, 3), weight=1)

    ctk.CTkButton(
        bottom,
        text="📷 Camera assignment",
        command=self._open_camera_assignment,
    ).grid(row=0, column=0, padx=6, pady=8, sticky="ew")

    ctk.CTkButton(
        bottom,
        text="📊 Benchmark",
        command=self._open_benchmark_window,
    ).grid(row=0, column=1, padx=6, pady=8, sticky="ew")

    ctk.CTkButton(
        bottom,
        text="🛰 Remote",
        command=self.toggle_remote_server,
    ).grid(row=0, column=2, padx=6, pady=8, sticky="ew")

    self.btn_exit = ctk.CTkButton(
        bottom,
        text="❌ Exit",
        fg_color="#DDDDDD",
        hover_color="#CCCCCC",
        text_color="#555555",
        command=self.on_close_all,
    )
    self.btn_exit.grid(row=0, column=3, padx=6, pady=8, sticky="ew")

    # =========================================================
    # TOAST / REFS / RESIZE
    # =========================================================
    self.status_toast = ctk.CTkLabel(
        self.main_frame,
        text="",
        fg_color="#333333",
        text_color="#FFFFFF",
        font=(FONT_FAMILY, 13, "bold"),
        corner_radius=10,
        padx=20,
        pady=10,
    )
    self.status_toast.place_forget()

    self._dashboard_header = header
    self._dashboard_body = body
    self._dashboard_preview_area = preview_area
    self._dashboard_right = right
    self._dashboard_bottom = bottom
    self._dashboard_path_label = self.path_label
    self._dashboard_summary_label = self.session_summary_var

    try:
        self.main_frame.bind("<Configure>", self._on_dashboard_resize_preview_first)
    except Exception:
        pass

    self._refresh_dashboard_metrics()
    self._sync_operator_state_ui()

def _on_dashboard_resize_preview_first(self, event=None):
    try:
        total_w = int(self.main_frame.winfo_width())
        total_h = int(self.main_frame.winfo_height())
        if total_w <= 10 or total_h <= 10:
            return

        body_w = max(500, total_w - 24)

        preview_w = int(body_w * 0.68)
        right_w = max(320, body_w - preview_w)

        if hasattr(self, "_dashboard_body"):
            self._dashboard_body.grid_columnconfigure(0, minsize=preview_w)
            self._dashboard_body.grid_columnconfigure(1, minsize=right_w)

        if hasattr(self, "_dashboard_path_label"):
            self._dashboard_path_label.configure(wraplength=max(220, right_w - 40))

        if hasattr(self, "stats_box"):
            self.stats_box.configure(width=max(240, right_w - 30))

        if hasattr(self, "quick_start_btn"):
            btn_w = max(140, min(180, int(total_w * 0.14)))
            self.quick_start_btn.configure(width=btn_w)

    except Exception:
        pass

def _open_benchmark_window(self):
    if hasattr(self, "_bench_window") and self._bench_window is not None:
        try:
            if self._bench_window.winfo_exists():
                self._bench_window.lift()
                self._bench_window.focus_force()
                return
        except Exception:
            pass

    win = ctk.CTkToplevel(self)
    self._bench_window = win
    win.title("Benchmark")
    win.geometry("980x760")
    win.grid_rowconfigure(1, weight=1)
    win.grid_columnconfigure(0, weight=1)

    top = ctk.CTkFrame(win, fg_color=CARD_BG, corner_radius=10)
    top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
    top.grid_columnconfigure((0, 1, 2), weight=1)
    ctk.CTkButton(top, text="⚡ Quick Test", command=lambda: self._run_benchmark_preset("quick")).grid(row=0, column=0, sticky="ew", padx=6, pady=8)
    ctk.CTkButton(top, text="📊 Full Benchmark", command=lambda: self._run_benchmark_preset("full")).grid(row=0, column=1, sticky="ew", padx=6, pady=8)
    ctk.CTkButton(top, text="🔧 Advanced", command=lambda: tabs.set("Advanced")).grid(row=0, column=2, sticky="ew", padx=6, pady=8)

    tabs = ctk.CTkTabview(win)
    tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
    t_quick = tabs.add("Quick Test")
    t_full = tabs.add("Full Benchmark")
    t_adv = tabs.add("Advanced")

    ctk.CTkLabel(t_quick, text="Szybki test: jeden model, jedna rozdzielczość, krótki warmup.", text_color=TEXT_DIM).pack(anchor="w", padx=10, pady=(10, 6))
    ctk.CTkButton(t_quick, text="Start Quick Test", command=lambda: self._run_benchmark_preset("quick")).pack(fill="x", padx=10, pady=(0, 10))
    ctk.CTkLabel(t_full, text="Pełny benchmark z bezpiecznymi ustawieniami domyślnymi.", text_color=TEXT_DIM).pack(anchor="w", padx=10, pady=(10, 6))
    ctk.CTkButton(t_full, text="Start Full Benchmark", command=lambda: self._run_benchmark_preset("full")).pack(fill="x", padx=10, pady=(0, 10))

    adv = ctk.CTkScrollableFrame(t_adv, fg_color=PANEL_BG)
    adv.pack(fill="both", expand=True, padx=8, pady=8)
    self._build_auto_benchmark_advanced_ui(adv)
    self._update_auto_bench_plan_summary()

def _run_benchmark_preset(self, mode: str):
    if mode == "quick":
        self.ab_model_source_var.set("Pojedynczy model")
        self.ab_backend_mode_var.set("Ultralytics (.pt/.onnx)")
        self.ab_imgsz_var.set("640")
        self.ab_batch_mode_var.set("Lista")
        self.ab_batch_var.set("4")
        self.ab_warmup_s_var.set("2")
        self.ab_measure_s_var.set("6")
        self.ab_modes_var.set("Brak zapisu")
    else:
        self.ab_model_source_var.set("Wiele modeli (folder)")
        self.ab_backend_mode_var.set("Auto (wg modelu)")
        self.ab_imgsz_var.set("640,960")
        self.ab_batch_mode_var.set("Zakres")
        self.ab_batch_from_var.set("4")
        self.ab_batch_to_var.set("16")
        self.ab_batch_step_var.set("4")
        self.ab_warmup_s_var.set("4")
        self.ab_measure_s_var.set("10")
        self.ab_modes_var.set("Wszystkie kombinacje")
    self._on_auto_bench_model_source_change()
    self._on_auto_bench_batch_mode_change()
    self._on_auto_bench_write_mode_change()
    self.start_auto_benchmark()
    self._set_status(f"Benchmark start ({mode})", "accent")

def _build_auto_benchmark_advanced_ui(self, parent):
    for w in parent.winfo_children():
        w.destroy()
    ctk.CTkLabel(parent, text="Advanced Benchmark", text_color=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
    ctk.CTkLabel(
        parent,
        text="Wybierz zakres testu, backend i parametry. Podsumowanie pokaże dokładnie, co zostanie uruchomione.",
        text_color=TEXT_DIM,
        wraplength=820,
        justify="left",
    ).pack(anchor="w", padx=10, pady=(0, 8))

    scope_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
    scope_card.pack(fill="x", padx=8, pady=(0, 8))
    ctk.CTkLabel(scope_card, text="1) Zakres benchmarku", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

    row_source = ctk.CTkFrame(scope_card, fg_color="transparent")
    row_source.pack(fill="x", padx=10, pady=(0, 4))
    ctk.CTkLabel(row_source, text="Tryb modeli", text_color=TEXT_DIM).pack(side="left")
    self.ab_model_source_menu = ctk.CTkOptionMenu(
        row_source,
        variable=self.ab_model_source_var,
        values=["Pojedynczy model", "Wiele modeli (folder)"],
        command=self._on_auto_bench_model_source_change,
        width=230,
    )
    self.ab_model_source_menu.pack(side="right")

    row_models = ctk.CTkFrame(scope_card, fg_color="transparent")
    row_models.pack(fill="x", padx=10, pady=(0, 4))
    ctk.CTkLabel(row_models, text="Folder modeli (.pt/.onnx/.engine)", text_color=TEXT_DIM).pack(side="left")
    self.ab_models_dir_btn = ctk.CTkButton(row_models, text="Wybierz", width=90, command=self._choose_auto_bench_models_dir)
    self.ab_models_dir_btn.pack(side="right")
    self.ab_models_dir_entry = ctk.CTkEntry(scope_card, textvariable=self.ab_models_dir_var, height=28, placeholder_text="np. C:/.../models")
    self.ab_models_dir_entry.pack(fill="x", padx=10, pady=(0, 6))

    row_single = ctk.CTkFrame(scope_card, fg_color="transparent")
    row_single.pack(fill="x", padx=10, pady=(0, 4))
    self.ab_single_model_label = ctk.CTkLabel(row_single, text="Pojedynczy model", text_color=TEXT_DIM)
    self.ab_single_model_label.pack(side="left")
    self.ab_single_model_btn = ctk.CTkButton(row_single, text="Wybierz", width=90, command=self._choose_auto_bench_model_file)
    self.ab_single_model_btn.pack(side="right")
    self.ab_single_model_entry = ctk.CTkEntry(scope_card, textvariable=self.ab_single_model_var, height=28, placeholder_text="np. best.pt / model.engine")
    self.ab_single_model_entry.pack(fill="x", padx=10, pady=(0, 8))

    backend_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
    backend_card.pack(fill="x", padx=8, pady=(0, 8))
    ctk.CTkLabel(backend_card, text="2) Silnik inferencji", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
    row_backend = ctk.CTkFrame(backend_card, fg_color="transparent")
    row_backend.pack(fill="x", padx=10, pady=(0, 4))
    ctk.CTkLabel(row_backend, text="Backend", text_color=TEXT_DIM).pack(side="left")
    self.ab_backend_mode_menu = ctk.CTkOptionMenu(
        row_backend,
        variable=self.ab_backend_mode_var,
        values=["Auto (wg modelu)", "Ultralytics (.pt/.onnx)", "TensorRT (.engine)"],
        command=self._on_auto_bench_backend_mode_change,
        width=230,
    )
    self.ab_backend_mode_menu.pack(side="right")
    self.ab_backend_hint_var = ctk.StringVar(value="")
    ctk.CTkLabel(backend_card, textvariable=self.ab_backend_hint_var, text_color=TEXT_DIM, wraplength=820, justify="left").pack(
        anchor="w", padx=10, pady=(0, 8)
    )

    params_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
    params_card.pack(fill="x", padx=8, pady=(0, 8))
    ctk.CTkLabel(params_card, text="3) Parametry testu", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

    row_lists = ctk.CTkFrame(params_card, fg_color="transparent")
    row_lists.pack(fill="x", padx=10, pady=(0, 4))
    row_lists.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(row_lists, text="Rozdzielczość wejścia (imgsz)", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(row_lists, text="Batch obrazów (lista)", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
    self.ab_imgsz_combo = ctk.CTkComboBox(
        row_lists,
        variable=self.ab_imgsz_var,
        values=["320", "416", "512", "640", "736", "832", "960", "1024", "1280", "640,960", "640,832,1024"],
    )
    self.ab_imgsz_combo.grid(row=1, column=0, sticky="ew", padx=(0, 4))
    self.ab_batch_list_combo = ctk.CTkComboBox(
        row_lists,
        variable=self.ab_batch_var,
        values=["2", "4", "8", "12", "16", "20", "24", "28", "32", "4,8,12", "4,8,12,16"],
    )
    self.ab_batch_list_combo.grid(row=1, column=1, sticky="ew", padx=(4, 0))

    row_batch_mode = ctk.CTkFrame(params_card, fg_color="transparent")
    row_batch_mode.pack(fill="x", padx=10, pady=(0, 4))
    ctk.CTkLabel(row_batch_mode, text="Tryb ustawiania batch", text_color=TEXT_DIM).pack(side="left")
    self.ab_batch_mode_menu = ctk.CTkOptionMenu(
        row_batch_mode,
        variable=self.ab_batch_mode_var,
        values=["Zakres", "Lista"],
        command=self._on_auto_bench_batch_mode_change,
        width=180,
    )
    self.ab_batch_mode_menu.pack(side="right")

    row_batch_range = ctk.CTkFrame(params_card, fg_color="transparent")
    row_batch_range.pack(fill="x", padx=10, pady=(0, 4))
    row_batch_range.grid_columnconfigure((0, 1, 2), weight=1)
    ctk.CTkLabel(row_batch_range, text="Od", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(row_batch_range, text="Do", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
    ctk.CTkLabel(row_batch_range, text="Krok", text_color=TEXT_DIM).grid(row=0, column=2, sticky="w")
    batch_values = [str(v) for v in range(1, 65)]
    self.ab_batch_from_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_from_var, values=batch_values)
    self.ab_batch_to_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_to_var, values=batch_values)
    self.ab_batch_step_combo = ctk.CTkComboBox(row_batch_range, variable=self.ab_batch_step_var, values=["1", "2", "4", "8"])
    self.ab_batch_from_combo.grid(row=1, column=0, sticky="ew", padx=(0, 4))
    self.ab_batch_to_combo.grid(row=1, column=1, sticky="ew", padx=4)
    self.ab_batch_step_combo.grid(row=1, column=2, sticky="ew", padx=(4, 0))

    row_times = ctk.CTkFrame(params_card, fg_color="transparent")
    row_times.pack(fill="x", padx=10, pady=(2, 8))
    row_times.grid_columnconfigure((0, 1, 2), weight=1)
    ctk.CTkLabel(row_times, text="Warmup [s]", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(row_times, text="Pomiar [s]", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
    ctk.CTkLabel(row_times, text="FPS kamery", text_color=TEXT_DIM).grid(row=0, column=2, sticky="w")
    ctk.CTkEntry(row_times, textvariable=self.ab_warmup_s_var, height=28).grid(row=1, column=0, sticky="ew", padx=(0, 4))
    ctk.CTkEntry(row_times, textvariable=self.ab_measure_s_var, height=28).grid(row=1, column=1, sticky="ew", padx=4)
    self.ab_camera_fps_entry = ctk.CTkEntry(row_times, textvariable=self.ab_camera_fps_var, height=28)
    self.ab_camera_fps_entry.grid(row=1, column=2, sticky="ew", padx=(4, 0))
    self.ab_camera_fps_entry.bind("<Return>", lambda _e: self.on_apply_camera_fps())

    write_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
    write_card.pack(fill="x", padx=8, pady=(0, 8))
    ctk.CTkLabel(write_card, text="4) Tryb zapisu danych", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
    row_write_mode = ctk.CTkFrame(write_card, fg_color="transparent")
    row_write_mode.pack(fill="x", padx=10, pady=(0, 4))
    ctk.CTkLabel(row_write_mode, text="Co zapisywać", text_color=TEXT_DIM).pack(side="left")
    self.ab_modes_menu = ctk.CTkOptionMenu(
        row_write_mode,
        variable=self.ab_modes_var,
        values=["Wszystkie kombinacje", "Brak zapisu", "Tylko BIN", "Tylko BUFOR", "BIN + BUFOR"],
        command=self._on_auto_bench_write_mode_change,
        width=210,
    )
    self.ab_modes_menu.pack(side="right")

    row_write_params = ctk.CTkFrame(write_card, fg_color="transparent")
    row_write_params.pack(fill="x", padx=10, pady=(0, 8))
    row_write_params.grid_columnconfigure((0, 1, 2), weight=1)
    ctk.CTkLabel(row_write_params, text="Długość BIN [s]", text_color=TEXT_DIM).grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(row_write_params, text="Liczba zapisów bufora", text_color=TEXT_DIM).grid(row=0, column=1, sticky="w")
    ctk.CTkLabel(row_write_params, text="Przerwa bufora [s]", text_color=TEXT_DIM).grid(row=0, column=2, sticky="w")
    self.ab_bin_s_entry = ctk.CTkEntry(row_write_params, textvariable=self.ab_bin_s_var, height=28)
    self.ab_buffer_count_entry = ctk.CTkEntry(row_write_params, textvariable=self.ab_buffer_count_var, height=28)
    self.ab_buffer_pause_s_entry = ctk.CTkEntry(row_write_params, textvariable=self.ab_buffer_pause_s_var, height=28)
    self.ab_bin_s_entry.grid(row=1, column=0, sticky="ew", padx=(0, 4))
    self.ab_buffer_count_entry.grid(row=1, column=1, sticky="ew", padx=4)
    self.ab_buffer_pause_s_entry.grid(row=1, column=2, sticky="ew", padx=(4, 0))

    run_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=10)
    run_card.pack(fill="x", padx=8, pady=(0, 8))
    ctk.CTkLabel(run_card, text="5) Uruchomienie", text_color=ACCENT, font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
    row_buttons = ctk.CTkFrame(run_card, fg_color="transparent")
    row_buttons.pack(fill="x", padx=10, pady=(0, 6))
    row_buttons.grid_columnconfigure((0, 1), weight=1)
    self.ab_start_btn = ctk.CTkButton(row_buttons, text="▶ Start benchmark", command=self.start_auto_benchmark, height=30)
    self.ab_stop_btn = ctk.CTkButton(row_buttons, text="■ Stop", command=self.stop_auto_benchmark, fg_color=DANGER, hover_color="#B91C1C", height=30)
    self.ab_start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    self.ab_stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
    ctk.CTkLabel(run_card, textvariable=self.ab_status_var, text_color=TEXT_DIM, wraplength=820, justify="left").pack(anchor="w", padx=10, pady=(0, 2))
    ctk.CTkLabel(run_card, textvariable=self.ab_plan_var, text_color=ACCENT, wraplength=820, justify="left").pack(anchor="w", padx=10, pady=(0, 8))

    watch_vars = [
        self.ab_models_dir_var,
        self.ab_single_model_var,
        self.ab_imgsz_var,
        self.ab_batch_var,
        self.ab_batch_from_var,
        self.ab_batch_to_var,
        self.ab_batch_step_var,
        self.ab_modes_var,
        self.ab_camera_fps_var,
        self.ab_backend_mode_var,
    ]
    for _v in watch_vars:
        try:
            _v.trace_add("write", lambda *_args: self._update_auto_bench_plan_summary())
        except Exception:
            pass

    self._on_auto_bench_model_source_change()
    self._on_auto_bench_backend_mode_change()
    self._on_auto_bench_batch_mode_change()
    self._on_auto_bench_write_mode_change()
