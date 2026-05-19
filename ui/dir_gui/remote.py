"""
Auto-generated module extracted from ui/gui.py.
Do not edit manually unless you know what you are doing.
"""



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

def _poll_remote_url(self):
    try:
        if not hasattr(self, "shared_state"):
            return

        url = self.shared_state.get("remote_url", None)

        if url:
            # serwer online
            if not getattr(self, "_remote_shown", False):
                self._remote_shown = True
                self.remote_url_var.set(f"{url}")
                self.remote_status.configure(
                    text="Server ONLINE ✅", text_color="#2ECC71"
                )
        else:
            # serwer offline – wyczyść status jeśli wcześniej był ONLINE
            if getattr(self, "_remote_shown", False):
                self._remote_shown = False
                self.remote_url_var.set("Remote: OFF")
                self.remote_status.configure(
                    text="Server wyłączony", text_color=TEXT_DIM
                )
    except Exception as e:
        print(f"[GUI][⚠️] URL poll error: {e}")

    try:
        self._sync_operator_state_ui()
    except Exception:
        pass

    self._after(1000, self._poll_remote_url)

def toggle_remote_server(self):
    self.remote_on = not self.remote_on

    if self.remote_on:
        try:
            self.remote_status.configure(text="Server uruchamiany...", text_color="#555")
        except Exception:
            pass
        self._set_status("🛰 Uruchamianie remote access...", "accent")
    else:
        try:
            self._remote_shown = False
            self.remote_url_var.set("Remote: OFF")
            self.remote_status.configure(text="Server zatrzymywany...", text_color="#555")
        except Exception:
            pass
        self._set_status("🛑 Zatrzymywanie remote access...", "warning")

    self._sync_operator_state_ui()

    if getattr(self, "_initing", False):
        return

    try:
        self.control_q.put(("remote_stream", {"enabled": self.remote_on}))
        print(f"[GUI] remote_stream enabled={self.remote_on}")
    except Exception as e:
        print(f"[GUI] ⚠️ Error toggling remote server: {e}")

def _handle_remote_started(self, msg):
    self._insert_stat_text(f"🛰 {msg}\n")
    self._set_status("🛰 Serwer gotowy – kliknij link w statystykach", "success")

def _open_remote_url(self):
    """Otwiera w przeglądarce adres z self.remote_url_var, jeśli jest poprawny."""
    import webbrowser

    url = self.remote_url_var.get() or ""
    url = url.strip()
    if url.lower().startswith("http"):
        try:
            webbrowser.open(url)
            self._set_status(f"Otwieram: {url}", "accent")
        except Exception as e:
            print(f"[GUI] ⚠️ open url error: {e}")
            self._set_status("Nie udało się otworzyć linku.", "danger")
    else:
        self._set_status("Brak aktywnego adresu zdalnego podglądu.", "warning")
