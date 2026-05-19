# style_volleyhub.py
import customtkinter as ctk

# ===== KOLORY =====
BG_LIGHT       = "#F3F7FD"
CARD_BG        = "#FFFFFF"
CARD_INNER_BG  = "#FFFFFF"

TEXT_MAIN      = "#182233"
TEXT_DIM       = "#8D96A8"
BORDER_COLOR   = "#D9E4F5"

PRIMARY        = "#00AEEF"
PRIMARY_HOVER  = "#0ABEFF"
PRIMARY_DARK   = "#008CC8"

SUCCESS        = "#19A85B"
DANGER         = "#E34F4F"

SLIDER_BG      = "#D3DDEE"
PROGRESS_BG    = "#F1F4FA"

FONT_FAMILY    = "Segoe UI Variable"
PANEL_BG   = "#EEF2FA"
LIVE_BG    = "#FFFFFF"
ACCENT     = "#00AEEF"
ACCENT_HOVER = "#0ABEFF"




def init_ctk_theme():
    """Ustawia globalny light mode + kolory VolleyHub."""
    ctk.set_appearance_mode("light")
    # możesz zostawić domyślny 'blue' albo 'light-blue', 'dark-blue' itd.
    ctk.set_default_color_theme("blue")

# style_volleyhub.py
import customtkinter as ctk

# ... TU Twoje kolory + init_ctk_theme jak już masz ...


# ===== KOMPONENTY BRANDOWE VOLLEYHUB =====

class VHCard(ctk.CTkFrame):
    """Główna karta (np. dla jednej kamery)."""
    def __init__(self, master=None, **kwargs):
        kwargs.setdefault("fg_color", CARD_BG)
        kwargs.setdefault("corner_radius", 14)
        super().__init__(master, **kwargs)


class VHInnerBox(ctk.CTkFrame):
    """Biała ramka wewnątrz karty (np. box na slidery)."""
    def __init__(self, master=None, **kwargs):
        kwargs.setdefault("fg_color", CARD_INNER_BG)
        kwargs.setdefault("corner_radius", 10)
        super().__init__(master, **kwargs)


class VHPreviewFrame(ctk.CTkFrame):
    """Ramka podglądu z cienką obwódką."""
    def __init__(self, master=None, **kwargs):
        kwargs.setdefault("fg_color", "#FFFFFF")
        kwargs.setdefault("corner_radius", 10)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", BORDER_COLOR)
        super().__init__(master, **kwargs)


class VHSlider(ctk.CTkSlider):
    """Slider w kolorach VolleyHub (exp/gain)."""
    def __init__(self, master=None, **kwargs):
        kwargs.setdefault("fg_color", SLIDER_BG)
        kwargs.setdefault("progress_color", PRIMARY)
        kwargs.setdefault("button_color", PRIMARY)
        kwargs.setdefault("button_hover_color", PRIMARY_HOVER)
        super().__init__(master, **kwargs)


class VHEffBar(ctk.CTkProgressBar):
    """Pasek efektywności zapisu."""
    def __init__(self, master=None, **kwargs):
        kwargs.setdefault("fg_color", PROGRESS_BG)
        kwargs.setdefault("progress_color", PRIMARY)
        super().__init__(master, **kwargs)


class VHLabelTitle(ctk.CTkLabel):
    """Nagłówek kamery."""
    def __init__(self, master=None, **kwargs):
        kwargs.setdefault("text_color", TEXT_MAIN)
        kwargs.setdefault("font", (FONT_FAMILY, 15, "bold"))
        super().__init__(master, **kwargs)


class VHLabelDim(ctk.CTkLabel):
    """Opis / małe, przygaszone teksty."""
    def __init__(self, master=None, **kwargs):
        kwargs.setdefault("text_color", TEXT_DIM)
        kwargs.setdefault("font", (FONT_FAMILY, 11))
        super().__init__(master, **kwargs)
