STYLE_CAMERA_UI = """
/* ===== GLOBAL ===== */
QMainWindow, QWidget {
    background-color: #F3F7FD;
    color: #182233;
    font-family: 'Segoe UI Variable', 'Inter', 'Segoe UI';
    font-size: 14px;
}

/* ===== LEWY PANEL ===== */
QWidget#leftPanel {
    background-color: #EDF2FA;
}

/* ===== KARTY (sekcje w sidebarze) ===== */
QFrame#card {
    background-color: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #DFE6F2;
    padding: 12px 14px;
    margin: 6px 8px;
}

QLabel[role="cardTitle"] {
    font-size: 15px;
    font-weight: 700;
    color: #1C2C4B;
}

QLabel[role="cardSubtitle"] {
    font-size: 12px;
    color: #8C96AA;
}

/* ===== LABELS OGÓLNE ===== */
QLabel {
    color: #1C2C4B;
}
QLabel[dim="true"] {
    color: #97A0B5;
}

/* ===== PRZYCISKI – bazowy ===== */
QPushButton {
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
    border: none;
}

/* PRIMARY (np. Włącz podgląd, Zdjęcie) */
QPushButton#primaryBtn {
    background-color: #1C7DF2;
    color: #FFFFFF;
}
QPushButton#primaryBtn:hover {
    background-color: #2C8BFF;
}
QPushButton#primaryBtn:pressed {
    background-color: #1564C2;
}

/* SUCCESS (Rozpocznij) */
QPushButton#successBtn {
    background-color: #19A85B;
    color: #FFFFFF;
}
QPushButton#successBtn:hover {
    background-color: #22BD68;
}
QPushButton#successBtn:pressed {
    background-color: #158548;
}

/* DANGER (Zakończ) */
QPushButton#dangerBtn {
    background-color: #E34F4F;
    color: #FFFFFF;
}
QPushButton#dangerBtn:hover {
    background-color: #F16060;
}
QPushButton#dangerBtn:pressed {
    background-color: #C73F3F;
}

/* SECONDARY / GHOST (Raport zapisu, Remote OFF itp.) */
QPushButton#secondaryBtn {
    background-color: #FFFFFF;
    color: #1C7DF2;
    border: 1px solid #C3D7F3;
}
QPushButton#secondaryBtn:hover {
    background-color: #F3F7FF;
}

/* DISABLED */
QPushButton:disabled {
    background-color: #E0E6F0;
    color: #A9B3C6;
    border: none;
}

/* ===== SLIDER ===== */
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #D3DDEE;
}
QSlider::sub-page:horizontal {
    background: #1C7DF2;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #1C7DF2;
    width: 14px;
    border-radius: 7px;
    margin: -6px 0;
}

/* ===== SPINBOXY / LINEEDIT ===== */
QSpinBox, QDoubleSpinBox, QLineEdit {
    border: 1px solid #CED6E5;
    border-radius: 6px;
    padding: 4px 6px;
    background-color: #FFFFFF;
}

QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 1px solid #1C7DF2;
}

/* ===== STOPKA / PRZYCISK 'Zamknij aplikację' ===== */
QPushButton#closeAppBtn {
    background-color: #E4E7EF;
    color: #555E70;
    border-radius: 12px;
    padding: 10px 16px;
}
QPushButton#closeAppBtn:hover {
    background-color: #D6DAE5;
}

/* ===== SCROLLBARY (jeśli są) ===== */
QScrollBar:vertical, QScrollBar:horizontal {
    background: transparent;
    width: 10px;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle {
    background: rgba(28,125,242,0.35);
    border-radius: 5px;
}
QScrollBar::handle:hover {
    background: rgba(28,125,242,0.55);
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: none;
    border: none;
}
"""
