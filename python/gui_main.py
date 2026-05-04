#!/usr/bin/env python3
"""
AIOS Core — Phase 6: Premium PyQt6 GUI (Modern Dark Theme)
============================================================
واجهة مستخدم عصرية ومبتكرة تليق بمحرك ذكاء اصطناعي متطور.
تصميم مستوحى من Cursor / Discord / macOS Modern UI.

CTO Directive: OVERRIDE COMMAND — This is the final architectural spec.
Runnable as a standalone mockup without engine binding (Phase 7 concern).
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QSlider, QProgressBar,
    QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QFont, QIcon, QColor, QPainter, QLinearGradient, QPen, QBrush,
)


# =============================================================================
#  QSS STYLESHEET — Modern Dark Theme (Catppuccin Mocha Inspired)
# =============================================================================

DARK_THEME_QSS = """
/* ───────────────────────── Global ───────────────────────── */
QMainWindow {
    background-color: #11111b;
}

QWidget {
    background-color: transparent;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'SF Pro Display', 'Inter', sans-serif;
    font-size: 13px;
}

/* ───────────────────────── Scrollbars ───────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: rgba(205, 214, 244, 0.15);
    min-height: 30px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(205, 214, 244, 0.3);
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 0;
    border: none;
}
QScrollBar::handle:horizontal {
    background: rgba(205, 214, 244, 0.15);
    min-width: 30px;
    border-radius: 3px;
}
QScrollBar::handle:horizontal:hover {
    background: rgba(205, 214, 244, 0.3);
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: none;
    width: 0;
}

/* ───────────────────────── Left Sidebar ───────────────────────── */
#sidebar {
    background-color: #181825;
    border-right: 1px solid rgba(205, 214, 244, 0.08);
}

#sidebarTitle {
    font-size: 20px;
    font-weight: 700;
    color: #89b4fa;
    letter-spacing: 1.5px;
    padding: 0;
}

#sidebarSubtitle {
    font-size: 10px;
    font-weight: 400;
    color: rgba(205, 214, 244, 0.4);
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0;
}

/* ───────────────────────── Section Headers ───────────────────────── */
#sectionHeader {
    font-size: 11px;
    font-weight: 600;
    color: rgba(205, 214, 244, 0.5);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0 0 6px 0;
}

/* ───────────────────────── Cards (Telemetry) ───────────────────────── */
#telemetryCard {
    background-color: #1e1e2e;
    border: 1px solid rgba(205, 214, 244, 0.06);
    border-radius: 10px;
    padding: 10px 12px;
}

#telemetryLabel {
    font-size: 9px;
    font-weight: 500;
    color: rgba(205, 214, 244, 0.4);
    letter-spacing: 1px;
    text-transform: uppercase;
}

#telemetryValue {
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 18px;
    font-weight: 700;
    color: #cdd6f4;
    padding: 2px 0 0 0;
}

/* ───────────────────────── Buttons ───────────────────────── */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid rgba(205, 214, 244, 0.08);
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.3px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: rgba(205, 214, 244, 0.15);
}
QPushButton:pressed {
    background-color: #585b70;
}

#browseBtn {
    background-color: #313244;
    color: #a6adc8;
    border: 1px dashed rgba(205, 214, 244, 0.15);
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 12px;
}
#browseBtn:hover {
    background-color: #3b3b52;
    border-color: rgba(137, 180, 250, 0.3);
    color: #89b4fa;
}

#compileBtn {
    background-color: #89b4fa;
    color: #11111b;
    border: none;
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 12px;
    font-weight: 600;
}
#compileBtn:hover {
    background-color: #a6c8ff;
}
#compileBtn:pressed {
    background-color: #74a8f8;
}

/* ───────────────────────── Send & Stop Buttons ───────────────────────── */
#sendBtn {
    background-color: #89b4fa;
    color: #11111b;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
#sendBtn:hover {
    background-color: #a6c8ff;
}
#sendBtn:pressed {
    background-color: #74a8f8;
}

#stopBtn {
    background-color: rgba(243, 139, 168, 0.15);
    color: #f38ba8;
    border: 1px solid rgba(243, 139, 168, 0.25);
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 13px;
    font-weight: 600;
}
#stopBtn:hover {
    background-color: rgba(243, 139, 168, 0.25);
    border-color: rgba(243, 139, 168, 0.4);
}
#stopBtn:pressed {
    background-color: rgba(243, 139, 168, 0.35);
}

/* ───────────────────────── ComboBox ───────────────────────── */
QComboBox {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid rgba(205, 214, 244, 0.08);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    padding-right: 30px;
}
QComboBox:hover {
    border-color: rgba(137, 180, 250, 0.3);
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid rgba(205, 214, 244, 0.5);
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid rgba(205, 214, 244, 0.1);
    border-radius: 8px;
    selection-background-color: #313244;
    selection-color: #89b4fa;
    padding: 4px;
    outline: none;
}

/* ───────────────────────── Progress Bar ───────────────────────── */
QProgressBar {
    background-color: #1e1e2e;
    border: none;
    border-radius: 4px;
    min-height: 4px;
    max-height: 4px;
    padding: 0;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 4px;
}

/* ───────────────────────── Slider ───────────────────────── */
QSlider::groove:horizontal {
    background-color: #1e1e2e;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background-color: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background-color: #a6c8ff;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:1 #b4d0fb
    );
    border-radius: 2px;
}

/* ───────────────────────── Text Edits ───────────────────────── */
#displayArea {
    background-color: #11111b;
    color: #cdd6f4;
    border: none;
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 14px;
    line-height: 1.6;
}
#displayArea:focus {
    border: none;
}

#inputArea {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid rgba(205, 214, 244, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.5;
}
#inputArea:focus {
    border-color: rgba(137, 180, 250, 0.4);
}

/* ───────────────────────── Separator ───────────────────────── */
#separator {
    background-color: rgba(205, 214, 244, 0.06);
    max-height: 1px;
    min-height: 1px;
}

/* ───────────────────────── Gear Label ───────────────────────── */
#gearLabel {
    font-size: 11px;
    font-weight: 500;
    color: #89b4fa;
    letter-spacing: 0.5px;
}
"""


# =============================================================================
#  CUSTOM WIDGETS — Telemetry Cards & Glowing Label
# =============================================================================

class TelemetryCard(QFrame):
    """بطاقة مراقبة حية واحدة (RAM / CPU / Speed)."""

    def __init__(self, icon: str, label: str, value: str = "0", parent=None):
        super().__init__(parent)
        self.setObjectName("telemetryCard")
        self.setFixedHeight(72)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        # Top row: icon + label
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        self.icon_label = QLabel(icon)
        self.icon_label.setObjectName("telemetryLabel")
        self.icon_label.setFixedWidth(18)
        self.text_label = QLabel(label)
        self.text_label.setObjectName("telemetryLabel")
        top_row.addWidget(self.icon_label)
        top_row.addWidget(self.text_label)
        top_row.addStretch()

        # Value
        self.value_label = QLabel(value)
        self.value_label.setObjectName("telemetryValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout.addLayout(top_row)
        layout.addWidget(self.value_label)

    def set_value(self, text: str):
        self.value_label.setText(text)


class GradientTitle(QLabel):
    """عنوان بتدرج لوني من الأزرق للبنفسجي."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._gradient_colors = [
            QColor(137, 180, 250),   # #89b4fa blue
            QColor(203, 166, 247),   # #cba6f7 mauve
        ]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0.0, self._gradient_colors[0])
        gradient.setColorAt(1.0, self._gradient_colors[1])

        font = self.font()
        calc_size = int(font.pointSizeF() * 96 / 72)
        if calc_size > 0:
            font.setPixelSize(calc_size)
        painter.setFont(font)
        painter.setPen(QPen(QBrush(gradient), 1))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )
        painter.end()


# =============================================================================
#  MAIN WINDOW
# =============================================================================

class AIOSCoreGUI(QMainWindow):
    """النافذة الرئيسية لواجهة AIOS Core."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIOS Core — Neural Engine Interface")
        self.setMinimumSize(1200, 800)
        self.resize(1200, 800)

        # ── Central Widget ──
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ═══════════════════════════════════════════════════════════════
        #  LEFT SIDEBAR — The Control Center (1 part of 4)
        # ═══════════════════════════════════════════════════════════════
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(310)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)

        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(20, 28, 20, 20)
        sb_layout.setSpacing(0)

        # ── Brand Header ──
        brand_title = GradientTitle("AIOS Core")
        brand_title.setObjectName("sidebarTitle")
        brand_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        sb_layout.addWidget(brand_title)

        brand_sub = QLabel("NEURAL ENGINE v6.0")
        brand_sub.setObjectName("sidebarSubtitle")
        sb_layout.addWidget(brand_sub)

        sb_layout.addSpacing(24)

        # ── Separator ──
        sep1 = QFrame()
        sep1.setObjectName("separator")
        sb_layout.addWidget(sep1)

        sb_layout.addSpacing(20)

        # ── Section: Model Management ──
        sb_layout.addWidget(self._section_header("MODELS"))

        sb_layout.addSpacing(10)

        self.browse_btn = QPushButton("📁  Browse Model File")
        self.browse_btn.setObjectName("browseBtn")
        sb_layout.addWidget(self.browse_btn)

        sb_layout.addSpacing(8)

        self.compile_btn = QPushButton("⚡  Compile to AIOC")
        self.compile_btn.setObjectName("compileBtn")
        sb_layout.addWidget(self.compile_btn)

        sb_layout.addSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        sb_layout.addWidget(self.progress_bar)

        sb_layout.addSpacing(16)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "No model loaded",
            "tiny-llm-50m.aioc",
            "micro-gpt-20m.aioc",
            "nano-transformer-10m.aioc",
        ])
        self.model_combo.setCurrentIndex(0)
        sb_layout.addWidget(self.model_combo)

        sb_layout.addSpacing(24)

        # ── Separator ──
        sep2 = QFrame()
        sep2.setObjectName("separator")
        sb_layout.addWidget(sep2)

        sb_layout.addSpacing(20)

        # ── Section: Performance Governor ──
        sb_layout.addWidget(self._section_header("GOVERNOR"))

        sb_layout.addSpacing(12)

        self.gear_label = QLabel("⚙️  Gear 5: Balanced")
        self.gear_label.setObjectName("gearLabel")
        sb_layout.addWidget(self.gear_label)

        sb_layout.addSpacing(8)

        self.governor_slider = QSlider(Qt.Orientation.Horizontal)
        self.governor_slider.setRange(1, 10)
        self.governor_slider.setValue(5)
        self.governor_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.governor_slider.valueChanged.connect(self._on_gear_changed)
        sb_layout.addWidget(self.governor_slider)

        # Gear range labels
        gear_range = QHBoxLayout()
        gear_range.setContentsMargins(0, 4, 0, 0)
        gear_range.setSpacing(0)
        lbl_min = QLabel("Eco")
        lbl_min.setStyleSheet("font-size: 9px; color: rgba(205,214,244,0.3); letter-spacing: 0.5px;")
        lbl_max = QLabel("Max")
        lbl_max.setStyleSheet("font-size: 9px; color: rgba(205,214,244,0.3); letter-spacing: 0.5px;")
        lbl_max.setAlignment(Qt.AlignmentFlag.AlignRight)
        gear_range.addWidget(lbl_min)
        gear_range.addStretch()
        gear_range.addWidget(lbl_max)
        sb_layout.addLayout(gear_range)

        sb_layout.addSpacing(24)

        # ── Separator ──
        sep3 = QFrame()
        sep3.setObjectName("separator")
        sb_layout.addWidget(sep3)

        sb_layout.addSpacing(20)

        # ── Section: Telemetry Dashboard ──
        sb_layout.addWidget(self._section_header("TELEMETRY"))

        sb_layout.addSpacing(12)

        self.card_ram = TelemetryCard("💾", "RAM Usage", "0 MB")
        sb_layout.addWidget(self.card_ram)

        sb_layout.addSpacing(8)

        self.card_cpu = TelemetryCard("🔥", "CPU Load", "0 %")
        sb_layout.addWidget(self.card_cpu)

        sb_layout.addSpacing(8)

        self.card_speed = TelemetryCard("⚡", "Speed", "0 T/s")
        sb_layout.addWidget(self.card_speed)

        sb_layout.addStretch()

        # ── Version footer ──
        footer = QLabel("AIOS Core © 2026 — Phase 6/7")
        footer.setStyleSheet("font-size: 9px; color: rgba(205,214,244,0.2); letter-spacing: 1px;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_layout.addWidget(footer)

        # ═══════════════════════════════════════════════════════════════
        #  MAIN AREA — The Interaction Hub (3 parts of 4)
        # ═══════════════════════════════════════════════════════════════
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        main_layout.setContentsMargins(24, 24, 24, 20)
        main_layout.setSpacing(16)

        # ── Display Area (Chat Output) ──
        self.display_area = QTextEdit()
        self.display_area.setObjectName("displayArea")
        self.display_area.setReadOnly(True)
        self.display_area.setPlaceholderText("AIOS Core is ready. Start a conversation...")
        self.display_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Welcome message
        self.display_area.setHtml(
            '<div style="margin-top: 40px;">'
            '<div style="font-size: 28px; font-weight: 700; color: #89b4fa; margin-bottom: 8px;">'
            'Welcome to AIOS Core</div>'
            '<div style="font-size: 14px; color: rgba(205,214,244,0.5); line-height: 1.7;">'
            'Neural inference engine optimized for ultra-low-resource devices.<br>'
            'Load a model from the sidebar or type a message below to begin.<br><br>'
            '<span style="color: rgba(205,214,244,0.3); font-size: 12px;">'
            '▸ Ternary Weight Engine &nbsp; ▸ Sovereign Arena Memory &nbsp; ▸ Async Prefetch Pipeline'
            '</span></div></div>'
        )

        main_layout.addWidget(self.display_area, stretch=1)

        # ── Input Area ──
        input_container = QFrame()
        input_container.setStyleSheet("background-color: transparent;")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(10)

        self.input_area = QTextEdit()
        self.input_area.setObjectName("inputArea")
        self.input_area.setPlaceholderText("Type your message here...")
        self.input_area.setMaximumHeight(140)
        self.input_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.input_area.setFocus()

        input_layout.addWidget(self.input_area)

        # ── Buttons Row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setFixedHeight(42)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self.stop_btn)

        self.send_btn = QPushButton("Send  ➤")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedHeight(42)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setDefault(True)
        btn_row.addWidget(self.send_btn)

        input_layout.addLayout(btn_row)
        main_layout.addWidget(input_container)

        # ── Assemble Root ──
        root_layout.addWidget(sidebar_scroll, stretch=1)
        root_layout.addWidget(main_area, stretch=3)

        # ═══════════════════════════════════════════════════════════════
        #  SIGNALS & CONNECTIONS
        # ═══════════════════════════════════════════════════════════════
        self.send_btn.clicked.connect(self._on_send)
        self.stop_btn.clicked.connect(self._on_stop)
        self.browse_btn.clicked.connect(self._on_browse)
        self.compile_btn.clicked.connect(self._on_compile)

        # Simulate telemetry (mockup)
        self._mock_telemetry_timer = QTimer(self)
        self._mock_telemetry_timer.timeout.connect(self._update_mock_telemetry)
        self._mock_telemetry_timer.start(2000)

        # Simulated counter for mock
        self._mock_tick = 0

    # ────────────────────────── Helper: Section Header ──────────────────────────
    @staticmethod
    def _section_header(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionHeader")
        return lbl

    # ────────────────────────── Gear Slider ──────────────────────────
    _GEAR_NAMES = {
        1: "Eco Mode",
        2: "Ultra Light",
        3: "Light",
        4: "Moderate",
        5: "Balanced",
        6: "Performance",
        7: "High Power",
        8: "Aggressive",
        9: "Overdrive",
        10: "Maximum Power",
    }

    def _on_gear_changed(self, value: int):
        name = self._GEAR_NAMES.get(value, "")
        self.gear_label.setText(f"⚙️  Gear {value}: {name}")

    # ────────────────────────── Mock Telemetry ──────────────────────────
    def _update_mock_telemetry(self):
        """محاكاة بيانات المراقبة الحية (Mockup فقط)."""
        import random
        self._mock_tick += 1

        # Simulate slight variations
        ram = 180 + self._mock_tick * 3 + random.randint(-5, 5)
        cpu = 12 + random.randint(0, 8)
        speed = 4 + random.randint(0, 3) + (self._mock_tick % 5)

        self.card_ram.set_value(f"{ram} MB")
        self.card_cpu.set_value(f"{cpu} %")
        self.card_speed.set_value(f"{speed} T/s")

    # ────────────────────────── Button Handlers (Mockup) ──────────────────────────
    def _on_send(self):
        """إرسال رسالة المستخدم (Mockup)."""
        text = self.input_area.toPlainText().strip()
        if not text:
            return

        # Append user message to display
        self.display_area.append(
            f'<div style="margin: 16px 0 8px 0;">'
            f'<span style="color: #89b4fa; font-weight: 600; font-size: 12px;">YOU</span></div>'
            f'<div style="color: #cdd6f4; padding-left: 12px; line-height: 1.6;">'
            f'{self._escape_html(text)}</div>'
        )

        # Clear input
        self.input_area.clear()

        # Simulate AI response (mockup)
        self.display_area.append(
            f'<div style="margin: 20px 0 8px 0;">'
            f'<span style="color: #cba6f7; font-weight: 600; font-size: 12px;">AIOS</span></div>'
            f'<div style="color: rgba(205,214,244,0.6); padding-left: 12px; font-style: italic; line-height: 1.6;">'
            f'[Engine not bound — Phase 7 will connect the neural pipeline. '
            f'Mockup response placeholder.]</div>'
        )

        # Scroll to bottom
        scrollbar = self.display_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_stop(self):
        """إيقاف التوليد (Mockup)."""
        self.display_area.append(
            '<div style="color: #f38ba8; font-size: 11px; padding: 4px 0;">'
            '⏹ Generation stopped by user.</div>'
        )

    def _on_browse(self):
        """اختيار ملف نموذج (Mockup)."""
        self.display_area.append(
            '<div style="color: #a6adc8; font-size: 11px; padding: 4px 0;">'
            '📁 Browse dialog would open here (mockup).</div>'
        )

    def _on_compile(self):
        """تحويل النموذج (Mockup مع محاكاة شريط التقدم)."""
        self.display_area.append(
            '<div style="color: #89b4fa; font-size: 11px; padding: 4px 0;">'
            '⚡ Starting AIOC compilation...</div>'
        )
        self._compile_step = 0
        self._compile_timer = QTimer(self)
        self._compile_timer.timeout.connect(self._compile_tick)
        self._compile_timer.start(50)

    def _compile_tick(self):
        """محاكاة خطوات التحويل."""
        self._compile_step += 1
        self.progress_bar.setValue(min(self._compile_step, 100))

        if self._compile_step >= 100:
            self._compile_timer.stop()
            self.display_area.append(
                '<div style="color: #a6e3a1; font-size: 11px; padding: 4px 0;">'
                '✅ Compilation complete — model.aioc ready.</div>'
            )
            self.model_combo.setCurrentIndex(1)

    # ────────────────────────── Utilities ──────────────────────────
    @staticmethod
    def _escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    # High-DPI support — MUST be set BEFORE creating QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # Apply dark theme globally
    app.setStyleSheet(DARK_THEME_QSS)

    window = AIOSCoreGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
