#!/usr/bin/env python3
"""
AIOS Core — Phase 7: Integration & Async Multithreading
==========================================================
المرحلة السابعة والأخيرة: الدمج الشامل بين النواة + الجسر + المحول + الواجهة.

This module is the MAIN ENTRY POINT for the AIOS Core application.
It wires together:
  - gui_main.py   (Phase 6 — Dumb View / PyQt6 GUI)
  - engine_bridge.py (Phase 5 — AIOCEngine + PerformanceGovernor)
  - model_compiler.py (Phase 4 — ModelCompiler → AIOC)

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │  AppController (this file)                               │
  │  ├── AIOSCoreGUI  (gui_main.py — untouched dumb view)    │
  │  ├── CompilerWorker  (QThread → ModelCompiler)           │
  │  ├── InferenceWorker (QThread → AIOCEngine pipeline)     │
  │  ├── PerformanceGovernor + AIOCEngine (engine_bridge.py) │
  │  └── QTimer Telemetry (psutil RAM/CPU every 1s)          │
  └──────────────────────────────────────────────────────────┘

Constraints:
  - Phase 1–5 files: NOT modified.
  - gui_main.py: NOT modified (dumb view pattern).
  - All heavy ops (>10ms) run in QThread via pyqtSignal.

Author: AI OS Core Team
Version: 1.0.0
Since: 2026-04-09
License: MIT
"""

from __future__ import annotations

import sys
import subprocess
import os
import time
import logging
import threading
from typing import List, Optional, Tuple

# 🚀 AIOS Core: Surgical Path Resolver + Auto-Installer
# =============================================================================
# يحقن user site-packages (AppData) في sys.path قبل أي import،
# ثم يسطب المكتبات الناقصة إن وُجدت، ويعيد تحديث الـ cache.
# هذا يحل مشكلة: "pip installs to AppData but python.exe can't see it"
# =============================================================================
import site
import importlib

def _bootstrap_dependencies() -> None:
    """
    الخطوة 1 — حقن user site-packages في sys.path فوراً (قبل أي import).
    الخطوة 2 — لو safetensors لسه مش موجود، سطّبه بـ --user وحقن المسار تاني.
    الخطوة 3 — invalidate_caches() عشان Python يشوف الملفات الجديدة فوراً.
    """
    # ── Inject user site-packages (AppData/.../site-packages) ──
    user_site = site.getusersitepackages()
    if user_site and os.path.isdir(user_site) and user_site not in sys.path:
        sys.path.insert(1, user_site)

    # ── Fallback: PYTHONUSERBASE/Lib/site-packages ──
    fallback = os.path.join(site.getuserbase(), "Lib", "site-packages")
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(1, fallback)

    # ── Try importing; install only if truly missing ──
    try:
        import safetensors  # noqa: F401  (just a presence check)
    except ImportError:
        print("[AIOS Bootstrap] Installing missing libraries (--user)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "safetensors", "torch", "numpy", "psutil", "PyQt6",
            "--user", "-q",
        ])
        # Re-inject in case pip created the directory for the first time
        new_user_site = site.getusersitepackages()
        if new_user_site and new_user_site not in sys.path:
            sys.path.insert(1, new_user_site)

    # ── Force Python to discover newly installed packages ──
    importlib.invalidate_caches()

_bootstrap_dependencies()

# ── PyQt6 ──

# ── PyQt6 ──
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QObject,
)

# ── Local modules (Phase 4, 5, 6 — UNTOUCHED) ──
from gui_main import AIOSCoreGUI, DARK_THEME_QSS

try:
    from model_compiler import ModelCompiler
    _HAS_COMPILER = True
except ImportError:
    _HAS_COMPILER = False
    logging.warning(
        "model_compiler.py not importable — torch/safetensors missing. "
        "Compilation feature will be disabled."
    )

try:
    from engine_bridge import (
        AIOCEngine,
        PerformanceGovernor,
        create_engine_with_governor,
        detect_system_ram_mb,
    )
    _HAS_ENGINE = True
    print("✅ ENGINE LOADED SUCCESSFULLY!")
except Exception as e:
    _HAS_ENGINE = False
    print("\n" + "🔥"*20)
    print(f"🚨 الكارثة الحقيقية اللي كانت مستخبية هي:")
    print(f"ERROR: {e}")
    print("🔥"*20 + "\n")
    logging.error(f"Engine import failed: {e}")
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logging.warning("psutil not installed — telemetry will show limited data.")

logger = logging.getLogger("ai_os.app")


# =============================================================================
#  CompilerWorker — QThread for Model Compilation
# =============================================================================
#  Runs ModelCompiler.export_to_aioc() in a background thread.
#  Emits progress_signal(int) for each layer, finished_signal on completion.
#  The GUI thread stays 100% responsive during compilation.
# =============================================================================

class CompilerWorker(QThread):
    """
    خيط خلفي لتشغيل ModelCompiler.export_to_aioc() بدون تجميد الواجهة.

    Signals:
        progress_signal(int): emits 0–100 as compilation progresses.
        finished_signal(bool, str): emits (success, message) when done.
    """

    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        threshold: float = 0.05,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._input_path = input_path
        self._output_path = output_path
        self._threshold = threshold
        self._abort_flag = False

    def abort(self) -> None:
        """طلب إيقاف التحويل (يُفحص بين كل طبقة)."""
        self._abort_flag = True

    def run(self) -> None:
        """تنفيذ التحويل في الخيط الخلفي."""
        if not _HAS_COMPILER:
            self.finished_signal.emit(False, "ModelCompiler unavailable (torch missing).")
            return

        try:
            compiler = ModelCompiler(
                input_path=self._input_path,
                output_path=self._output_path,
                threshold=self._threshold,
            )

            def _progress_callback(
                layer_idx: int,
                total_layers: int,
                layer_name: str,
                elapsed_ms: float,
            ) -> None:
                if self._abort_flag:
                    raise InterruptedError("Compilation aborted by user.")
                pct = int((layer_idx + 1) / total_layers * 100)
                self.progress_signal.emit(pct)

            report = compiler.export_to_aioc(callback=_progress_callback)

            if self._abort_flag:
                self.finished_signal.emit(False, "Compilation aborted.")
                return

            summary = (
                f"Compiled {report.num_layers} layers "
                f"({report.total_bytes / (1024*1024):.1f} MB) "
                f"in {report.total_export_ms:.0f} ms"
            )
            self.finished_signal.emit(True, summary)
            logger.info(summary)

        except InterruptedError:
            self.finished_signal.emit(False, "Compilation aborted by user.")
        except Exception as e:
            error_msg = f"Compilation failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.finished_signal.emit(False, error_msg)


# =============================================================================
#  InferenceWorker — QThread for Token Generation
# =============================================================================
#  Runs the inference loop (wait→commit→process→retire per layer)
#  in a background thread. Emits token_generated(str) for streaming.
#  Supports stop via boolean flag (non-blocking).
# =============================================================================

class InferenceWorker(QThread):
    """
    خيط خلفي لحلقة الاستدلال التكرارية (Autoregressive Token Generation).

    Signals:
        token_generated(str): emits each generated token/word for streaming.
        finished_signal(bool, str): emits (success, full_text_or_error).
    """

    token_generated = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        engine: Optional[object],
        prompt: str,
        layer_dims: Optional[List[Tuple[int, int]]] = None,
        max_tokens: int = 256,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._prompt = prompt
        self._layer_dims = layer_dims or []
        self._max_tokens = max_tokens
        self._stop_flag = False

    def request_stop(self) -> None:
        """طلب إيقاف التوليد (آمن — يُفحص بين كل توكن)."""
        self._stop_flag = True

    def run(self) -> None:
        """
        تنفيذ حلقة التوليد في الخيط الخلفي.

        إذا كان المحرك غير متاح (demo mode)، يولّد نصاً تجريبياً
        يحاكي السلوك الحقيقي مع تأخير واقعي.
        """
        generated_text = ""

        try:
            # ── Real Engine Path ──
            if (
                _HAS_ENGINE
                and self._engine is not None
                and self._engine.is_initialized
                and self._layer_dims
            ):
                generated_text = self._run_real_inference()
            else:
                # ── Demo / Simulated Path ──
                generated_text = self._run_demo_inference()

            self.finished_signal.emit(True, generated_text)

        except Exception as e:
            error_msg = f"Inference error: {e}"
            logger.error(error_msg, exc_info=True)
            self.finished_signal.emit(False, error_msg)

    def _run_real_inference(self) -> str:
        """
        استدلال حقيقي عبر AIOCEngine pipeline.

        يستخدم:
          - engine.wait_layer() → commit_layer() → process_layer() → retire_layer()
          - engine.reload_model() لكل توكن جديد (autoregressive)
        """
        import numpy as np

        generated_tokens: List[str] = []
        engine = self._engine
        dims = self._layer_dims

        for token_idx in range(self._max_tokens):
            if self._stop_flag:
                break

            # Token-level timing for speed telemetry
            t0 = time.perf_counter()

            # Run forward pass through all layers
            # For the first token, the model is already loaded.
            # For subsequent tokens, reload (reset + load from page cache).
            if token_idx > 0:
                engine.reload_model()

            # Build dummy input (in real usage, this would be the token embedding)
            in_dim = dims[0][0] if dims else 1
            input_vec = np.zeros(in_dim, dtype=np.float32)

            for layer_id, (in_d, out_d) in enumerate(dims):
                if self._stop_flag:
                    break
                engine.wait_layer(layer_id)
                engine.commit_layer()
                input_vec = engine.process_layer(input_vec, in_d, out_d)
                engine.retire_layer()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Simulate token extraction (argmax on output logits)
            token_char = chr(65 + (token_idx % 26))  # Placeholder
            generated_tokens.append(token_char)
            self.token_generated.emit(token_char)

        return "".join(generated_tokens)

    def _run_demo_inference(self) -> str:
        """
        استدلال تجريبي (Demo Mode) — يولّد نصاً يحاكي السلوك الحقيقي.

        يُستخدم عندما لا يتوفر libai_os_core أو لم يتم تحميل نموذج.
        يحاكي سرعة واقعية (~5-15 T/s) مع تأخير لكل توكن.
        """
        import random

        demo_responses = [
            "Hello! I am AIOS Core, a neural inference engine optimized for "
            "ultra-low-resource devices. I run on devices with as little as "
            "2 GB RAM and no GPU, thanks to ternary weight quantization "
            "(weights are -1, 0, or +1 — no multiplication needed!).",

            "My architecture consists of several key components working in "
            "harmony: the SovereignArena pins physical RAM to prevent swapping, "
            "the SlidingWindowManager implements a ring buffer for zero-allocation "
            "layer management, and the AsyncPrefetcher loads layers in the "
            "background for zero-latency inference.",

            "The Performance Governor lets you dynamically adjust between "
            "10 gear levels, from Eco Mode (300 MB RAM, no cache tiling) to "
            "Maximum Power (80% system RAM, large cache tiles). You can shift "
            "gears in real-time using the slider in the sidebar.",

            "I was built in 7 phases: C++17 kernel with platform detection, "
            "custom exceptions, SovereignArena memory management, "
            "SlidingWindowManager ring buffer, AsyncPrefetcher with zero-copy I/O, "
            "TernaryMathEngine with AVX2 SIMD, C API facade, Python engine bridge, "
            "model compiler with streaming architecture, OpenAI-compatible API "
            "server, premium PyQt6 GUI, and finally this integration layer.",
        ]

        # Pick a response based on prompt content hash
        response = demo_responses[hash(self._prompt) % len(demo_responses)]
        words = response.split(" ")

        generated_text_parts: List[str] = []

        for i, word in enumerate(words):
            if self._stop_flag:
                break

            # Simulate realistic token latency (50-150ms per token)
            latency = random.uniform(0.03, 0.12)
            time.sleep(latency)

            generated_text_parts.append(word)
            self.token_generated.emit(word + " ")

        return " ".join(generated_text_parts)


# =============================================================================
#  AppController — Central Orchestrator
# =============================================================================
#  Initializes the GUI, wires all signals/slots, manages QThread workers,
#  handles telemetry, and orchestrates the engine lifecycle.
# =============================================================================

class AppController:
    """
    وحدة التحكم الرئيسية — تجمع كل المكونات في تطبيق حيّ متكامل.

    Responsibilities:
      1. Create and show AIOSCoreGUI (dumb view from Phase 6).
      2. Initialize PerformanceGovernor + AIOCEngine (Phase 5).
      3. Wire all button signals to QThread workers.
      4. Run real-time telemetry via QTimer (1s interval).
      5. Handle gear shifting (Governor ↔ Slider).
      6. Manage QFileDialog for model browsing.
    """

    def __init__(self) -> None:
        self._app: Optional[QApplication] = None
        self._window: Optional[AIOSCoreGUI] = None

        # Engine & Governor
        self._engine: Optional[AIOCEngine] = None
        self._governor: Optional[PerformanceGovernor] = None
        self._current_gear: int = 5
        self._current_model_path: str = ""
        self._lib_path: str = ""

        # [FIX #2] متغير منفصل يحتفظ بالمسار الكامل للملف المختار للـ compile
        # بدل قراءة المسار من نص الزر (fragile وبيرجع اسم الملف مش المسار الكامل)
        self._pending_compile_path: str = ""

        # Workers
        self._compiler_worker: Optional[CompilerWorker] = None
        self._inference_worker: Optional[InferenceWorker] = None

        # Telemetry
        self._telemetry_timer: Optional[QTimer] = None
        self._token_count: int = 0
        self._inference_start_time: float = 0.0
        self._is_generating: bool = False

        # Chat history (for display)
        self._ai_response_buffer: str = ""

    # =====================================================================
    #  Initialization
    # =====================================================================

    def initialize(self) -> None:
        """
        تهيئة التطبيق بالكامل: GUI + Engine + Governor + Signal Wiring.

        This is the single entry point called from main().
        """
        # High-DPI support — MUST be set BEFORE creating QApplication
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        self._app = QApplication(sys.argv)
        self._app.setStyleSheet(DARK_THEME_QSS)

        # ── Create GUI (dumb view — we control everything) ──
        self._window = AIOSCoreGUI()

        # ── Disconnect GUI's internal mockup handlers ──
        # We replace them with real AppController handlers.
        self._window.send_btn.clicked.disconnect()
        self._window.stop_btn.clicked.disconnect()
        self._window.browse_btn.clicked.disconnect()
        self._window.compile_btn.clicked.disconnect()
        self._window.governor_slider.valueChanged.disconnect()

        # Stop mock telemetry timer — we run our own real one
        self._window._mock_telemetry_timer.stop()

        # ── Try to initialize engine ──
        self._try_init_engine()

        # ── Wire Signals ──
        self._wire_signals()

        # ── Start real telemetry ──
        self._start_telemetry()

        # ── Update version footer ──
        engine_status = " Connected" if self._engine else " Demo"
        self._window.setWindowTitle(
            f"AIOS Core — Neural Engine Interface [{engine_status} Mode]"
        )

        # ── Show ──
        self._window.show()

        # [FIX #5] ربط closeEvent بـ shutdown() عشان تتضمن الـ cleanup الكامل
        # لما المستخدم يضغط X مباشرة على النافذة.
        # بدون ده: InferenceWorker + VirtualLock + AIOCEngine بتفضل شغّالة بعد الإغلاق.
        _ctrl = self
        def _on_window_close(event):
            _ctrl.shutdown()
            event.accept()
        self._window.closeEvent = _on_window_close

        logger.info("AppController initialized successfully.")

    def _try_init_engine(self) -> None:
        """
        محاولة تهيئة المحرك (AIOCEngine) تلقائياً.

        يبحث عن libai_os_core في مسارات معروفة. إذا فشل، يعمل بوضع Demo.
        """
        if not _HAS_ENGINE:
            self._append_to_display(
                '<div style="color: #f9e2af; padding: 6px 0;">'
                '⚠ Engine bridge not available — running in <b>Demo Mode</b>.<br>'
                '<span style="color: rgba(205,214,244,0.4); font-size: 11px;">'
                'To enable real inference, compile the C++ kernel and place '
                'libai_os_core.so alongside this script.</span></div>'
            )
            return

        # Search for the shared library in common locations
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "build"),
            os.path.join(os.path.dirname(__file__), ".."),
            os.path.dirname(__file__),
            "/usr/local/lib",
            "/usr/lib",
        ]

        # [FIX #1] البحث الديناميكي عن الـ DLL بدل الـ Hardcoded path
        # الـ path القديم كان: r"C:\Users\Husse\Downloads\..." — محذوف
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _dll_candidates = [
            os.path.normpath(os.path.join(_script_dir, "..", "build", "Release", "libai_os_core.dll")),
            os.path.normpath(os.path.join(_script_dir, "..", "build", "libai_os_core.dll")),
            os.path.normpath(os.path.join(_script_dir, "..", "build", "libai_os_core.so")),
            os.path.join(_script_dir, "libai_os_core.dll"),
            os.path.join(_script_dir, "libai_os_core.so"),
        ]
        lib_name = next((p for p in _dll_candidates if os.path.isfile(p)), None)

        if not os.path.isfile(lib_name):
            self._append_to_display(
                '<div style="color: #f9e2af; padding: 6px 0;">'
                '⚠ libai_os_core not found — running in <b>Demo Mode</b>.<br>'
                '<span style="color: rgba(205,214,244,0.4); font-size: 11px;">'
                'Compile the C++ kernel (Phase 1) and place the library in '
                'the build/ directory.</span></div>'
            )
            return
        try:
            system_ram = detect_system_ram_mb()
            self._engine, self._governor = create_engine_with_governor(
                lib_path=lib_name,
                gear_level=self._current_gear,
                system_ram_mb=system_ram,
            )
            self._lib_path = lib_name

            info = self._engine.engine_info
            avx = "AVX2" if self._engine.has_avx2 else "Scalar"

            self._append_to_display(
                f'<div style="color: #a6e3a1; padding: 6px 0;">'
                f'✅ Engine initialized — {avx} mode | '
                f'{system_ram} MB RAM detected<br>'
                f'<span style="color: rgba(205,214,244,0.4); font-size: 11px;">'
                f'{info} | Lib: {os.path.basename(lib_name)}</span></div>'
            )

        except Exception as e:
            self._append_to_display(
                f'<div style="color: #f38ba8; padding: 6px 0;">'
                f'⚠ Engine init failed: {e}<br>'
                f'<span style="color: rgba(205,214,244,0.4); font-size: 11px;">'
                f'Falling back to Demo Mode.</span></div>'
            )
            self._engine = None
            self._governor = None

    # =====================================================================
    #  Signal Wiring — Connect GUI buttons to real handlers
    # =====================================================================

    def _wire_signals(self) -> None:
        """ربط جميع الإشارات بين الواجهة والتحكم."""

        # ── Browse → QFileDialog ──
        self._window.browse_btn.clicked.connect(self._on_browse)

        # ── Compile → CompilerWorker ──
        self._window.compile_btn.clicked.connect(self._on_compile)

        # ── Send → InferenceWorker ──
        self._window.send_btn.clicked.connect(self._on_send)

        # ── Stop → InferenceWorker.abort ──
        self._window.stop_btn.clicked.connect(self._on_stop)

        # ── Gear Slider → Governor ──
        self._window.governor_slider.valueChanged.connect(self._on_gear_changed)

    # =====================================================================
    #  Browse Handler — QFileDialog for model selection
    # =====================================================================

    def _on_browse(self) -> None:
        """
        فتح نافذة اختيار ملف النموذج الأصلي (.pt / .pth / .bin / .safetensors).
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Select Model File",
            "",
            "Model Files (*.pt *.pth *.bin *.safetensors *.gguf);;"
            "All Files (*)",
        )

        if not file_path:
            return

        file_name = os.path.basename(file_path)
        self._append_to_display(
            f'<div style="color: #89b4fa; font-size: 11px; padding: 4px 0;">'
            f'📁 Selected: <b>{self._escape_html(file_name)}</b> '
            f'({os.path.getsize(file_path) / (1024*1024):.1f} MB)</div>'
        )

        # Update browse button text (اسم الملف فقط للعرض — المسار الكامل محفوظ منفصل)
        self._window.browse_btn.setText(f"📁  {file_name}")

        # [FIX #2] احفظ المسار الكامل في متغير منفصل عشان _on_compile يستخدمه
        # بدل الطريقة القديمة اللي كانت تقرأ المسار من نص الزر وترجع اسم الملف مش المسار
        self._pending_compile_path = file_path

        # Update model combo if it's an AIOC file
        if file_path.endswith(".aioc"):
            self._current_model_path = file_path
            # Add to combo if not already there
            exists = False
            for i in range(self._window.model_combo.count()):
                if self._window.model_combo.itemText(i) == file_name:
                    self._window.model_combo.setCurrentIndex(i)
                    exists = True
                    break
            if not exists:
                self._window.model_combo.addItem(file_name)
                self._window.model_combo.setCurrentIndex(
                    self._window.model_combo.count() - 1
                )
        else:
            # Store for compilation
            self._current_model_path = ""

    # =====================================================================
    #  Compile Handler — CompilerWorker
    # =====================================================================

    def _on_compile(self) -> None:
        """
        بدء تحويل النموذج عبر CompilerWorker (QThread).
        """
        if not _HAS_COMPILER:
            QMessageBox.warning(
                self._window,
                "Compiler Unavailable",
                "ModelCompiler requires PyTorch and torch/safetensors.\n"
                "Install with: pip install torch safetensors",
            )
            return

        # [FIX #2] استخدم المسار الكامل المحفوظ في _pending_compile_path
        # بدل الطريقة القديمة اللي كانت تستخرج المسار من نص الزر —
        # الزر بيحتوي على اسم الملف فقط مش المسار الكامل، فالـ isfile كان يفشل دايماً
        input_path = self._pending_compile_path
        if not input_path or not os.path.isfile(input_path):
            QMessageBox.information(
                self._window,
                "No Model Selected",
                "Please select a model file first using the Browse button.",
            )
            return

        # Generate output path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"{base_name}.aioc")

        # Check if already compiling
        if (
            self._compiler_worker is not None
            and self._compiler_worker.isRunning()
        ):
            QMessageBox.information(
                self._window, "Busy",
                "Compilation is already in progress.",
            )
            return

        # Disable compile button during compilation
        self._window.compile_btn.setEnabled(False)
        self._window.compile_btn.setText("⏳ Compiling...")
        self._window.progress_bar.setValue(0)

        self._append_to_display(
            '<div style="color: #89b4fa; font-size: 11px; padding: 4px 0;">'
            f'⚡ Starting compilation: {self._escape_html(base_name)} → AIOC</div>'
        )

        # Create and start worker
        self._compiler_worker = CompilerWorker(
            input_path=input_path,
            output_path=output_path,
            threshold=0.05,
            parent=self._window,
        )
        self._compiler_worker.progress_signal.connect(self._on_compile_progress)
        self._compiler_worker.finished_signal.connect(self._on_compile_finished)
        self._compiler_worker.start()

    def _on_compile_progress(self, percent: int) -> None:
        """تحديث شريط التقدم أثناء التحويل (من الخيط الرئيسي)."""
        self._window.progress_bar.setValue(percent)

    def _on_compile_finished(self, success: bool, message: str) -> None:
        """معالجة نتيجة التحويل (من الخيط الرئيسي)."""
        self._window.compile_btn.setEnabled(True)
        self._window.compile_btn.setText("⚡  Compile to AIOC")

        if success:
            self._window.progress_bar.setValue(100)
            self._append_to_display(
                f'<div style="color: #a6e3a1; font-size: 11px; padding: 4px 0;">'
                f'✅ {self._escape_html(message)}</div>'
            )

            # [FIX #3] output_path = message كان غلط — message هي نص وصفي مش مسار ملف
            # الحل: اقرأ المسار مباشرة من الـ worker اللي عنده _output_path الحقيقي
            output_path = ""
            if self._compiler_worker is not None:
                output_path = getattr(self._compiler_worker, '_output_path', "")

            if output_path and os.path.isfile(output_path):
                file_label = os.path.basename(output_path)
                exists = False
                for i in range(self._window.model_combo.count()):
                    if self._window.model_combo.itemText(i) == file_label:
                        self._window.model_combo.setCurrentIndex(i)
                        exists = True
                        break
                if not exists:
                    self._window.model_combo.addItem(file_label)
                    self._window.model_combo.setCurrentIndex(
                        self._window.model_combo.count() - 1
                    )
                # احفظ مسار الـ AIOC المُنتج عشان نقدر نشغّله مباشرة
                self._current_model_path = output_path
        else:
            self._window.progress_bar.setValue(0)
            self._append_to_display(
                f'<div style="color: #f38ba8; font-size: 11px; padding: 4px 0;">'
                f'❌ {self._escape_html(message)}</div>'
            )

    # =====================================================================
    #  Send Handler — InferenceWorker
    # =====================================================================

    def _on_send(self) -> None:
        """
        إرسال رسالة المستخدم وبدء توليد الرد عبر InferenceWorker.
        """
        text = self._window.input_area.toPlainText().strip()
        if not text:
            return

        # Check if already generating
        if (
            self._inference_worker is not None
            and self._inference_worker.isRunning()
        ):
            return

        # Display user message
        self._append_to_display(
            f'<div style="margin: 16px 0 6px 0;">'
            f'<span style="color: #89b4fa; font-weight: 600; font-size: 12px;">'
            f'YOU</span></div>'
            f'<div style="color: #cdd6f4; padding-left: 12px; line-height: 1.6;">'
            f'{self._escape_html(text)}</div>'
        )

        # Clear input
        self._window.input_area.clear()
        self._window.input_area.setFocus()

        # Start AI response header
        self._append_to_display(
            '<div style="margin: 20px 0 6px 0;">'
            '<span style="color: #cba6f7; font-weight: 600; font-size: 12px;">'
            'AIOS</span></div>'
            '<div id="active_response" style="color: #cdd6f4; padding-left: 12px; '
            'line-height: 1.6;"></div>'
        )

        # Reset streaming buffer
        self._ai_response_buffer = ""
        self._token_count = 0
        self._inference_start_time = time.perf_counter()
        self._is_generating = True

        # Update UI state
        self._window.send_btn.setEnabled(False)
        self._window.send_btn.setText("⏳ Generating...")

        # [FIX #4] layer_dims كانت فارغة دايماً [] → الشرط في InferenceWorker.run()
        # كان يفشل دايماً ويروح للـ Demo Mode حتى لو المحرك شغّال.
        # الحل: ابنِ layer_dims من عدد الطبقات الفعلية في النموذج المحمّل.
        # لو النموذج محمّل، ابعت list غير فارغة عشان يدخل الـ real inference path.
        # الـ dims الحقيقية مش متاحة دلوقتي في الـ C API (بيتحسن في مرحلة لاحقة)،
        # فبنستخدم placeholder (768, 768) كعدد الطبقات — الـ process_layer هيقرأ
        # الأبعاد الفعلية من الـ AIOC weights مباشرة.
        _layer_dims: List[Tuple[int, int]] = []
        if self._engine and self._engine.is_initialized:
            _n_layers = self._engine.total_layers
            if _n_layers > 0:
                _layer_dims = [(768, 768)] * _n_layers  # placeholder dims

        # Start inference worker
        self._inference_worker = InferenceWorker(
            engine=self._engine,
            prompt=text,
            layer_dims=_layer_dims,
            max_tokens=256,
            parent=self._window,
        )
        self._inference_worker.token_generated.connect(self._on_token_streamed)
        self._inference_worker.finished_signal.connect(self._on_inference_finished)
        self._inference_worker.start()

    def _on_token_streamed(self, token: str) -> None:
        """
        استقبال توكن جديد وعرضه في الشاشة (Streaming).
        يُستدعى من خيط InferenceWorker عبر pyqtSignal — آمن للـ GUI.
        """
        self._ai_response_buffer += token
        self._token_count += 1

        # Append to display area using cursor
        cursor = self._window.display_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)

        # Auto-scroll
        scrollbar = self._window.display_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_inference_finished(self, success: bool, message: str) -> None:
        """
        معالجة انتهاء التوليد (من الخيط الرئيسي).
        """
        self._is_generating = False

        # Restore UI state
        self._window.send_btn.setEnabled(True)
        self._window.send_btn.setText("Send  ➤")

        elapsed = time.perf_counter() - self._inference_start_time
        tokens_per_sec = (
            self._token_count / elapsed if elapsed > 0 else 0
        )

        if success:
            # Add newline after response
            self._append_to_display("")
        else:
            self._append_to_display(
                f'<div style="color: #f38ba8; font-size: 11px; padding: 4px 12px;">'
                f'⚠ {self._escape_html(message)}</div>'
            )

        # Update speed card with real metrics
        self._window.card_speed.set_value(f"{tokens_per_sec:.1f} T/s")

    # =====================================================================
    #  Stop Handler
    # =====================================================================

    def _on_stop(self) -> None:
        """
        إيقاف التوليد الحالي عبر إنهاء InferenceWorker.
        """
        if (
            self._inference_worker is not None
            and self._inference_worker.isRunning()
        ):
            self._inference_worker.request_stop()
            self._append_to_display(
                '<div style="color: #f38ba8; font-size: 11px; padding: 4px 0;">'
                '⏹ Generation stopped by user.</div>'
            )

        if (
            self._compiler_worker is not None
            and self._compiler_worker.isRunning()
        ):
            self._compiler_worker.abort()
            self._append_to_display(
                '<div style="color: #f38ba8; font-size: 11px; padding: 4px 0;">'
                '⏹ Compilation aborted by user.</div>'
            )

    # =====================================================================
    #  Gear Shifting — Dynamic Governor Control
    # =====================================================================

    def _on_gear_changed(self, value: int) -> None:
        """
        تغيير الترس أثناء التشغيل.

        إذا كان المحرك متاحاً، يعيد تهيئته بالإعدادات الجديدة.
        إذا لم يكن متاحاً، يحدث التسمية فقط.
        """
        self._current_gear = value

        # Update label in GUI (GUI's internal handler is disconnected,
        # so we need to update it ourselves)
        gear_names = {
            1: "Eco Mode", 2: "Ultra Light", 3: "Light", 4: "Moderate",
            5: "Balanced", 6: "Performance", 7: "High Power",
            8: "Aggressive", 9: "Overdrive", 10: "Maximum Power",
        }
        name = gear_names.get(value, "")
        self._window.gear_label.setText(f"⚙️  Gear {value}: {name}")

        if self._governor is not None:
            config = self._governor.get_config(value)
            self._append_to_display(
                f'<div style="color: rgba(205,214,244,0.4); font-size: 10px; '
                f'padding: 2px 0;">'
                f'⚙ Gear {value}: {config["description"]} '
                f'({config["ram_budget_mb"]} MB RAM, '
                f'block={config["block_size"]})</div>'
            )

    # =====================================================================
    #  Telemetry — Real-time System Monitoring (QTimer, 1s)
    # =====================================================================

    def _start_telemetry(self) -> None:
        """بدء مؤقت المراقبة الحية — يقرأ RAM/CPU كل ثانية."""
        parent = self._window if isinstance(self._window, QObject) else None
        self._telemetry_timer = QTimer(parent)
        self._telemetry_timer.timeout.connect(self._update_telemetry)
        self._telemetry_timer.start(1000)  # 1 second

    def _update_telemetry(self) -> None:
        """
        قراءة واستخدام بيانات النظام الحية.

        RAM: psutil.Process().memory_info().rss
        CPU: psutil.cpu_percent()
        Speed: calculated from token generation rate
        """
        # ── RAM ──
        if _HAS_PSUTIL:
            try:
                process = psutil.Process()
                rss_bytes = process.memory_info().rss
                rss_mb = rss_bytes / (1024 * 1024)
                self._window.card_ram.set_value(f"{rss_mb:.0f} MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._window.card_ram.set_value("— MB")
        else:
            # Fallback: read from /proc/self/status
            try:
                with open("/proc/self/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            self._window.card_ram.set_value(f"{kb // 1024} MB")
                            break
            except Exception:
                self._window.card_ram.set_value("— MB")

        # ── CPU ──
        if _HAS_PSUTIL:
            try:
                cpu_pct = psutil.cpu_percent(interval=0)
                self._window.card_cpu.set_value(f"{cpu_pct:.0f} %")
            except Exception:
                self._window.card_cpu.set_value("— %")
        else:
            self._window.card_cpu.set_value("— %")

        # ── Speed (tokens/sec) — only meaningful during generation ──
        if self._is_generating and self._token_count > 0:
            elapsed = time.perf_counter() - self._inference_start_time
            if elapsed > 0:
                tps = self._token_count / elapsed
                self._window.card_speed.set_value(f"{tps:.1f} T/s")
        elif not self._is_generating:
            # Show arena utilization if engine is available
            if self._engine and self._engine.is_initialized:
                util = self._engine.arena_utilization
                if util >= 0:
                    self._window.card_speed.set_value(
                        f"Arena {util:.0f}%"
                    )

    # =====================================================================
    #  Cleanup
    # =====================================================================

    def shutdown(self) -> None:
        """إيقاف جميع العمال والمحرك بأمان."""
        logger.info("Shutting down AppController...")

        # Stop inference
        if (
            self._inference_worker is not None
            and self._inference_worker.isRunning()
        ):
            self._inference_worker.request_stop()
            self._inference_worker.wait(3000)  # 3s timeout

        # Stop compiler
        if (
            self._compiler_worker is not None
            and self._compiler_worker.isRunning()
        ):
            self._compiler_worker.abort()
            self._compiler_worker.wait(3000)

        # Stop telemetry
        if self._telemetry_timer is not None:
            self._telemetry_timer.stop()

        # Teardown engine
        if self._engine is not None:
            try:
                self._engine.teardown()
            except Exception:
                pass

        logger.info("AppController shut down cleanly.")

    # =====================================================================
    #  Utilities
    # =====================================================================

    def _append_to_display(self, html: str) -> None:
        """إضافة HTML إلى منطقة العرض مع التمرير التلقائي."""
        if self._window is None:
            return
        self._window.display_area.append(html)
        scrollbar = self._window.display_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _escape_html(text: str) -> str:
        """هروب النص لعرضه بأمان في HTML."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
            .replace('"', "&quot;")
        )

    # =====================================================================
    #  Run Loop
    # =====================================================================

    def run(self) -> int:
        """
        تشغيل حلقة الأحداث الرئيسية (Event Loop).

        Returns:
            Exit code from QApplication.
        """
        if self._app is None:
            raise RuntimeError("AppController not initialized. Call initialize() first.")

        exit_code = self._app.exec()
        self.shutdown()
        return exit_code


# =============================================================================
#  Entry Point — This is the ONLY way to launch the application
# =============================================================================
#  Usage: python app.py
# =============================================================================

def main() -> int:
    """
    نقطة الدخول الرئيسية للمشروع.

    Creates AppController, initializes all subsystems, and enters the
    Qt event loop. Returns the application exit code.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    controller = AppController()
    controller.initialize()
    return controller.run()


if __name__ == "__main__":
    sys.exit(main())
