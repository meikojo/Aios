"""
AI OS Core — Phase 5: Engine Bridge & Performance Governor
============================================================

This module provides two components:

1. AIOCEngine
   A Pythonic wrapper around the AI OS Core C library (libai_os_core).
   Uses ctypes to call the extern "C" functions declared in c_api.h.
   Every C error code is translated to a Python RuntimeError with a
   descriptive message — the caller never sees raw integer codes.

2. PerformanceGovernor
   Translates a user-facing "Gear Level" (1–10) into concrete hardware
   settings for the AI OS Core:
     - ram_budget_bytes: how much physical RAM to lock
     - block_size: TernaryMathEngine tiling parameter

   Gear 1  (Ultra Conservative): 300 MB RAM, no blocking — for weakest devices
   Gear 5  (Balanced):           min(1 GB, 30% system), default blocking
   Gear 10 (Maximum Greed):     80% system RAM, max blocking — for powerful boxes

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  Python Application                                          │
  │  governor.get_config(5) → engine.init_system(ram_budget)     │
  │  engine.load_model("model.aioc")                             │
  │  for layer in range(engine.total_layers):                    │
  │      engine.wait_layer(layer)                                │
  │      engine.commit_layer()                                   │
  │      output = engine.process_layer(input, in_dim, out_dim)   │
  │      engine.retire_layer()                                   │
  └──────────────────────┬───────────────────────────────────────┘
                         │  ctypes FFI
  ┌──────────────────────▼───────────────────────────────────────┐
  │  libai_os_core.so / ai_os_core.dll                           │
  │  c_api.cpp — extern "C" facade                               │
  └──────────────────────────────────────────────────────────────┘

Dependencies:
  - ctypes (stdlib)
  - numpy (for input/output array handling)
  - platform, os, sys, struct (stdlib)

Author: AI OS Core Team
Version: 1.0.0
Since: 2026-04-08
License: MIT
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# psutil for reliable cross-platform RAM detection (preferred method).
# Falls back to platform-specific /proc/sysctl/ctypes methods if not installed.
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# =============================================================================
#  C API Error Code Constants (must match c_api.h exactly)
# =============================================================================

AIOC_OK: int = 0
AIOC_ERR_NULL_CONTEXT: int = -1
AIOC_ERR_NULL_ARG: int = -2
AIOC_ERR_INVALID_ARG: int = -3
AIOC_ERR_ALLOCATION: int = -10
AIOC_ERR_LOCK: int = -11
AIOC_ERR_CAPACITY: int = -12
AIOC_ERR_INVALID_LAYER: int = -13
AIOC_ERR_WINDOW_OVERFLOW: int = -14
AIOC_ERR_IO: int = -20
AIOC_ERR_PREFETCHER: int = -21
AIOC_ERR_NOT_STARTED: int = -22
AIOC_ERR_NOT_LOADED: int = -23
AIOC_ERR_ENGINE: int = -30
AIOC_ERR_NO_ACTIVE_LAYER: int = -31
AIOC_ERR_UNKNOWN: int = -99

_ERROR_DESCRIPTIONS: Dict[int, str] = {
    AIOC_OK:                   "OK",
    AIOC_ERR_NULL_CONTEXT:     "Null context pointer",
    AIOC_ERR_NULL_ARG:         "Null function argument",
    AIOC_ERR_INVALID_ARG:      "Invalid argument value",
    AIOC_ERR_ALLOCATION:       "OS memory allocation failed",
    AIOC_ERR_LOCK:             "OS memory lock failed (insufficient privileges?)",
    AIOC_ERR_CAPACITY:         "Arena capacity exhausted",
    AIOC_ERR_INVALID_LAYER:    "Invalid layer ID or size",
    AIOC_ERR_WINDOW_OVERFLOW:  "Sliding window overflow (ring buffer full)",
    AIOC_ERR_IO:               "File I/O error",
    AIOC_ERR_PREFETCHER:       "Async prefetcher error",
    AIOC_ERR_NOT_STARTED:      "Prefetcher not started",
    AIOC_ERR_NOT_LOADED:       "Model not loaded",
    AIOC_ERR_ENGINE:           "Math engine computation error",
    AIOC_ERR_NO_ACTIVE_LAYER:  "No active layer (call commit_layer first)",
    AIOC_ERR_UNKNOWN:          "Unknown / unclassified error",
}


# =============================================================================
#  AIOCEngine — ctypes Bridge to the C Library
# =============================================================================

class AIOCEngine:
    """
    Pythonic wrapper around the AI OS Core shared library.

    Loads libai_os_core via ctypes, defines all function prototypes
    (argtypes / restype), and exposes a clean Python API that translates
    C error codes into Python RuntimeError exceptions.

    Usage:
        engine = AIOCEngine("/path/to/libai_os_core.so")
        engine.init_system(ram_budget_bytes=300 * 1024 * 1024)
        engine.load_model("model.aioc")

        for layer_id in range(engine.total_layers):
            engine.wait_layer(layer_id)
            engine.commit_layer()
            output = engine.process_layer(input_vec, input_dim, output_dim)
            engine.retire_layer()

        engine.teardown()
    """

    def __init__(self, lib_path: str) -> None:
        """
        Load the AI OS Core shared library.

        Args:
            lib_path: Absolute or relative path to the shared library file.
                      Examples:
                        - Linux:   "./libai_os_core.so"
                        - macOS:   "./libai_os_core.dylib"
                        - Windows: ".\\ai_os_core.dll"

        Raises:
            OSError: If the library cannot be found or loaded.
        """
        if not lib_path:
            raise ValueError("AIOCEngine: lib_path must not be empty.")

        # [FIX #8] Use add_dll_directory on Windows (Python 3.8+) to bypass
        # the restricted DLL search path. Without this, ctypes.CDLL fails
        # with "module not found" even if the file exists on disk.
        if hasattr(os, "add_dll_directory"):
            lib_dir = os.path.dirname(os.path.abspath(lib_path))
            # Temporarily add the DLL's directory to the search path
            os.add_dll_directory(lib_dir)

        try:
            if not os.path.isfile(lib_path):
                # Try to resolve from the system's library search paths
                try:
                    self._lib = ctypes.CDLL(lib_path)
                except OSError:
                    raise OSError(
                        f"AIOCEngine: cannot find shared library '{lib_path}'. "
                        f"Ensure the file exists and its directory is in "
                        f"LD_LIBRARY_PATH (Linux), DYLD_LIBRARY_PATH (macOS), "
                        f"or PATH (Windows)."
                    )
            else:
                self._lib = ctypes.CDLL(lib_path)
        except Exception:
            # Clean up add_dll_directory on failure
            if hasattr(os, "remove_dll_directory"):
                try:
                    os.remove_dll_directory(os.path.dirname(os.path.abspath(lib_path)))
                except Exception:
                    pass
            raise

        self._ctx: Optional[ctypes.c_void_p] = None
        self._define_prototypes()

    def _define_prototypes(self) -> None:
        """
        Set argtypes and restype for every C API function.

        This is CRITICAL for two reasons:
          1. It enables ctypes to marshal Python types → C types correctly.
          2. It enables ctypes to catch type mismatches at the boundary
             instead of corrupting memory silently.
        """
        lib = self._lib

        # --- Lifecycle ---------------------------------------------------

        # void* aioc_init_system(size_t ram_budget_bytes)
        lib.aioc_init_system.argtypes = [ctypes.c_size_t]
        lib.aioc_init_system.restype = ctypes.c_void_p

        # void* aioc_load_model(void* context, const char* file_path)
        lib.aioc_load_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.aioc_load_model.restype = ctypes.c_void_p

        # void aioc_teardown(void* context)
        lib.aioc_teardown.argtypes = [ctypes.c_void_p]
        lib.aioc_teardown.restype = None

        # --- Inference Pipeline ------------------------------------------

        # int aioc_wait_layer(void* context, int32_t layer_id)
        lib.aioc_wait_layer.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.aioc_wait_layer.restype = ctypes.c_int

        # int aioc_commit_layer(void* context)
        lib.aioc_commit_layer.argtypes = [ctypes.c_void_p]
        lib.aioc_commit_layer.restype = ctypes.c_int

        # int aioc_process_layer(void* context, const float* inputs,
        #                         float* outputs, uint32_t input_dim,
        #                         uint32_t output_dim)
        lib.aioc_process_layer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.aioc_process_layer.restype = ctypes.c_int

        # int aioc_retire_layer(void* context)
        lib.aioc_retire_layer.argtypes = [ctypes.c_void_p]
        lib.aioc_retire_layer.restype = ctypes.c_int

        # --- Query ------------------------------------------------------

        # uint32_t aioc_total_layers(void* context)
        lib.aioc_total_layers.argtypes = [ctypes.c_void_p]
        lib.aioc_total_layers.restype = ctypes.c_uint32

        # uint64_t aioc_layer_size(void* context, uint32_t layer_id)
        lib.aioc_layer_size.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.aioc_layer_size.restype = ctypes.c_uint64

        # const char* aioc_get_last_error(void* context)
        lib.aioc_get_last_error.argtypes = [ctypes.c_void_p]
        lib.aioc_get_last_error.restype = ctypes.c_char_p

        # const char* aioc_engine_info(void* context)
        lib.aioc_engine_info.argtypes = [ctypes.c_void_p]
        lib.aioc_engine_info.restype = ctypes.c_char_p

        # int aioc_has_avx2(void* context)
        lib.aioc_has_avx2.argtypes = [ctypes.c_void_p]
        lib.aioc_has_avx2.restype = ctypes.c_int

        # --- Engine Configuration ---------------------------------------

        # int aioc_set_block_size(void* context, uint32_t block_size)
        lib.aioc_set_block_size.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.aioc_set_block_size.restype = ctypes.c_int

        # uint32_t aioc_get_block_size(void* context)
        lib.aioc_get_block_size.argtypes = [ctypes.c_void_p]
        lib.aioc_get_block_size.restype = ctypes.c_uint32

        # double aioc_arena_utilization(void* context)
        lib.aioc_arena_utilization.argtypes = [ctypes.c_void_p]
        lib.aioc_arena_utilization.restype = ctypes.c_double

        # --- Pipeline Control ------------------------------------------

        # const char* aioc_get_model_path(void* context)
        lib.aioc_get_model_path.argtypes = [ctypes.c_void_p]
        lib.aioc_get_model_path.restype = ctypes.c_char_p

        # int aioc_reset_pipeline(void* context)
        lib.aioc_reset_pipeline.argtypes = [ctypes.c_void_p]
        lib.aioc_reset_pipeline.restype = ctypes.c_int

        # int aioc_reload_model(void* context)
        lib.aioc_reload_model.argtypes = [ctypes.c_void_p]
        lib.aioc_reload_model.restype = ctypes.c_int

    # =====================================================================
    #  Error Handling
    # =====================================================================

    def _check(self, error_code: int) -> None:
        """
        Raise RuntimeError if the C API returned a non-zero error code.

        The error message combines the human-readable code description
        with the detailed C++ error string (via aioc_get_last_error).
        """
        if error_code == AIOC_OK:
            return

        description = _ERROR_DESCRIPTIONS.get(
            error_code, f"Unknown error code {error_code}"
        )
        detail = self.last_error
        if detail:
            msg = f"[AIOC {error_code}] {description}: {detail}"
        else:
            msg = f"[AIOC {error_code}] {description}"

        raise RuntimeError(msg)

    @property
    def last_error(self) -> str:
        """
        Retrieve the last error message from the C library.

        Works both before init (returns thread-local init error)
        and after init (returns context-specific error).

        Returns:
            Human-readable error string, or empty string if no error.
        """
        if self._ctx:
            raw = self._lib.aioc_get_last_error(self._ctx)
        else:
            raw = self._lib.aioc_get_last_error(ctypes.c_void_p(0))

        if raw:
            return raw.decode("utf-8", errors="replace")
        return ""

    # =====================================================================
    #  Lifecycle
    # =====================================================================

    def init_system(self, ram_budget_bytes: int) -> None:
        """
        Initialize the AI OS Core inference system.

        Creates the SovereignArena (locked RAM) and SlidingWindowManager
        (ring buffer) with the specified memory budget. Also initializes
        the TernaryMathEngine with AVX2 auto-detection.

        Args:
            ram_budget_bytes: Amount of physical RAM to lock, in bytes.
                              Must be > 0. Will be page-aligned internally.

        Raises:
            RuntimeError: If initialization fails (allocation, lock, etc.).
            ValueError:   If ram_budget_bytes <= 0.

        Note:
            On Linux, you may need: sudo setcap cap_ipc_lock+ep <program>
            Or run with: ulimit -l unlimited
        """
        if ram_budget_bytes <= 0:
            raise ValueError(
                f"AIOCEngine.init_system: ram_budget_bytes must be > 0, "
                f"got {ram_budget_bytes}."
            )

        ctx_ptr = self._lib.aioc_init_system(ctypes.c_size_t(ram_budget_bytes))
        if not ctx_ptr:
            raise RuntimeError(
                f"AIOCEngine.init_system failed: {self.last_error}"
            )

        self._ctx = ctx_ptr

    def load_model(self, file_path: str) -> None:
        """
        Load an AIOC model file and start background prefetching.

        Args:
            file_path: Path to the .aioc model file (created by ModelCompiler).

        Raises:
            RuntimeError: If the model cannot be loaded (invalid file, etc.).
            FileNotFoundError: If file_path does not exist.
            ValueError: If file_path is empty.
        """
        if not file_path:
            raise ValueError("AIOCEngine.load_model: file_path must not be empty.")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"AIOCEngine.load_model: file not found: '{file_path}'"
            )
        if not self._ctx:
            raise RuntimeError(
                "AIOCEngine.load_model: system not initialized. "
                "Call init_system() first."
            )

        encoded = file_path.encode("utf-8")
        result = self._lib.aioc_load_model(self._ctx, encoded)
        if not result:
            raise RuntimeError(
                f"AIOCEngine.load_model failed: {self.last_error}"
            )

    def teardown(self) -> None:
        """
        Shut down the system and release all resources.

        Stops the background prefetch thread, closes the model file,
        unlocks and frees the physical RAM, and destroys the context.

        Safe to call multiple times. After teardown, the engine is
        unusable until init_system() is called again.
        """
        if self._ctx:
            self._lib.aioc_teardown(self._ctx)
            self._ctx = None
            self._model_path_cache = ""

    # =====================================================================
    #  Pipeline Control
    # =====================================================================

    def reset_pipeline(self) -> None:
        """
        Reset the inference pipeline for a new forward pass.

        Stops the background prefetch thread, recreates the
        SlidingWindowManager over the same locked RAM region, and
        prepares the engine for a new load_model() call.

        The SovereignArena (locked RAM) is NOT destroyed — it is
        reused for the next forward pass. Only the ring buffer slots
        and the AsyncPrefetcher are reset.

        This is required for autoregressive token generation: after
        processing all layers for one token, call reset_pipeline()
        before reload_model() or load_model() for the next token.

        Raises:
            RuntimeError: If the reset fails.
        """
        if not self._ctx:
            raise RuntimeError(
                "AIOCEngine.reset_pipeline: system not initialized."
            )
        self._check(self._lib.aioc_reset_pipeline(self._ctx))

    def reload_model(self) -> None:
        """
        Reload the previously loaded model file (one-shot convenience).

        Equivalent to calling reset_pipeline() + load_model() in sequence,
        but uses the internally stored model path so the caller doesn't
        need to pass it again.

        This is the recommended pattern for autoregressive token generation:
            engine.load_model("model.aioc")        # First load (from disk)
            for layer_id in range(n):
                engine.wait_layer(layer_id)
                engine.commit_layer()
                output = engine.process_layer(x, in_d, out_d)
                engine.retire_layer()
            engine.reload_model()                 # Reload from OS page cache
            for layer_id in range(n):             # Process next token
                ...

        Raises:
            RuntimeError: If no model was previously loaded or reload fails.
        """
        if not self._ctx:
            raise RuntimeError(
                "AIOCEngine.reload_model: system not initialized."
            )
        self._check(self._lib.aioc_reload_model(self._ctx))

    @property
    def model_path(self) -> str:
        """
        Path of the currently loaded AIOC model file.

        Returns empty string if no model is loaded.
        """
        if not self._ctx:
            return ""
        raw = self._lib.aioc_get_model_path(self._ctx)
        if raw:
            return raw.decode("utf-8", errors="replace")
        return ""

    # =====================================================================
    #  Inference Pipeline
    # =====================================================================

    def wait_layer(self, layer_id: int) -> None:
        """
        Block until the specified layer has been loaded into RAM.

        If the layer has already been fetched by the background thread,
        this returns immediately (zero latency). Otherwise, the calling
        thread is suspended until the I/O thread finishes loading.

        Args:
            layer_id: 0-indexed layer identifier.

        Raises:
            RuntimeError: If the wait fails or the prefetcher has stopped.
        """
        self._check(
            self._lib.aioc_wait_layer(self._ctx, ctypes.c_int32(layer_id))
        )

    def commit_layer(self) -> None:
        """
        Commit the most recently fetched layer as the active layer.

        After this call, the layer's weight data is available for
        computation via process_layer().

        Raises:
            RuntimeError: If no layer is in the Preparing state.
        """
        self._check(self._lib.aioc_commit_layer(self._ctx))

    def process_layer(
        self,
        inputs: np.ndarray,
        input_dim: int,
        output_dim: int,
    ) -> np.ndarray:
        """
        Process the active layer: Y = X ⊛ W (multiply-free ternary matmul).

        Reads ternary weights (-1, 0, +1) from the active layer in the
        SlidingWindowManager and computes the output using only ADD/SUB
        operations via the TernaryMathEngine.

        Args:
            inputs:    Input activation vector as a numpy array.
                       Will be converted to float32 if needed.
            input_dim: Number of input features (must match inputs.shape[0]).
            output_dim: Number of output features.

        Returns:
            numpy array of shape (output_dim,) with dtype float32.

        Raises:
            RuntimeError: If no active layer or computation fails.
            ValueError:   If dimensions don't match the input array.

        Note:
            The weights are read directly from the pinned RAM ring buffer
            (zero-copy). No data is copied or allocated during computation.
        """
        # Validate and prepare inputs
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs, dtype=np.float32)
        else:
            inputs = np.ascontiguousarray(inputs, dtype=np.float32).ravel()

        if inputs.shape[0] != input_dim:
            raise ValueError(
                f"process_layer: input array has {inputs.shape[0]} elements, "
                f"but input_dim={input_dim}."
            )

        # Allocate output buffer
        outputs = np.zeros(output_dim, dtype=np.float32)

        # Get raw pointers for ctypes
        inputs_ptr = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        outputs_ptr = outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._check(
            self._lib.aioc_process_layer(
                self._ctx,
                inputs_ptr,
                outputs_ptr,
                ctypes.c_uint32(input_dim),
                ctypes.c_uint32(output_dim),
            )
        )

        return outputs

    def retire_layer(self) -> None:
        """
        Retire the active layer and signal the background thread.

        The active layer's memory is marked as reusable in the ring buffer,
        and the background I/O thread is notified to begin loading the
        next layer.

        Raises:
            RuntimeError: If no layer is active.
        """
        self._check(self._lib.aioc_retire_layer(self._ctx))

    # =====================================================================
    #  Convenience: Full Inference Loop
    # =====================================================================

    def run_inference(
        self,
        input_activations: np.ndarray,
        layer_dims: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Run inference through all layers sequentially.

        This is a convenience method that executes the full pipeline:
          wait → commit → process → retire  for each layer.

        Args:
            input_activations: Initial input vector (float32 numpy array).
            layer_dims: List of (input_dim, output_dim) tuples, one per
                        layer in the model. The order must match the AIOC
                        file's layer order.

        Returns:
            Final output activations after the last layer (float32 numpy array).

        Raises:
            RuntimeError: If any pipeline step fails.
            ValueError:   If layer_dims is empty or dimensions are invalid.

        Example:
            dims = [(768, 768), (768, 3072), (3072, 768)]  # 3 layers
            result = engine.run_inference(input_vec, dims)
        """
        if not layer_dims:
            raise ValueError("run_inference: layer_dims must not be empty.")

        current = np.ascontiguousarray(input_activations, dtype=np.float32).ravel()

        for layer_id, (in_dim, out_dim) in enumerate(layer_dims):
            self.wait_layer(layer_id)
            self.commit_layer()
            current = self.process_layer(current, in_dim, out_dim)
            self.retire_layer()

        return current

    # =====================================================================
    #  Query Properties
    # =====================================================================

    @property
    def total_layers(self) -> int:
        """Total number of layers in the loaded model (0 if not loaded)."""
        if not self._ctx:
            return 0
        return int(self._lib.aioc_total_layers(self._ctx))

    @property
    def is_initialized(self) -> bool:
        """True if init_system() has been called successfully."""
        return self._ctx is not None

    @property
    def has_avx2(self) -> bool:
        """True if AVX2 SIMD instructions are available on the CPU."""
        if not self._ctx:
            return False
        return self._lib.aioc_has_avx2(self._ctx) == 1

    @property
    def engine_info(self) -> str:
        """Human-readable description of the math engine capabilities."""
        if not self._ctx:
            return ""
        raw = self._lib.aioc_engine_info(self._ctx)
        if raw:
            return raw.decode("utf-8", errors="replace")
        return ""

    @property
    def block_size(self) -> int:
        """Current output-dimension block size for tiled computation."""
        if not self._ctx:
            return 0
        return int(self._lib.aioc_get_block_size(self._ctx))

    @block_size.setter
    def block_size(self, value: int) -> None:
        """Set the output-dimension block size for tiled computation."""
        if not self._ctx:
            raise RuntimeError("AIOCEngine: system not initialized.")
        if value < 0:
            raise ValueError(f"block_size must be >= 0, got {value}.")
        self._check(
            self._lib.aioc_set_block_size(self._ctx, ctypes.c_uint32(value))
        )

    @property
    def arena_utilization(self) -> float:
        """SovereignArena memory utilization as a percentage (0.0–100.0)."""
        if not self._ctx:
            return -1.0
        return float(self._lib.aioc_arena_utilization(self._ctx))

    def layer_size(self, layer_id: int) -> int:
        """
        Get the byte size of a specific layer in the loaded model.

        Args:
            layer_id: 0-indexed layer identifier.

        Returns:
            Layer size in bytes, or 0 if the model is not loaded.
        """
        if not self._ctx:
            return 0
        return int(self._lib.aioc_layer_size(self._ctx, ctypes.c_uint32(layer_id)))

    # =====================================================================
    #  Context Manager Support
    # =====================================================================

    def __enter__(self) -> "AIOCEngine":
        return self

    def __exit__(self, exc_type, object, exc_tb) -> None:
        self.teardown()

    def __del__(self) -> None:
        try:
            self.teardown()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "initialized" if self._ctx else "not initialized"
        return f"AIOCEngine(lib_path='{self._lib._name}', {status})"


# =============================================================================
#  PerformanceGovernor — Gear Level → Hardware Configuration
# =============================================================================

class PerformanceGovernor:
    """
    Translates a user-facing "Gear Level" (1–10) into concrete hardware
    settings for the AI OS Core inference engine.

    The governor encapsulates the mapping between what the user wants
    ("run faster" / "use less RAM") and what the hardware actually needs
    (ram_budget_bytes, block_size, etc.).

    Gear Levels:
        Gear 1  — Ultra Conservative (300 MB, no blocking)
            For the weakest devices: old phones, Raspberry Pi Zero,
            2GB Chromebooks. Disables cache tiling to minimize memory
            overhead. Only 300 MB of RAM is locked.

        Gear 3  — Conservative (500 MB, small blocking)
            For budget devices with 2–4 GB RAM. Enables small cache
            tiles for modest speed improvements.

        Gear 5  — Balanced (min(1 GB, 30% system), default blocking)
            The recommended default for most devices. Allocates a
            reasonable amount of RAM and uses the default block size
            that balances cache efficiency with memory usage.

        Gear 7  — Performance (50% system RAM, large blocking)
            For mid-range devices with 4–8 GB RAM. Aggressively
            locks RAM and uses large cache tiles for maximum
            throughput.

        Gear 10 — Maximum Greed (80% system RAM, maximum blocking)
            For powerful devices with 8+ GB RAM. Locks 80% of
            available RAM and uses the largest possible cache tiles.
            WARNING: may cause system instability on low-RAM devices.

    Usage:
        governor = PerformanceGovernor(system_total_ram_mb=4096)

        # Get recommended settings for gear 5 (balanced)
        config = governor.get_config(gear_level=5)
        # config = {
        #     "gear_level": 5,
        #     "ram_budget_bytes": 1073741824,
        #     "ram_budget_mb": 1024.0,
        #     "block_size": 4096,
        #     "prefetch_ahead": 1,
        #     "description": "Balanced — stable performance with moderate RAM usage",
        # }

        # Apply to engine
        engine.init_system(ram_budget_bytes=config["ram_budget_bytes"])
        engine.block_size = config["block_size"]

        # Or use the convenience method:
        governor.apply_config(engine, gear_level=5)
    """

    # -------------------------------------------------------------------------
    #  Gear Level Anchors
    #
    #  Each anchor defines the exact settings for that gear level.
    #  Intermediate levels are linearly interpolated between anchors.
    # -------------------------------------------------------------------------

    # (gear_level, ram_budget_mb, block_size, description)
    _GEAR_ANCHORS: List[Tuple[int, int, int, str]] = [
        (1,  300,    0,     "Ultra Conservative — minimal RAM (300 MB), no cache tiling"),
        (2,  400,    0,     "Very Conservative — 400 MB, no cache tiling"),
        (3,  512,    2048,  "Conservative — 512 MB, small cache tiles"),
        (4,  768,    4096,  "Moderate — 768 MB, standard cache tiles"),
        (5,  1024,   4096,  "Balanced — stable performance with moderate RAM usage"),
        (6,  0,      8192,  "Performance-Trending — 40% system RAM, large tiles"),   # RAM = 40%
        (7,  0,      8192,  "Performance — 50% system RAM, large cache tiles"),      # RAM = 50%
        (8,  0,      16384, "Aggressive — 60% system RAM, very large tiles"),       # RAM = 60%
        (9,  0,      16384, "Very Aggressive — 70% system RAM, very large tiles"),   # RAM = 70%
        (10, 0,      32768, "Maximum Greed — 80% system RAM, maximum cache tiles"),  # RAM = 80%
    ]

    # Description templates for interpolated levels
    _GEAR_DESCRIPTIONS: Dict[int, str] = {
        1:  "Ultra Conservative — minimal RAM (300 MB), no cache tiling",
        2:  "Very Conservative — 400 MB RAM, no cache tiling",
        3:  "Conservative — 512 MB RAM, small cache tiles",
        4:  "Moderate — 768 MB RAM, standard cache tiles",
        5:  "Balanced — stable performance with moderate RAM usage",
        6:  "Performance-Trending — high RAM, large cache tiles",
        7:  "Performance — elevated RAM, large cache tiles",
        8:  "Aggressive — high RAM, very large cache tiles",
        9:  "Very Aggressive — very high RAM, very large tiles",
        10: "Maximum Greed — 80% system RAM, maximum cache tiles",
    }

    def __init__(self, system_total_ram_mb: int) -> None:
        """
        Create a PerformanceGovernor for a specific device.

        Args:
            system_total_ram_mb: Total physical RAM of the target device,
                                 in megabytes. For example: 2048 for a 2 GB
                                 device, 8192 for an 8 GB device.

        Raises:
            ValueError: If system_total_ram_mb <= 0.
        """
        if system_total_ram_mb <= 0:
            raise ValueError(
                f"PerformanceGovernor: system_total_ram_mb must be > 0, "
                f"got {system_total_ram_mb}."
            )
        self._total_ram_mb: int = system_total_ram_mb
        self._total_ram_bytes: int = system_total_ram_mb * 1024 * 1024

    @property
    def system_total_ram_mb(self) -> int:
        """Total system RAM in megabytes (read-only)."""
        return self._total_ram_mb

    @property
    def system_total_ram_bytes(self) -> int:
        """Total system RAM in bytes (read-only)."""
        return self._total_ram_bytes

    # =====================================================================
    #  Configuration Generation
    # =====================================================================

    def get_config(self, gear_level: int) -> Dict[str, object]:
        """
        Generate hardware configuration for a given gear level.

        The configuration includes:
          - ram_budget_bytes: How much physical RAM to lock.
          - block_size: TernaryMathEngine cache tiling parameter.
          - prefetch_ahead: Number of layers to prefetch ahead (always 1).
          - description: Human-readable explanation of the gear level.

        Args:
            gear_level: Integer from 1 to 10.
                        1 = ultra conservative, 10 = maximum greed.

        Returns:
            Dictionary with all configuration parameters.

        Raises:
            ValueError: If gear_level is not in [1, 10].
        """
        gear = max(1, min(10, gear_level))

        ram_budget_bytes = self._compute_ram_budget(gear)
        block_size = self._compute_block_size(gear)

        ram_budget_mb = ram_budget_bytes / (1024.0 * 1024.0)
        ram_percent = (ram_budget_bytes / self._total_ram_bytes) * 100.0

        return {
            "gear_level": gear,
            "ram_budget_bytes": ram_budget_bytes,
            "ram_budget_mb": round(ram_budget_mb, 1),
            "ram_percent_of_system": round(ram_percent, 1),
            "block_size": block_size,
            "prefetch_ahead": 1,
            "description": self._GEAR_DESCRIPTIONS.get(
                gear, f"Gear {gear}"
            ),
        }

    def apply_config(self, engine: AIOCEngine, gear_level: int) -> Dict[str, object]:
        """
        Apply a gear level configuration directly to an AIOCEngine.

        This is a convenience method that combines get_config() with
        the engine's init_system() and block_size setter.

        Args:
            engine:     An AIOCEngine instance (must not be initialized yet,
                        or must be torn down first).
            gear_level: Integer from 1 to 10.

        Returns:
            The configuration dictionary that was applied.

        Raises:
            RuntimeError: If the engine is already initialized.
            ValueError:   If gear_level is out of range.
        """
        if engine.is_initialized:
            raise RuntimeError(
                "PerformanceGovernor.apply_config: engine is already initialized. "
                "Call engine.teardown() before applying a new config."
            )

        config = self.get_config(gear_level)

        engine.init_system(ram_budget_bytes=config["ram_budget_bytes"])
        engine.block_size = config["block_size"]

        return config

    # =====================================================================
    #  Internal Computation Methods
    # =====================================================================

    def _compute_ram_budget(self, gear: int) -> int:
        """
        Compute the RAM budget in bytes for a given gear level.

        Strategy:
          - Gears 1–5: Use fixed MB values from anchors. If the fixed
            value exceeds the system RAM, clamp to a safe percentage.
          - Gears 6–10: Use percentage of system RAM (40%, 50%, 60%, 70%, 80%).
            The result is always clamped to leave at least 256 MB for the OS.
        """
        if gear <= 5:
            # Fixed MB values from anchors
            idx = gear - 1  # 0-based index
            fixed_mb = self._GEAR_ANCHORS[idx][1]

            # Clamp: never allocate more than 80% of system RAM
            max_bytes = int(self._total_ram_bytes * 0.80)
            budget = min(fixed_mb * 1024 * 1024, max_bytes)

            # Ensure at least 256 MB remains for the OS
            min_os_bytes = 256 * 1024 * 1024
            budget = min(budget, self._total_ram_bytes - min_os_bytes)

            return max(budget, 64 * 1024 * 1024)  # At least 64 MB
        else:
            # Percentage-based: gears 6–10 map to 40%–80%
            # gear 6 → 40%, gear 10 → 80%
            t = (gear - 6) / 4.0  # 0.0 (gear 6) to 1.0 (gear 10)
            percent = 0.40 + t * 0.40  # 0.40 to 0.80

            budget = int(self._total_ram_bytes * percent)

            # Ensure at least 256 MB remains for the OS
            min_os_bytes = 256 * 1024 * 1024
            budget = min(budget, self._total_ram_bytes - min_os_bytes)

            return max(budget, 64 * 1024 * 1024)  # At least 64 MB

    def _compute_block_size(self, gear: int) -> int:
        """
        Compute the TernaryMathEngine block size for a given gear level.

        Strategy:
          - Gears 1–2: 0 (disabled) — saves memory, no cache tiling overhead.
          - Gears 3–4: Interpolate 2048 → 4096.
          - Gears 5–6: 4096 (default).
          - Gears 7–8: Interpolate 8192 → 16384.
          - Gear 9: 16384.
          - Gear 10: 32768 (maximum — largest L1-friendly tile).

        The block size controls the TernaryMathEngine's cache tiling:
          - 0: No tiling. Processes entire output at once.
              Pros: No loop overhead. Cons: Cache misses on large layers.
          - 4096: 16 KB per tile. Fits comfortably in 32 KB L1 cache.
          - 8192: 32 KB per tile. Fits in 64 KB L1 cache.
          - 16384: 64 KB per tile. Requires 64+ KB L1 or L2 cache.
          - 32768: 128 KB per tile. Requires large L2 cache.
        """
        # Direct lookup table for exact values at each gear
        block_sizes = {
            1:  0,
            2:  0,
            3:  2048,
            4:  4096,
            5:  4096,
            6:  4096,
            7:  8192,
            8:  16384,
            9:  16384,
            10: 32768,
        }
        return block_sizes[gear]

    # =====================================================================
    #  Diagnostics
    # =====================================================================

    def describe_all_gears(self) -> List[Dict[str, object]]:
        """
        Generate configuration summaries for all 10 gear levels.

        Returns:
            List of 10 configuration dictionaries, one per gear level.
            Useful for displaying a table of options to the user.
        """
        return [self.get_config(g) for g in range(1, 11)]

    def recommend_gear(self, min_ram_mb: int = 0, prefer_speed: bool = False) -> int:
        """
        Recommend a gear level based on system constraints.

        Args:
            min_ram_mb:   Minimum RAM budget needed by the model (in MB).
                          The governor will pick the lowest gear that provides
                          at least this much RAM.
            prefer_speed: If True, picks a higher gear (more RAM, more speed).
                          If False, picks the lowest sufficient gear.

        Returns:
            Recommended gear level (1–10).
        """
        # Find the lowest gear that provides enough RAM
        for gear in range(1, 11):
            config = self.get_config(gear)
            if config["ram_budget_mb"] >= min_ram_mb:
                if prefer_speed and gear < 10:
                    # Bump up 2 gears for speed preference (but cap at 10)
                    gear = min(gear + 2, 10)
                return gear

        # Even gear 10 doesn't provide enough RAM
        return 10

    def __repr__(self) -> str:
        return (
            f"PerformanceGovernor(system_total_ram_mb={self._total_ram_mb})"
        )


# =============================================================================
#  Platform Utility Functions
# =============================================================================

def detect_system_ram_mb() -> int:
    """
    Detect the total physical RAM of the current system, in megabytes.

    Priority:
      1. psutil (preferred — cross-platform, reliable, one-liner)
      2. /proc/meminfo (Linux fallback)
      3. sysctl hw.memsize (macOS fallback)
      4. GlobalMemoryStatusEx (Windows fallback)

    Returns:
        Total RAM in MB (integer), or 0 if detection fails.
    """
    # --- Method 1: psutil (preferred) -----------------------------------
    if _HAS_PSUTIL:
        try:
            return int(psutil.virtual_memory().total // (1024 * 1024))
        except Exception:
            pass

    # --- Method 2: Platform-specific fallbacks --------------------------
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        kb = int(parts[1])
                        return kb // 1024
        except (OSError, ValueError, IndexError):
            pass

    elif system == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            bytes_val = int(result.stdout.strip())
            return bytes_val // (1024 * 1024)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass

    elif system == "Windows":
        try:
            import ctypes as _ctypes
            kernel32 = _ctypes.windll.kernel32
            class MEMORYSTATUSEX(_ctypes.Structure):
                _fields_ = [
                    ("dwLength", _ctypes.c_ulong),
                    ("dwMemoryLoad", _ctypes.c_ulong),
                    ("ullTotalPhys", _ctypes.c_ulonglong),
                    ("ullAvailPhys", _ctypes.c_ulonglong),
                    ("ullTotalPageFile", _ctypes.c_ulonglong),
                    ("ullAvailPageFile", _ctypes.c_ulonglong),
                    ("ullTotalVirtual", _ctypes.c_ulonglong),
                    ("ullAvailVirtual", _ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", _ctypes.c_ulonglong),
                ]
            status = MEMORYSTATUSEX()
            status.dwLength = _ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(_ctypes.byref(status))
            return int(status.ullTotalPhys // (1024 * 1024))
        except Exception:
            pass

    return 0


def create_engine_with_governor(
    lib_path: str,
    gear_level: int,
    system_ram_mb: Optional[int] = None,
) -> Tuple[AIOCEngine, PerformanceGovernor]:
    """
    Convenience function: create an AIOCEngine pre-configured with a governor.

    Args:
        lib_path:       Path to the shared library.
        gear_level:     Desired gear level (1–10).
        system_ram_mb:  Total system RAM in MB. If None, auto-detects.

    Returns:
        Tuple of (AIOCEngine, PerformanceGovernor).
        The engine is initialized but no model is loaded yet.

    Example:
        engine, gov = create_engine_with_governor("./libai_os_core.so", gear_level=5)
        engine.load_model("model.aioc")
        # ... inference ...
        engine.teardown()
    """
    if system_ram_mb is None:
        system_ram_mb = detect_system_ram_mb()
        if system_ram_mb == 0:
            raise RuntimeError(
                "Cannot detect system RAM. Please provide system_ram_mb explicitly."
            )

    engine = AIOCEngine(lib_path)
    governor = PerformanceGovernor(system_total_ram_mb=system_ram_mb)
    governor.apply_config(engine, gear_level)

    return engine, governor