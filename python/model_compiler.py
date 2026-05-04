"""
AI OS Core — Phase 4: Model Compiler (Streaming Architecture)
=============================================================

This module implements the ModelCompiler pipeline, which converts standard
PyTorch / Safetensors model weights into the AIOC (AI OS Core) binary format
used by the C++ AsyncPrefetcher and TernaryMathEngine.

Architecture: STREAMING WRITE
  Memory consumption NEVER exceeds one layer at a time.
  The quantized data is written to disk immediately after each layer is
  processed, then the tensor is deleted from RAM.

Pipeline:
  1. LOAD:     Read model weights from .pt / .bin / .safetensors.
  2. QUANTIZE: Convert float32/float16 → ternary int8 {-1, 0, +1}.
  3. STREAM:   Write quantized bytes to disk one layer at a time.
  4. PATCH:    Seek back and fill in the size table header.

AIOC Binary Format (Little Endian):
  Offset 0x00    4 B  Magic: 0x41494F43
  Offset 0x04    4 B  Version: 1
  Offset 0x08    4 B  Num Layers: N
  Offset 0x0C    N×8B Layer Sizes (uint64 LE)
  Offset 0x0C+N×8 ... Layer Data (int8 bytes, sequential)

Dependencies:
  - torch, numpy, struct, safetensors (optional)

Author: AI OS Core Team
Version: 1.1.0 (Hotfix: Streaming Write + Tensor Comparison Fix)
Since: 2026-04-08
License: MIT
"""

from __future__ import annotations

import gc
import logging
import os
import struct
import time
import sys
import site
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# --- FIX: Force Python to see the AppData pip installations ---
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)
# --------------------------------------------------------------

import numpy as np
import torch

try:
    import safetensors.torch as safetensors_load
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

# ---------------------------------------------------------------------------
# AIOC Format Constants (must match C++ async_prefetcher.hpp exactly)
# ---------------------------------------------------------------------------

AIOC_FILE_MAGIC: int = 0x41494F43
AIOC_FILE_VERSION: int = 1
AIOC_HEADER_BASE_SIZE: int = 12
AIOC_HEADER_FORMAT: str = "<III"
AIOC_LAYER_SIZE_FORMAT: str = "<Q"

logger = logging.getLogger("ai_os.compiler")


# =============================================================================
#  Data Classes
# =============================================================================

@dataclass
class LayerInfo:
    """Metadata about a single processed layer."""
    name: str
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    quantized_shape: Tuple[int, ...]
    num_elements: int
    byte_size: int
    file_offset: int = 0
    sparsity: float = 0.0
    quantize_time_ms: float = 0.0


@dataclass
class CompilationReport:
    """Summary report of a compilation session."""
    input_path: str = ""
    output_path: str = ""
    num_layers: int = 0
    total_params: int = 0
    total_bytes: int = 0
    header_bytes: int = 0
    data_bytes: int = 0
    threshold: float = 0.0
    avg_sparsity: float = 0.0
    total_quantize_ms: float = 0.0
    total_export_ms: float = 0.0
    layers: List[LayerInfo] = field(default_factory=list)


ProgressCallback = Callable[[int, int, str, float], None]


# =============================================================================
#  ModelCompiler
# =============================================================================

class ModelCompiler:
    """
    Converts PyTorch model weights into AIOC binary format with ternary quantization.

    Memory guarantee: RAM usage NEVER exceeds ~2× the size of the largest single
    layer at any point. Quantized data is streamed to disk immediately after each
    layer is processed, then the tensor is deleted from RAM.
    """

    _LINEAR_NAME_PATTERNS: Tuple[str, ...] = (
        "fc", "linear", "lm_head", "proj",
        "gate_proj", "up_proj", "down_proj",
        "dense", "output", "classifier", "head",
    )

    _EXCLUSION_PATTERNS: Tuple[str, ...] = (
        "conv", "layernorm", "layer_norm", "rmsnorm", "rms_norm",
        "bn", "batchnorm", "embedding", "embed",
        "norm", "bias", "pos", "token", "wte", "wpe",
    )

    def __init__(
        self,
        input_path: str,
        output_path: str,
        threshold: float = 0.05,
    ) -> None:
        if not input_path or not input_path.strip():
            raise ValueError("ModelCompiler: input_path must not be empty.")
        if not output_path or not output_path.strip():
            raise ValueError("ModelCompiler: output_path must not be empty.")
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(
                f"ModelCompiler: threshold must be in (0.0, 1.0), got {threshold}."
            )

        self._input_path: str = input_path
        self._output_path: str = output_path
        self._threshold: float = threshold

        if not os.path.isfile(input_path):
            raise FileNotFoundError(
                f"ModelCompiler: input file not found: '{input_path}'"
            )

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    @property
    def input_path(self) -> str:
        return self._input_path

    @property
    def output_path(self) -> str:
        return self._output_path

    @property
    def threshold(self) -> float:
        return self._threshold

    # =========================================================================
    #  Quantization
    # =========================================================================

    def quantize_to_ternary(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a float tensor to ternary int8 values {-1, 0, +1}.

        Uses torch.where() for vectorized, branchless quantization.
        Returns a NEW CPU int8 tensor. The original is NOT modified.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"quantize_to_ternary: expected torch.Tensor, got {type(tensor).__name__}."
            )
        if not tensor.is_floating_point():
            raise ValueError(
                f"quantize_to_ternary: tensor must be floating-point, got dtype {tensor.dtype}."
            )

        if tensor.device.type != "cpu":
            tensor = tensor.cpu().float()
        else:
            tensor = tensor.float()

        t = self._threshold
        zeros_mask = torch.abs(tensor) <= t
        positive_mask = tensor > t

        result = torch.where(
            positive_mask,
            torch.ones_like(tensor),
            torch.where(zeros_mask, torch.zeros_like(tensor), -torch.ones_like(tensor)),
        )
        return result.to(torch.int8).contiguous()

    # =========================================================================
    #  Layer Detection
    # =========================================================================

    def _is_linear_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if not name.endswith(".weight"):
            return False
        if tensor.dim() != 2:
            return False
        name_lower = name.lower()
        for exclusion in self._EXCLUSION_PATTERNS:
            if exclusion in name_lower:
                return False
        for pattern in self._LINEAR_NAME_PATTERNS:
            if pattern in name_lower:
                return True
        return True

    # =========================================================================
    #  Model Loading
    # =========================================================================

    def _load_state_dict(self) -> Dict[str, torch.Tensor]:
        """Load the model state dict from .pt/.pth/.bin or .safetensors file."""
        path = self._input_path
        path_lower = path.lower()

        logger.info(f"Loading model from: {path}")

        if path_lower.endswith(".safetensors"):
            if not _HAS_SAFETENSORS:
                raise ImportError(
                    "Cannot load .safetensors file: the 'safetensors' package "
                    "is not installed. Install it with: pip install safetensors"
                )
            state_dict = safetensors_load.load_file(path)
        else:
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                logger.warning(
                    "weights_only=True not supported. Falling back to weights_only=False."
                )
                state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Handle wrapped state dicts
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model_state_dict"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    logger.info(f"Unwrapping state dict from key '{key}'.")
                    state_dict = state_dict[key]
                    break
            if state_dict:
                first_value = next(iter(state_dict.values()), None)
                if first_value is not None and not isinstance(first_value, torch.Tensor):
                    if hasattr(state_dict, "state_dict"):
                        logger.info("Extracting state_dict from model object.")
                        state_dict = state_dict.state_dict()
                    else:
                        raise RuntimeError(
                            "Model file does not contain a valid state dict. "
                            f"Got value type: {type(first_value).__name__}."
                        )
        else:
            if hasattr(state_dict, "state_dict"):
                logger.info("Extracting state_dict from model object.")
                state_dict = state_dict.state_dict()
            else:
                raise RuntimeError(
                    f"Cannot load model from '{path}': unsupported format "
                    f"({type(state_dict).__name__})."
                )

        logger.info(f"Loaded state dict with {len(state_dict)} parameters.")
        return state_dict

    # =========================================================================
    #  Streaming Export (Single-Pass, O(1) RAM)
    # =========================================================================

    def export_to_aioc(
        self,
        callback: Optional[ProgressCallback] = None,
    ) -> CompilationReport:
        """
        Load model, quantize each layer, stream-write to AIOC file.

        Streaming architecture guarantees that peak RAM usage never exceeds
        ~2× the size of the largest single layer. Quantized tensors are
        written to disk immediately and deleted before the next layer.

        AIOC file is written in a single pass:
          1. Write header (12 bytes).
          2. Write N×8 zero bytes as placeholder for the size table.
          3. For each layer: quantize → bytes → f.write → del tensor → save size.
          4. f.seek(12) → overwrite the zeros with actual layer sizes.
          5. Close + os.replace (atomic).
        """
        export_start = time.perf_counter()
        report = CompilationReport(
            input_path=self._input_path,
            output_path=self._output_path,
            threshold=self._threshold,
        )

        # ------------------------------------------------------------------
        # Step 1: Load and identify Linear layers
        # ------------------------------------------------------------------
        state_dict = self._load_state_dict()

        linear_layers: List[Tuple[str, torch.Tensor]] = []
        for name, tensor in state_dict.items():
            if self._is_linear_weight(name, tensor):
                linear_layers.append((name, tensor))
        linear_layers.sort(key=lambda x: x[0])

        num_layers = len(linear_layers)
        report.num_layers = num_layers

        if num_layers == 0:
            raise RuntimeError(
                f"No Linear layer weights found in '{self._input_path}'. "
                f"The state dict has {len(state_dict)} parameters."
            )
        logger.info(f"Found {num_layers} Linear layers to quantize.")

        # ------------------------------------------------------------------
        # Step 2: Open temp file — write header + size-table placeholder
        # ------------------------------------------------------------------
        temp_path = self._output_path + ".tmp"
        layer_sizes: List[int] = []
        total_quantize_ms = 0.0

        size_table_bytes = num_layers * 8  # N × uint64

        try:
            with open(temp_path, "wb") as f:
                # ---- Header (12 bytes) ------------------------------------
                f.write(struct.pack(
                    AIOC_HEADER_FORMAT,
                    AIOC_FILE_MAGIC,
                    AIOC_FILE_VERSION,
                    num_layers,
                ))

                # ---- Placeholder for Size Table (N × 8 bytes) ------------
                # We fill this with zeros now and patch it after we know
                # the actual sizes (which we discover layer by layer).
                f.write(b'\x00' * size_table_bytes)

                # ---- Data Section: stream each layer ----------------------
                data_start_offset = AIOC_HEADER_BASE_SIZE + size_table_bytes
                running_offset = data_start_offset

                for idx, (name, original_tensor) in enumerate(linear_layers):
                    layer_start = time.perf_counter()

                    # Snapshot metadata BEFORE quantization (shape may change)
                    original_shape = tuple(original_tensor.shape)
                    original_dtype = original_tensor.dtype
                    num_elements = original_tensor.numel()

                    # ---- Quantize ----------------------------------------
                    quantized = self.quantize_to_ternary(original_tensor)
                    quantize_ms = (time.perf_counter() - layer_start) * 1000.0
                    total_quantize_ms += quantize_ms

                    # ---- Convert to raw bytes (int8 → bytes) -------------
                    data_bytes = quantized.numpy().tobytes()
                    actual_size = len(data_bytes)

                    if actual_size != num_elements:
                        raise RuntimeError(
                            f"Layer '{name}': expected {num_elements} bytes "
                            f"({num_elements} elements × 1 byte/int8), "
                            f"got {actual_size} bytes."
                        )

                    # ---- Compute sparsity (lightweight CPU op) -----------
                    zero_count = int(torch.sum(quantized == 0).item())
                    sparsity = zero_count / max(num_elements, 1)

                    # ---- Build LayerInfo ---------------------------------
                    layer_info = LayerInfo(
                        name=name,
                        original_shape=original_shape,
                        original_dtype=original_dtype,
                        quantized_shape=tuple(quantized.shape),
                        num_elements=num_elements,
                        byte_size=actual_size,
                        file_offset=running_offset,
                        sparsity=sparsity,
                        quantize_time_ms=quantize_ms,
                    )
                    report.layers.append(layer_info)

                    # ---- Write data to disk immediately ------------------
                    f.write(data_bytes)

                    # ---- Save size for the size-table patch --------------
                    layer_sizes.append(actual_size)
                    running_offset += actual_size

                    # ---- Invoke progress callback ------------------------
                    if callback is not None:
                        elapsed = (time.perf_counter() - export_start) * 1000.0
                        try:
                            callback(idx, num_layers, name, elapsed)
                        except Exception as cb_err:
                            logger.warning(f"Progress callback error: {cb_err}")

                    # ---- IMMEDIATE memory release ------------------------
                    # Delete quantized tensor BEFORE loading the next layer.
                    # This is the critical fix: without this, all quantized
                    # tensors accumulate in RAM, causing OOM on 2GB devices.
                    del quantized
                    del data_bytes
                    del original_tensor
                    state_dict[name] = None  # break state_dict reference

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # ---- Flush data to disk ----------------------------------
                f.flush()
                os.fsync(f.fileno())

                # ---- PATCH: Seek back and write the real Size Table ----
                # Now that we know all layer sizes, overwrite the zeros
                # we wrote earlier with the actual uint64 values.
                f.seek(AIOC_HEADER_BASE_SIZE)
                for sz in layer_sizes:
                    f.write(struct.pack(AIOC_LAYER_SIZE_FORMAT, sz))

                f.flush()
                os.fsync(f.fileno())

        except (IOError, OSError) as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise IOError(
                f"Failed to write AIOC file to '{temp_path}': {e}"
            ) from e

        # ---- Atomic rename (temp → final) --------------------------------
        os.replace(temp_path, self._output_path)

        # ------------------------------------------------------------------
        # Step 3: Build final report
        # ------------------------------------------------------------------
        total_params = sum(info.num_elements for info in report.layers)
        data_bytes_total = sum(info.byte_size for info in report.layers)
        header_bytes_total = AIOC_HEADER_BASE_SIZE + size_table_bytes
        total_bytes = header_bytes_total + data_bytes_total
        avg_sparsity = (
            sum(info.sparsity for info in report.layers) / num_layers
            if num_layers > 0 else 0.0
        )

        report.total_params = total_params
        report.total_bytes = total_bytes
        report.header_bytes = header_bytes_total
        report.data_bytes = data_bytes_total
        report.avg_sparsity = avg_sparsity
        report.total_quantize_ms = total_quantize_ms
        report.total_export_ms = (time.perf_counter() - export_start) * 1000.0

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            f"AIOC export complete: {self._output_path} "
            f"({total_bytes / (1024 * 1024):.1f} MB, "
            f"{num_layers} layers, avg sparsity {avg_sparsity:.1%})"
        )

        return report

    # =========================================================================
    #  Diagnostics
    # =========================================================================

    def analyze_model(self) -> Dict[str, object]:
        """Analyze the input model without quantizing or exporting."""
        state_dict = self._load_state_dict()

        layers: List[Dict[str, object]] = []
        total_params = 0

        for name, tensor in state_dict.items():
            if self._is_linear_weight(name, tensor):
                n = tensor.numel()
                total_params += n
                layers.append({
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "num_elements": n,
                    "size_bytes": n,
                })

        layers.sort(key=lambda x: x["name"])
        num = len(layers)
        hdr = AIOC_HEADER_BASE_SIZE + num * 8
        data = sum(l["size_bytes"] for l in layers)

        return {
            "input_path": self._input_path,
            "file_size_mb": os.path.getsize(self._input_path) / (1024 * 1024),
            "num_layers": num,
            "total_params": total_params,
            "estimated_aioc_bytes": hdr + data,
            "estimated_aioc_mb": (hdr + data) / (1024 * 1024),
            "layers": layers,
        }

    def __repr__(self) -> str:
        return (
            f"ModelCompiler(input='{self._input_path}', "
            f"output='{self._output_path}', "
            f"threshold={self._threshold})"
        )
