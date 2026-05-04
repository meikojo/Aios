, OpenAI SDK, Web UI)                      │
  │  POST /v1/chat/completions {"messages": [...], "stream": true}│
  └──────────────────────────────┬───────────────────────────────┘
                                 │ HTTP
  ┌──────────────────────────────▼───────────────────────────────┐
  │  FastAPI (uvicorn)                                           │
  │  asyncio.Lock → FIFO queue → single-engine access           │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ Python calls
  ┌──────────────────────────────▼───────────────────────────────┐
  │  engine_bridge.py                                           │
  │  AIOCEngine · PerformanceGovernor                           │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ ctypes FFI
  ┌──────────────────────────────▼───────────────────────────────┐
  │  libai_os_core.so (C API)                                    │
  │  SovereignArena + SlidingWindowManager + AsyncPrefetcher     │
  │  + TernaryMathEngine                                        │
  └──────────────────────────────────────────────────────────────┘

Usage (Headless Server):
  # Install dependencies
  pip install fastapi uvicorn numpy psutil

  # Run the server
  python api_server.py \\
    --lib-path ./libai_os_core.so \\
    --model-path ./model.aioc \\
    --model-name "my-model" \\
    --layer-dims 768:768 768:3072 3072:768 \\
    --vocab-size 32000 \\
    --hidden-dim 768 \\
    --gear 5 \\
    --host 0.0.0.0 \\
    --port 8000

  # Or via environment variables
  AIOC_LIB_PATH=./libai_os_core.so \\
  AIOC_MODEL_PATH=./model.aioc \\
  AIOC_MODEL_NAME=my-model \\
  AIOC_LAYER_DIMS=768:768,768:3072,3072:768 \\
  AIOC_VOCAB_SIZE=32000 \\
  AIOC_HIDDEN_DIM=768 \\
  AIOC_GEAR=5 \\
  python api_server.py

  # Test with curl
  curl -X POST http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"my-model","messages":[{"role":"user","content":"Hello"}],'
       '"max_tokens":50,"stream":false}'

Dependencies:
  - fastapi, uvicorn (server)
  - numpy (array operations)
  - pydantic (request/response models, bundled with FastAPI)
  - psutil (RAM detection, optional)

Author: AI OS Core Team
Version: 1.0.0
Since: 2026-04-09
License: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_os.server")


# =============================================================================
#  Tokenizer Protocol & Implementations
# =============================================================================

class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.

    The AI OS Core only handles ternary matmul — it has no built-in
    tokenizer. The server uses a pluggable tokenizer to convert between
    text and token IDs.

    For production use, integrate a real tokenizer:
      - tiktoken (OpenAI models)
      - sentencepiece (LLaMA, Mistral)
      - HuggingFace tokenizers
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token IDs."""
        ...

    @abstractmethod
    def decode(self, token_id: int) -> str:
        """Convert a single token ID to text."""
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...


class CharacterTokenizer(Tokenizer):
    """
    Simple character-level tokenizer for testing and development.

    Maps each Unicode codepoint to its integer value. NOT suitable
    for production models (real models need subword tokenizers).
    """

    def __init__(self) -> None:
        self._vocab_size = 128  # ASCII subset for safety

    def encode(self, text: str) -> List[int]:
        return [min(ord(c), self._vocab_size - 1) for c in text]

    def decode(self, token_id: int) -> str:
        return chr(token_id) if 0 <= token_id < self._vocab_size else ""

    def vocab_size(self) -> int:
        return self._vocab_size


class IdentityTokenizer(Tokenizer):
    """
    Passthrough tokenizer that returns raw byte values.

    Useful when the embedding table is organized by raw byte values.
    Each byte (0-255) maps directly to a vocabulary entry.
    """

    def __init__(self) -> None:
        self._vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_id: int) -> str:
        if 0 <= token_id < self._vocab_size:
            return bytes([token_id]).decode("utf-8", errors="replace")
        return ""

    def vocab_size(self) -> int:
        return self._vocab_size


# =============================================================================
#  Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for a loaded AIOC model.

    This metadata is required by the inference pipeline to know how
    to feed data through the model's layers and produce output tokens.

    Attributes:
        name:            Model name (returned in API responses).
        file_path:       Path to the .aioc model file.
        layer_dims:      List of (input_dim, output_dim) for each layer.
        vocab_size:      Total vocabulary size (for final logits).
        hidden_dim:      Hidden dimension size (for embedding lookup).
        eos_token_id:    End-of-sequence token ID.
        max_seq_len:     Maximum sequence length.
        embedding_table: Optional pre-loaded embedding matrix (float32).
                         Shape: (vocab_size, hidden_dim).
    """

    name: str
    file_path: str
    layer_dims: List[Tuple[int, int]]
    vocab_size: int
    hidden_dim: int
    eos_token_id: int = 2
    max_seq_len: int = 2048
    embedding_table: Optional[np.ndarray] = None

    @property
    def num_layers(self) -> int:
        return len(self.layer_dims)

    def __post_init__(self) -> None:
        if not self.layer_dims:
            raise ValueError("ModelConfig: layer_dims must not be empty.")
        if self.vocab_size <= 0:
            raise ValueError(f"ModelConfig: vocab_size must be > 0, got {self.vocab_size}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"ModelConfig: hidden_dim must be > 0, got {self.hidden_dim}.")


# =============================================================================
#  Sampling Utilities
# =============================================================================

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax with temperature scaling.

    Applies temperature to logits before softmax. Higher temperature
    produces more uniform distributions (more random). Lower temperature
    produces sharper distributions (more deterministic).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature > 0:
        logits = logits / temperature
    # Numerical stability: subtract max
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def top_k_filtering(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep only the top-k logits, set the rest to -inf."""
    if k <= 0 or k >= len(logits):
        return logits
    indices = np.argsort(logits)[::-1]
    threshold = logits[indices[k - 1]]
    logits = logits.copy()
    logits[logits < threshold] = -np.inf
    return logits


def top_p_filtering(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filtering: keep the smallest set of tokens with cumulative prob >= p."""
    if p >= 1.0:
        return logits
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    probs = softmax(sorted_logits, temperature=1.0)
    cumulative_probs = np.cumsum(probs)
    # Find cutoff: first index where cumulative prob exceeds p
    cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
    # Zero out everything beyond the cutoff
    filtered = logits.copy()
    removed_indices = sorted_indices[cutoff_idx:]
    filtered[removed_indices] = -np.inf
    return filtered


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample a token ID from logits using temperature, top-k, and top-p.

    Args:
        logits:      Raw output logits from the model (float32 array).
        temperature: Scaling factor (1.0 = normal, <1 = sharp, >1 = random).
        top_k:       Keep only top-k tokens (0 = disabled).
        top_p:       Nucleus filtering threshold (1.0 = disabled).

    Returns:
        Sampled token ID (int).
    """
    filtered = logits.copy()
    if top_k > 0:
        filtered = top_k_filtering(filtered, top_k)
    if top_p < 1.0:
        filtered = top_p_filtering(filtered, top_p)
    probs = softmax(filtered, temperature=temperature)
    return int(np.random.choice(len(probs), p=probs))


# =============================================================================
#  Inference Pipeline
# =============================================================================

class InferencePipeline:
    """
    Orchestrates the complete inference cycle: embed → forward → sample.

    This class wraps the AIOCEngine's per-layer processing into a
    complete autoregressive token generation loop. It handles:

    1. Embedding: Converts token IDs to float vectors.
    2. Forward pass: Processes through all ternary layers sequentially.
    3. Sampling: Converts output logits to next token ID.
    4. Autoregressive loop: Repeats for max_tokens iterations.

    The pipeline is NOT thread-safe. External synchronization
    (asyncio.Lock in the server) is required for concurrent access.

    Performance Note:
      After the first forward pass, the model file is cached by the OS
      in the page cache. Subsequent reload_model() calls read from RAM
      (~50-200ms) instead of disk (~1-5s), making autoregressive
      generation feasible even on HDD-based systems.
    """

    def __init__(
        self,
        engine: Any,  # AIOCEngine (avoid circular import)
        model_config: ModelConfig,
        tokenizer: Tokenizer,
    ) -> None:
        self.engine = engine
        self.config = model_config
        self.tokenizer = tokenizer
        self._first_pass_done = False

    def _embed_token(self, token_id: int) -> np.ndarray:
        """
        Convert a token ID to a float32 activation vector.

        Uses the model's embedding table if available. Otherwise,
        creates a simple one-hot-like representation.

        Args:
            token_id: Token ID to embed.

        Returns:
            Float32 numpy array of shape (hidden_dim,).
        """
        if self.config.embedding_table is not None:
            table = self.config.embedding_table
            if 0 <= token_id < len(table):
                return table[token_id].astype(np.float32)
            # Out of vocabulary — use zero vector
            return np.zeros(self.config.hidden_dim, dtype=np.float32)
        else:
            # Simple hash-based embedding for testing without a real table.
            # In production, always provide an embedding_table.
            rng = np.random.RandomState(token_id + 42)
            vec = rng.randn(self.config.hidden_dim).astype(np.float32) * 0.1
            return vec

    def forward_pass(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Run a complete forward pass through all model layers.

        For each layer: wait → commit → process → retire.

        Args:
            input_vec: Input activation vector (float32).

        Returns:
            Output activation vector (float32) after the last layer.
        """
        current = input_vec.copy()
        for layer_id, (in_dim, out_dim) in enumerate(self.config.layer_dims):
            self.engine.wait_layer(layer_id)
            self.engine.commit_layer()
            current = self.engine.process_layer(current, in_dim, out_dim)
            self.engine.retire_layer()
        return current

    def generate_next_token(
        self,
        last_token_id: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """
        Generate a single next token given the previous token.

        This is one step of the autoregressive loop:
          1. Embed the last token
          2. Forward pass through all layers
          3. Reload pipeline for the next iteration
          4. Sample next token from output logits

        Args:
            last_token_id: The most recently generated token ID.
            temperature:   Sampling temperature.
            top_k:         Top-k filtering (0 = disabled).
            top_p:         Nucleus filtering (1.0 = disabled).

        Returns:
            Next token ID (int).
        """
        # Step 1: Embed
        x = self._embed_token(last_token_id)

        # Step 2: Forward pass
        output = self.forward_pass(x)

        # Step 3: Reload pipeline for next iteration
        # (do this BEFORE sampling so we're ready for the next call)
        if self._first_pass_done:
            self.engine.reload_model()
        else:
            self._first_pass_done = True

        # Step 4: Sample from logits
        # The last layer's output_dim should equal vocab_size
        logits = output[:self.config.vocab_size]  # Truncate if needed
        if len(logits) < self.config.vocab_size:
            # Pad with zeros if the model output is smaller than vocab_size
            padded = np.zeros(self.config.vocab_size, dtype=np.float32)
            padded[:len(logits)] = logits
            logits = padded

        return sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    def generate(
        self,
        input_ids: List[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Generate a sequence of tokens autoregressively.

        Args:
            input_ids:   Initial token IDs (prompt).
            max_tokens:  Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k:       Top-k filtering.
            top_p:       Nucleus filtering.
            stop_tokens: List of token IDs that stop generation.

        Returns:
            List of generated token IDs (excluding the prompt).
        """
        if stop_tokens is None:
            stop_tokens = [self.config.eos_token_id]

        generated: List[int] = []
        last_token = input_ids[-1] if input_ids else 0

        for _ in range(max_tokens):
            next_token = self.generate_next_token(
                last_token, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            generated.append(next_token)

            if next_token in stop_tokens:
                break

            last_token = next_token

            if len(input_ids) + len(generated) >= self.config.max_seq_len:
                break

        return generated

    def reset(self) -> None:
        """Reset the pipeline for a new generation sequence."""
        self._first_pass_done = False
        if self.engine.is_initialized:
            try:
                self.engine.reset_pipeline()
            except Exception as e:
                logger.warning(f"Pipeline reset warning: {e}")


# =============================================================================
#  FastAPI Application & OpenAI-Compatible Endpoints
# =============================================================================

# Lazy imports — only import FastAPI when actually running the server.
# This allows the module to be imported for testing without requiring FastAPI.

_fastapi_app = None
_app_state: Dict[str, Any] = {}


def _get_app():
    """Lazily create and return the FastAPI application."""
    global _fastapi_app
    if _fastapi_app is not None:
        return _fastapi_app

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field

    # =================================================================
    #  Pydantic Models (OpenAI Format)
    # =================================================================

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: float = Field(default=1.0, ge=0.0, le=10.0)
        top_k: int = Field(default=0, ge=0)
        top_p: float = Field(default=1.0, ge=0.0, le=1.0)
        max_tokens: int = Field(default=256, ge=1, le=8192)
        stream: bool = False
        stop: Optional[List[str]] = None

    class ModelObject(BaseModel):
        id: str
        object: str = "model"
        owned_by: str = "ai-os-core"

    class ModelListResponse(BaseModel):
        object: str = "list"
        data: List[ModelObject]

    class UsageInfo(BaseModel):
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    # =================================================================
    #  Application Factory
    # =================================================================

    app = FastAPI(
        title="AI OS Core API",
        description=(
            "OpenAI-compatible API server for AI OS Core inference engine. "
            "Runs ternary-quantized models on resource-constrained devices."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # --- Global State --------------------------------------------------
    # These are set during server initialization (main block).
    # engine: AIOCEngine
    # pipeline: InferencePipeline
    # engine_lock: asyncio.Lock
    # model_config: ModelConfig
    # tokenizer: Tokenizer

    # =================================================================
    #  Startup & Shutdown
    # =================================================================

    @app.on_event("startup")
    async def on_startup() -> None:
        """Log server readiness on startup."""
        engine = _app_state.get("engine")
        pipeline = _app_state.get("pipeline")
        model_config = _app_state.get("model_config")
        if engine and model_config:
            logger.info(
                f"AI OS Core API server ready. "
                f"Model: {model_config.name}, "
                f"Layers: {model_config.num_layers}, "
                f"Vocab: {model_config.vocab_size}, "
                f"AVX2: {engine.has_avx2}, "
                f"Arena: {engine.arena_utilization:.1f}%"
            )
        else:
            logger.warning("Server started without a loaded model. "
                           "Load a model via the initialization flow.")

    # =================================================================
    #  Health Check
    # =================================================================

    @app.get("/health")
    async def health_check():
        """Basic liveness probe for load balancers and monitoring."""
        engine = _app_state.get("engine")
        return {
            "status": "ok",
            "engine_initialized": engine is not None and engine.is_initialized,
            "model_loaded": engine is not None and engine.model_path != "",
        }

    # =================================================================
    #  List Models (OpenAI-Compatible)
    # =================================================================

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        """
        List available models (OpenAI-compatible).

        Returns a list with the currently loaded model, if any.
        """
        model_config = _app_state.get("model_config")
        models = []
        if model_config:
            models.append(ModelObject(
                id=model_config.name,
                owned_by="ai-os-core",
            ))
        return ModelListResponse(data=models)

    # =================================================================
    #  Chat Completions (OpenAI-Compatible)
    # =================================================================

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint.

        Accepts a list of messages and generates a response.
        Supports both streaming (SSE) and non-streaming modes.

        Resource Management:
          An asyncio.Lock ensures only one request is processed at a time.
          Concurrent requests wait in FIFO order. This prevents OOM by
          guaranteeing the engine's memory budget is never exceeded.
        """
        engine = _app_state.get("engine")
        pipeline = _app_state.get("pipeline")
        model_config = _app_state.get("model_config")
        engine_lock = _app_state.get("engine_lock")

        if not engine or not pipeline or not model_config:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. The server is running but no model "
                       "has been initialized. Check server logs."
            )

        if request.model != model_config.name:
            # Soft warning — we only have one model, so accept any model name.
            logger.warning(
                f"Requested model '{request.model}' does not match "
                f"loaded model '{model_config.name}'. Using '{model_config.name}'."
            )

        # --- Prepare input ------------------------------------------------
        # Combine all messages into a single prompt string
        prompt_parts: List[str] = []
        for msg in request.messages:
            prompt_parts.append(f"{msg.role}: {msg.content}")
        prompt_text = "\n".join(prompt_parts)

        tokenizer = _app_state["tokenizer"]
        input_ids = tokenizer.encode(prompt_text)
        prompt_tokens = len(input_ids)

        # Stop tokens
        stop_token_ids: List[int] = [model_config.eos_token_id]
        if request.stop:
            for s in request.stop:
                encoded_stop = tokenizer.encode(s)
                if encoded_stop:
                    stop_token_ids.extend(encoded_stop)

        # --- Generate completion ID --------------------------------------
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # --- Stream mode -------------------------------------------------
        if request.stream:
            return StreamingResponse(
                _stream_generator(
                    pipeline=pipeline,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    completion_id=completion_id,
                    created=created,
                    model_name=model_config.name,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    stop_tokens=stop_token_ids,
                    engine_lock=engine_lock,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # --- Non-stream mode ---------------------------------------------
        async with engine_lock:
            try:
                pipeline.reset()
                generated_ids = pipeline.generate(
                    input_ids=input_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    stop_tokens=stop_token_ids,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Decode generated tokens
        output_text = "".join(tokenizer.decode(tid) for tid in generated_ids)
        completion_tokens = len(generated_ids)

        # Determine finish reason
        finish_reason = "length"
        if generated_ids and generated_ids[-1] == model_config.eos_token_id:
            finish_reason = "stop"

        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_config.name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        return JSONResponse(content=response)

    # =================================================================
    #  SSE Stream Generator
    # =================================================================

    async def _stream_generator(
        pipeline: InferencePipeline,
        tokenizer: Tokenizer,
        input_ids: List[int],
        completion_id: str,
        created: int,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        stop_tokens: List[int],
        engine_lock: asyncio.Lock,
    ) -> AsyncIterator[str]:
        """
        Generate tokens one at a time and yield SSE chunks.

        Each chunk follows the OpenAI SSE format:
          data: {"id":"...","object":"chat.completion.chunk","created":...,
                 "model":"...","choices":[{"index":0,"delta":{"content":"..."},
                 "finish_reason":null}]}

        The stream is terminated with:
          data: [DONE]
        """
        async with engine_lock:
            try:
                pipeline.reset()

                # Send initial role chunk
                initial_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(initial_chunk)}\n\n"

                # Generate tokens one at a time
                generated_ids: List[int] = []
                last_token = input_ids[-1] if input_ids else 0
                finish_reason = None

                for step in range(max_tokens):
                    # Run the engine in a thread to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    next_token = await loop.run_in_executor(
                        None,
                        pipeline.generate_next_token,
                        last_token,
                        temperature,
                        top_k,
                        top_p,
                    )
                    generated_ids.append(next_token)

                    # Decode and yield
                    token_text = tokenizer.decode(next_token)
                    if token_text:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Check stop conditions
                    if next_token in stop_tokens:
                        finish_reason = "stop"
                        break

                    last_token = next_token

                    if len(input_ids) + len(generated_ids) >= _app_state["model_config"].max_seq_len:
                        finish_reason = "length"
                        break
                else:
                    finish_reason = "length"

                # Send final chunk with finish_reason
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"

            except RuntimeError as e:
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"\n[ERROR] {e}"},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

            # Terminate stream
            yield "data: [DONE]\n\n"

    _fastapi_app = app
    return app


# =============================================================================
#  Server Initialization
# =============================================================================

def parse_layer_dims(dim_string: str) -> List[Tuple[int, int]]:
    """
    Parse layer dimensions from a string like "768:768,768:3072,3072:768".

    Each pair is "input_dim:output_dim" separated by commas.
    """
    dims: List[Tuple[int, int]] = []
    for pair in dim_string.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid layer dimension format: '{pair}'. "
                f"Expected 'input_dim:output_dim'."
            )
        in_dim = int(parts[0].strip())
        out_dim = int(parts[1].strip())
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(
                f"Layer dimensions must be > 0, got ({in_dim}, {out_dim})."
            )
        dims.append((in_dim, out_dim))
    return dims


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    """Build a ModelConfig from command-line arguments or env vars."""
    lib_path = args.lib_path or os.environ.get("AIOC_LIB_PATH", "")
    model_path = args.model_path or os.environ.get("AIOC_MODEL_PATH", "")
    model_name = args.model_name or os.environ.get("AIOC_MODEL_NAME", "ai-os-model")
    layer_dims_str = args.layer_dims or os.environ.get("AIOC_LAYER_DIMS", "")
    vocab_size = int(
        args.vocab_size or os.environ.get("AIOC_VOCAB_SIZE", "32000")
    )
    hidden_dim = int(
        args.hidden_dim or os.environ.get("AIOC_HIDDEN_DIM", "768")
    )

    if not lib_path:
        raise ValueError(
            "Library path is required. Set AIOC_LIB_PATH or use --lib-path."
        )
    if not model_path:
        raise ValueError(
            "Model path is required. Set AIOC_MODEL_PATH or use --model-path."
        )
    if not layer_dims_str:
        raise ValueError(
            "Layer dimensions are required. Set AIOC_LAYER_DIMS or use --layer-dims. "
            "Format: '768:768,768:3072,3072:768'"
        )

    layer_dims = parse_layer_dims(layer_dims_str)

    return ModelConfig(
        name=model_name,
        file_path=model_path,
        layer_dims=layer_dims,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )


def initialize_engine(
    config: ModelConfig,
    gear_level: int,
) -> Tuple[Any, InferencePipeline]:
    """
    Initialize the AIOCEngine, PerformanceGovernor, and InferencePipeline.

    Returns:
        Tuple of (engine, pipeline).
    """
    from engine_bridge import (
        AIOCEngine,
        PerformanceGovernor,
        create_engine_with_governor,
        detect_system_ram_mb,
    )

    # Detect system RAM
    system_ram_mb = detect_system_ram_mb()
    if system_ram_mb == 0:
        logger.warning(
            "Could not detect system RAM. Assuming 2048 MB. "
            "Install psutil for accurate detection: pip install psutil"
        )
        system_ram_mb = 2048

    logger.info(f"Detected system RAM: {system_ram_mb} MB")

    # Load the shared library
    lib_path = os.environ.get("AIOC_LIB_PATH", "")
    engine = AIOCEngine(lib_path)

    # Use PerformanceGovernor to configure based on gear level
    governor = PerformanceGovernor(system_total_ram_mb=system_ram_mb)
    gov_config = governor.apply_config(engine, gear_level=gear_level)

    logger.info(
        f"Gear {gear_level}: RAM budget = {gov_config['ram_budget_mb']} MB "
        f"({gov_config['ram_percent_of_system']}% of system), "
        f"block_size = {gov_config['block_size']}"
    )

    # Load the model
    engine.load_model(config.file_path)
    logger.info(
        f"Model loaded: {config.file_path} "
        f"({engine.total_layers} layers)"
    )

    # Create tokenizer
    tokenizer = CharacterTokenizer()

    # Create pipeline
    pipeline = InferencePipeline(
        engine=engine,
        model_config=config,
        tokenizer=tokenizer,
    )

    return engine, pipeline


# =============================================================================
#  CLI Entry Point
# =============================================================================

def main() -> None:
    """Parse arguments, initialize engine, and start the server."""
    parser = argparse.ArgumentParser(
        description="AI OS Core — OpenAI-Compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python api_server.py --lib-path ./libai_os_core.so \\\n"
            "    --model-path ./model.aioc \\\n"
            "    --layer-dims 768:768,768:3072,3072:768 \\\n"
            "    --vocab-size 32000 --hidden-dim 768 --gear 5\n"
            "\n"
            "  AIOC_LIB_PATH=./libai_os_core.so \\\n"
            "  AIOC_MODEL_PATH=./model.aioc \\\n"
            "  AIOC_LAYER_DIMS=768:768,768:3072,3072:768 \\\n"
            "  python api_server.py --gear 5 --port 8080\n"
        ),
    )
    parser.add_argument(
        "--lib-path", type=str, default="",
        help="Path to the shared library (libai_os_core.so / ai_os_core.dll). "
             "Env: AIOC_LIB_PATH",
    )
    parser.add_argument(
        "--model-path", type=str, default="",
        help="Path to the .aioc model file. Env: AIOC_MODEL_PATH",
    )
    parser.add_argument(
        "--model-name", type=str, default="",
        help="Model name (returned in API responses). Env: AIOC_MODEL_NAME",
    )
    parser.add_argument(
        "--layer-dims", type=str, default="",
        help="Layer dimensions: 'in1:out1,in2:out2,...'. "
             "Env: AIOC_LAYER_DIMS",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=0,
        help="Vocabulary size. Env: AIOC_VOCAB_SIZE (default: 32000)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=0,
        help="Hidden dimension. Env: AIOC_HIDDEN_DIM (default: 768)",
    )
    parser.add_argument(
        "--gear", type=int, default=5,
        help="Performance gear level (1-10). Default: 5 (balanced).",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server bind address. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server bind port. Default: 8000",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of uvicorn workers. Default: 1 (single-process). "
             "WARNING: >1 workers will each lock their own RAM budget.",
    )

    args = parser.parse_args()

    # --- Build model config and initialize engine -----------------------
    try:
        config = build_model_config(args)
    except ValueError as e:
        parser.error(str(e))
        return

    gear = max(1, min(10, args.gear))

    try:
        engine, pipeline = initialize_engine(config, gear)
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        sys.exit(1)

    # --- Store global state for the FastAPI app -------------------------
    _app_state["engine"] = engine
    _app_state["pipeline"] = pipeline
    _app_state["model_config"] = config
    _app_state["tokenizer"] = pipeline.tokenizer
    _app_state["engine_lock"] = asyncio.Lock()

    # --- Start the server ----------------------------------------------
    app = _get_app()

    import uvicorn

    logger.info(
        f"Starting AI OS Core API server on {args.host}:{args.port} "
        f"(gear={gear}, workers={args.workers})"
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
