/**
 * @file c_api.cpp
 * @brief AI OS Core Phase 5 — C API Implementation (extern "C" Facade + DLL Export)
 *
 * This file implements the C interface declared in c_api.h. It wraps all
 * four C++ subsystems behind opaque void* handles and translates C++
 * exceptions into integer error codes that Python can safely inspect
 * without crashing.
 *
 * DLL Export Compliance:
 *   Every public function is decorated with AIOC_API (__declspec(dllexport)
 *   on Windows) and wrapped in extern "C" to prevent C++ name mangling.
 *   This ensures ctypes.LoadLibrary("libai_os_core.dll") can resolve
 *   every exported symbol by its exact C name (e.g. "aioc_init_system").
 *
 * Architecture:
 *   ┌────────────────────────────────────────────────────────────┐
 *   │                    Python (ctypes)                          │
 *   │   AIOCEngine  ·  PerformanceGovernor                       │
 *   └──────────────────────────┬─────────────────────────────────┘
 *                              │  C function calls (extern "C" + AIOC_API)
 *   ┌──────────────────────────▼─────────────────────────────────┐
 *   │                    c_api.cpp  (THIS FILE)                   │
 *   │   InferenceContext struct  ·  exception-to-errorcode       │
 *   └────────┬─────────────┬──────────────┬──────────────────────┘
 *            │             │              │
 *   ┌────────▼───┐  ┌──────▼──────┐  ┌───▼────────────┐
 *   │Sovereign   │  │SlidingWindow│  │TernaryMath     │
 *   │Arena       │  │Manager      │  │Engine          │
 *   │(Phase 1)   │  │(Phase 1)    │  │(Phase 3)       │
 *   └────────────┘  └──────┬──────┘  └────────────────┘
 *                         │
 *                  ┌──────▼──────┐
 *                  │AsyncPrefetch│
 *                  │er (Phase 2) │
 *                  └─────────────┘
 *
 * Exception Handling Strategy:
 *   Every extern "C" function wraps its body in a try/catch block that
 *   catches the full exception hierarchy from Phase 1 (MemoryError and
 *   all subclasses), plus std::exception as a catch-all. Each caught
 *   exception is translated to a specific negative error code and its
 *   what() message is stored in the context for retrieval via
 *   aioc_get_last_error(). This ensures that C++ exceptions NEVER
 *   propagate across the C/Python boundary.
 *
 * MSVC Strict Compliance:
 *   - No duplicate catch clauses in any try block.
 *   - Catch order: most-derived → least-derived → catch-all.
 *   - void* functions use AIOC_CATCH_NULLPTR (returns nullptr, not int).
 *   - int functions use AIOC_CATCH or AIOC_CATCH_STD_CUSTOM for overrides.
 *   - All functions decorated with AIOC_API for DLL export.
 *
 * Thread Safety:
 *   - aioc_init_system() and aioc_teardown() are NOT thread-safe.
 *     They must be called from a single thread (typically the main thread).
 *   - Pipeline functions delegate to thread-safe subsystems:
 *     * AsyncPrefetcher has its own internal mutex + condition variables.
 *     * SlidingWindowManager has its own recursive_mutex.
 *     * TernaryMathEngine is stateless (except for stats).
 *   - The InferenceContext itself is not protected by a mutex because the
 *     two threads (main + prefetch BG) access different members:
 *     Main thread: reads/writes via pipeline functions.
 *     BG thread: only accesses members through AsyncPrefetcher internals.
 *
 * @author AI OS Core Team
 * @version 1.2.0 (DLL Export Hotfix)
 * @since 2026-04-08
 * @license MIT
 */

#include "ai_os/c_api.h"
#include "ai_os/sovereign_arena.hpp"
#include "ai_os/sliding_window_manager.hpp"
#include "ai_os/async_prefetcher.hpp"
#include "ai_os/ternary_math_engine.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>

/* ===========================================================================
 *  InferenceContext — Internal aggregate holding all C++ subsystems.
 *
 *  This struct is allocated on the heap by aioc_init_system() and freed
 *  by aioc_teardown(). It is never exposed to callers — they only see
 *  an opaque void*.
 *
 *  Member destruction order (reverse of declaration) is critical:
 *    1. prefetcher  — holds reference to window_manager
 *    2. math_engine — stateless, no dependencies
 *    3. window_manager — uses memory from arena (raw pointer, no ownership)
 *    4. arena       — owns the locked RAM block
 *  std::unique_ptr guarantees this order automatically.
 * ========================================================================== */

struct InferenceContext {
    // --- Phase 1: Memory Subsystem ---
    std::unique_ptr<ai_os::memory::SovereignArena>       arena;
    std::unique_ptr<ai_os::memory::SlidingWindowManager>  window_manager;

    // --- Phase 2: I/O Subsystem ---
    std::unique_ptr<ai_os::io::AsyncPrefetcher> prefetcher;

    // --- Phase 3: Compute Subsystem ---
    ai_os::compute::TernaryMathEngine math_engine;

    // --- State ---
    std::string last_error;      ///< Last error message for this context
    std::string engine_info_str; ///< Cached engine info string
    std::string model_file_path; ///< Path of the currently loaded AIOC model
    bool        model_loaded{false};

    InferenceContext() {
        // Cache engine info immediately after construction
        engine_info_str = math_engine.engine_info();
    }

    ~InferenceContext() = default;

    // Non-copyable, non-movable
    InferenceContext(const InferenceContext&) = delete;
    InferenceContext& operator=(const InferenceContext&) = delete;
    InferenceContext(InferenceContext&&) = delete;
    InferenceContext& operator=(InferenceContext&&) = delete;
};

/* ===========================================================================
 *  Thread-Local Error Buffer for aioc_init_system()
 *
 *  When aioc_init_system() fails, no InferenceContext exists yet, so there
 *  is nowhere to store the error message. We use a thread_local string to
 *  capture the error. aioc_get_last_error(NULL) returns this string.
 *
 *  This is thread-safe because each thread has its own copy.
 * ========================================================================== */

static thread_local std::string tl_init_error;

/* ===========================================================================
 *  Helper: Store error message in context or thread-local buffer.
 *
 *  Centralizes the if-null-check pattern used by every catch block.
 *  Declared noexcept so it can be called from catch handlers safely.
 * ========================================================================== */

static void store_error(InferenceContext* ctx, const char* msg) noexcept {
    if (ctx) {
        ctx->last_error = msg;
    } else {
        tl_init_error = msg;
    }
}

/* ===========================================================================
 *  Exception Translation Macros
 *
 *  MSVC Strict Compliance Rules:
 *    1. No duplicate catch types within the same try block.
 *    2. Derived exceptions MUST be caught before their base classes.
 *    3. void*-returning functions MUST NOT return int error codes.
 *
 *  Macro Architecture:
 *    - AIOC_CATCH_TYPE(ctx, Type, code)  — single catch handler (int return)
 *    - AIOC_CATCH_TYPE_PTR(ctx, Type)   — single catch handler (nullptr return)
 *    - AIOC_CATCH_MEMORY(ctx)           — all MemoryError subclasses + base
 *    - AIOC_CATCH_STD(ctx)              — standard library exception chain
 *    - AIOC_CATCH_STD_CUSTOM(ctx, ia, le, re) — with override codes
 *    - AIOC_CATCH(ctx)                  — complete default chain (int return)
 *    - AIOC_CATCH_NULLPTR(ctx)          — complete chain (nullptr return)
 *
 *  Catch Order (most-derived → least-derived → catch-all):
 *    1. MemoryAllocationError   (most specific memory error)
 *    2. MemoryLockError
 *    3. ArenaCapacityExhaustedError
 *    4. ArenaNotInitializedError
 *    5. InvalidLayerSizeError
 *    6. WindowOverflowError
 *    7. AlignmentError
 *    8. MemoryError             (base of all above)
 *    9. std::invalid_argument   (derives from std::logic_error)
 *   10. std::logic_error        (base of std::invalid_argument)
 *   11. std::runtime_error      (sibling of std::logic_error)
 *   12. std::bad_alloc          (direct child of std::exception)
 *   13. std::exception          (base of 9-12)
 *   14. catch (...)             (catch-all)
 * ========================================================================== */

// --- Individual catch handler for int-returning functions ---
#define AIOC_CATCH_TYPE(ctx_ptr, ExceptionType, errcode)                       \
    catch (const ExceptionType& e) {                                          \
        store_error(ctx_ptr, e.what());                                        \
        return errcode;                                                       \
    }

// --- Individual catch handler for void*-returning functions ---
#define AIOC_CATCH_TYPE_PTR(ctx_ptr, ExceptionType)                            \
    catch (const ExceptionType& e) {                                          \
        store_error(ctx_ptr, e.what());                                        \
        return nullptr;                                                       \
    }

// --- Catch-all for int-returning functions ---
#define AIOC_CATCH_ALL(ctx_ptr, errcode)                                       \
    catch (...) {                                                             \
        store_error(ctx_ptr, "Unknown C++ exception");                        \
        return errcode;                                                       \
    }

// --- Catch-all for void*-returning functions ---
#define AIOC_CATCH_ALL_PTR(ctx_ptr)                                            \
    catch (...) {                                                             \
        store_error(ctx_ptr, "Unknown C++ exception");                        \
        return nullptr;                                                       \
    }

// --- Memory error chain (all specific subclasses, then base) ---
#define AIOC_CATCH_MEMORY(ctx_ptr)                                             \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::MemoryAllocationError,             \
                    AIOC_ERR_ALLOCATION)                                       \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::MemoryLockError,                   \
                    AIOC_ERR_LOCK)                                             \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::ArenaCapacityExhaustedError,       \
                    AIOC_ERR_CAPACITY)                                         \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::ArenaNotInitializedError,          \
                    AIOC_ERR_CAPACITY)                                         \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::InvalidLayerSizeError,             \
                    AIOC_ERR_INVALID_LAYER)                                    \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::WindowOverflowError,               \
                    AIOC_ERR_WINDOW_OVERFLOW)                                  \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::AlignmentError,                    \
                    AIOC_ERR_INVALID_ARG)                                      \
    AIOC_CATCH_TYPE(ctx_ptr, ai_os::memory::MemoryError,                       \
                    AIOC_ERR_UNKNOWN)

// --- Standard library chain (default error codes) ---
#define AIOC_CATCH_STD(ctx_ptr)                                               \
    AIOC_CATCH_TYPE(ctx_ptr, std::invalid_argument, AIOC_ERR_INVALID_ARG)      \
    AIOC_CATCH_TYPE(ctx_ptr, std::logic_error, AIOC_ERR_UNKNOWN)               \
    AIOC_CATCH_TYPE(ctx_ptr, std::runtime_error, AIOC_ERR_UNKNOWN)             \
    AIOC_CATCH_TYPE(ctx_ptr, std::bad_alloc, AIOC_ERR_ALLOCATION)              \
    AIOC_CATCH_TYPE(ctx_ptr, std::exception, AIOC_ERR_UNKNOWN)                 \
    AIOC_CATCH_ALL(ctx_ptr, AIOC_ERR_UNKNOWN)

// --- Standard library chain with custom error codes for specific types ---
//     ia_code = code for std::invalid_argument
//     le_code = code for std::logic_error
//     re_code = code for std::runtime_error
#define AIOC_CATCH_STD_CUSTOM(ctx_ptr, ia_code, le_code, re_code)             \
    AIOC_CATCH_TYPE(ctx_ptr, std::invalid_argument, ia_code)                  \
    AIOC_CATCH_TYPE(ctx_ptr, std::logic_error, le_code)                        \
    AIOC_CATCH_TYPE(ctx_ptr, std::runtime_error, re_code)                      \
    AIOC_CATCH_TYPE(ctx_ptr, std::bad_alloc, AIOC_ERR_ALLOCATION)              \
    AIOC_CATCH_TYPE(ctx_ptr, std::exception, AIOC_ERR_UNKNOWN)                 \
    AIOC_CATCH_ALL(ctx_ptr, AIOC_ERR_UNKNOWN)

// --- Complete default chain for int-returning functions ---
#define AIOC_CATCH(ctx_ptr)                                                    \
    AIOC_CATCH_MEMORY(ctx_ptr)                                                \
    AIOC_CATCH_STD(ctx_ptr)

// --- Complete chain for void*-returning functions (returns nullptr) ---
#define AIOC_CATCH_NULLPTR(ctx_ptr)                                            \
    AIOC_CATCH_TYPE_PTR(ctx_ptr, ai_os::memory::MemoryError)                  \
    AIOC_CATCH_TYPE_PTR(ctx_ptr, std::exception)                              \
    AIOC_CATCH_ALL_PTR(ctx_ptr)

/* ===========================================================================
 *  Helper: Validate context pointer and cast.
 * ========================================================================== */

static inline InferenceContext* get_ctx(void* context) noexcept {
    return static_cast<InferenceContext*>(context);
}

/* ===========================================================================
 *  Lifecycle Functions
 *
 *  All functions are decorated with AIOC_API and wrapped in extern "C"
 *  to ensure proper DLL export and C-linkage symbol names on MSVC.
 * ========================================================================== */

extern "C" {

AIOC_API void* aioc_init_system(size_t ram_budget_bytes) {
    tl_init_error.clear();

    if (ram_budget_bytes == 0) {
        tl_init_error = "aioc_init_system: ram_budget_bytes must be > 0.";
        return nullptr;
    }

    try {
        // Step 1: Create the SovereignArena — allocates and locks physical RAM.
        // The constructor page-aligns the capacity automatically.
        auto ctx = std::make_unique<InferenceContext>();
        ctx->arena = std::make_unique<ai_os::memory::SovereignArena>(ram_budget_bytes);

        // Step 2: Allocate the full arena capacity for the SlidingWindowManager.
        // The bump allocator gives us a contiguous region that spans the entire
        // locked RAM block. SovereignArena::allocate is O(1) with zero fragmentation.
        const auto actual_cap = ctx->arena->capacity();
        void* region = ctx->arena->allocate(actual_cap);

        if (!region) {
            ctx->last_error = "SovereignArena::allocate() returned nullptr for capacity="
                           + std::to_string(actual_cap);
            tl_init_error = ctx->last_error;
            return nullptr;
        }

        // Step 3: Create the SlidingWindowManager (ring buffer) over the region.
        // The SWM treats the region as a circular buffer for layer data.
        ctx->window_manager = std::make_unique<ai_os::memory::SlidingWindowManager>(
            region, actual_cap
        );

        // Step 4: TernaryMathEngine is already default-constructed in InferenceContext.
        // It has performed AVX2 detection in its constructor.

        // Transfer ownership to the caller.
        return static_cast<void*>(ctx.release());

    } AIOC_CATCH_NULLPTR(nullptr)

    // Unreachable — the macro always returns. But suppress compiler warning.
    return nullptr;
}

/* ---------------------------------------------------------------------- */

AIOC_API void* aioc_load_model(void* context, const char* file_path) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_load_model: context is NULL.";
        return nullptr;
    }
    if (!file_path) {
        ctx->last_error = "aioc_load_model: file_path is NULL.";
        return nullptr;
    }
    if (!ctx->window_manager) {
        ctx->last_error = "aioc_load_model: window_manager not initialized. "
                         "Call aioc_init_system() first.";
        return nullptr;
    }

    // Prevent double-loading: stop any existing prefetcher first.
    if (ctx->prefetcher) {
        try {
            ctx->prefetcher->stop();
        } catch (...) {
            // Ignore errors during cleanup of the old prefetcher
        }
        ctx->prefetcher.reset();
        ctx->model_loaded = false;
    }

    try {
        // Create the AsyncPrefetcher and bind it to the SlidingWindowManager.
        // The constructor opens the file, validates the AIOC header, and
        // parses the layer size table. It does NOT start the thread yet.
        ctx->prefetcher = std::make_unique<ai_os::io::AsyncPrefetcher>(
            *ctx->window_manager,
            std::string(file_path)
        );

        // Start the background I/O thread. It immediately begins loading
        // the first layer from disk into the ring buffer.
        // Store the model file path for later use (reload, query, etc.)
        ctx->model_file_path = std::string(file_path);

        ctx->prefetcher->start();
        ctx->model_loaded = true;

        return context;  // Return same context on success (for chaining)

    } AIOC_CATCH_NULLPTR(ctx)

    return nullptr;
}

/* ---------------------------------------------------------------------- */

AIOC_API void aioc_teardown(void* context) {
    if (!context) {
        return;  // Safe no-op
    }

    InferenceContext* ctx = get_ctx(context);

    // Step 1: Stop the background prefetch thread (if running).
    // This must happen BEFORE destroying the window_manager, because
    // the prefetcher holds a reference to it. AsyncPrefetcher::stop()
    // joins the background thread, so this may block briefly.
    if (ctx->prefetcher) {
        try {
            ctx->prefetcher->stop();
        } catch (...) {
            // Suppress all exceptions during teardown.
            // We must not throw from a cleanup path.
        }
    }

    // Step 2: Destroy the InferenceContext.
    // unique_ptr destruction order (reverse of declaration):
    //   prefetcher    → stops and joins BG thread, closes file handle
    //   math_engine   → trivial (no heap resources)
    //   window_manager→ releases slot bookkeeping (no memory deallocation)
    //   arena         → munmaps/VirtualFrees the locked RAM block
    delete ctx;
}

/* ===========================================================================
 *  Inference Pipeline Functions
 * ========================================================================== */

AIOC_API int aioc_wait_layer(void* context, int32_t layer_id) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_wait_layer: context is NULL.";
        return AIOC_ERR_NULL_CONTEXT;
    }
    if (!ctx->model_loaded || !ctx->prefetcher) {
        ctx->last_error = "aioc_wait_layer: no model loaded. Call aioc_load_model() first.";
        return AIOC_ERR_NOT_LOADED;
    }

    try {
        ctx->prefetcher->wait_for_layer(layer_id);
        return AIOC_OK;
    }
    // Memory errors (specific subclasses before base)
    AIOC_CATCH_MEMORY(ctx)
    // Standard errors — runtime_error maps to PREFETCHER error
    AIOC_CATCH_STD_CUSTOM(ctx,
        AIOC_ERR_INVALID_ARG,   // std::invalid_argument
        AIOC_ERR_UNKNOWN,       // std::logic_error
        AIOC_ERR_PREFETCHER)    // std::runtime_error  ← OVERRIDE
}

/* ---------------------------------------------------------------------- */

AIOC_API int aioc_commit_layer(void* context) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_commit_layer: context is NULL.";
        return AIOC_ERR_NULL_CONTEXT;
    }
    if (!ctx->model_loaded || !ctx->prefetcher) {
        ctx->last_error = "aioc_commit_layer: no model loaded.";
        return AIOC_ERR_NOT_LOADED;
    }

    try {
        ctx->prefetcher->commit_fetched_layer();
        return AIOC_OK;
    }
    // Memory errors (specific subclasses before base)
    AIOC_CATCH_MEMORY(ctx)
    // Standard errors — logic_error maps to INVALID_LAYER error
    AIOC_CATCH_STD_CUSTOM(ctx,
        AIOC_ERR_INVALID_ARG,   // std::invalid_argument
        AIOC_ERR_INVALID_LAYER, // std::logic_error  ← OVERRIDE
        AIOC_ERR_UNKNOWN)       // std::runtime_error

    return AIOC_ERR_UNKNOWN;
}

/* ---------------------------------------------------------------------- */

AIOC_API int aioc_process_layer(void* context, const float* inputs, float* outputs,
                                uint32_t input_dim, uint32_t output_dim) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_process_layer: context is NULL.";
        return AIOC_ERR_NULL_CONTEXT;
    }
    if (!inputs) {
        ctx->last_error = "aioc_process_layer: inputs pointer is NULL.";
        return AIOC_ERR_NULL_ARG;
    }
    if (!outputs) {
        ctx->last_error = "aioc_process_layer: outputs pointer is NULL.";
        return AIOC_ERR_NULL_ARG;
    }
    if (input_dim == 0 || output_dim == 0) {
        ctx->last_error = "aioc_process_layer: input_dim and output_dim must be > 0 (got "
                         + std::to_string(input_dim) + ", " + std::to_string(output_dim) + ").";
        return AIOC_ERR_INVALID_ARG;
    }
    if (!ctx->window_manager) {
        ctx->last_error = "aioc_process_layer: window_manager not initialized.";
        return AIOC_ERR_NULL_CONTEXT;
    }

    try {
        // Retrieve the active layer's data address from the SlidingWindowManager.
        // This points to pinned RAM where the AsyncPrefetcher loaded the weights.
        const void* weights_ptr = ctx->window_manager->active_layer_address();
        if (!weights_ptr) {
            ctx->last_error = "aioc_process_layer: no active layer in the sliding window. "
                             "Call aioc_commit_layer() before aioc_process_layer().";
            return AIOC_ERR_NO_ACTIVE_LAYER;
        }

        // Cast the void* to int8_t* — ternary weights are stored as signed bytes.
        const int8_t* ternary_weights = static_cast<const int8_t*>(weights_ptr);

        // Delegate to TernaryMathEngine for multiply-free matmul: Y = X ⊛ W.
        // zero_output=true ensures outputs are cleared before accumulation.
        ctx->math_engine.compute_linear_layer(
            inputs,
            ternary_weights,
            outputs,
            input_dim,
            output_dim,
            true  // zero_output
        );

        return AIOC_OK;

    }
    // Memory errors (specific subclasses before base)
    AIOC_CATCH_MEMORY(ctx)
    // Standard errors — invalid_argument maps to ENGINE error
    AIOC_CATCH_STD_CUSTOM(ctx,
        AIOC_ERR_ENGINE,        // std::invalid_argument  ← OVERRIDE
        AIOC_ERR_UNKNOWN,       // std::logic_error
        AIOC_ERR_UNKNOWN)       // std::runtime_error

    return AIOC_ERR_UNKNOWN;
}

/* ---------------------------------------------------------------------- */

AIOC_API int aioc_retire_layer(void* context) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_retire_layer: context is NULL.";
        return AIOC_ERR_NULL_CONTEXT;
    }
    if (!ctx->model_loaded || !ctx->prefetcher) {
        ctx->last_error = "aioc_retire_layer: no model loaded.";
        return AIOC_ERR_NOT_LOADED;
    }

    try {
        ctx->prefetcher->retire_and_signal();
        return AIOC_OK;
    }
    // Memory errors (specific subclasses before base)
    AIOC_CATCH_MEMORY(ctx)
    // Standard errors — logic_error maps to INVALID_LAYER error
    AIOC_CATCH_STD_CUSTOM(ctx,
        AIOC_ERR_INVALID_ARG,   // std::invalid_argument
        AIOC_ERR_INVALID_LAYER, // std::logic_error  ← OVERRIDE
        AIOC_ERR_UNKNOWN)       // std::runtime_error

    return AIOC_ERR_UNKNOWN;
}

/* ===========================================================================
 *  Query Functions
 * ========================================================================== */

AIOC_API const char* aioc_get_model_path(void* context) {
    if (!context) {
        return "Null context";
    }
    InferenceContext* ctx = get_ctx(context);
    return ctx->model_file_path.c_str();
}

/* ---------------------------------------------------------------------- */

AIOC_API uint32_t aioc_total_layers(void* context) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx || !ctx->prefetcher) {
        return 0;
    }
    try {
        return ctx->prefetcher->total_layers();
    } catch (...) {
        return 0;
    }
}

/* ---------------------------------------------------------------------- */

AIOC_API uint64_t aioc_layer_size(void* context, uint32_t layer_id) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx || !ctx->prefetcher) {
        return 0;
    }
    try {
        return ctx->prefetcher->layer_size(layer_id);
    } catch (...) {
        return 0;
    }
}

/* ---------------------------------------------------------------------- */

AIOC_API const char* aioc_get_last_error(void* context) {
    if (!context) {
        // Context is NULL — return the thread-local init error.
        // If tl_init_error is empty, return a descriptive default.
        if (tl_init_error.empty()) {
            return "No error";
        }
        return tl_init_error.c_str();
    }
    InferenceContext* ctx = get_ctx(context);
    return ctx->last_error.c_str();
}

/* ---------------------------------------------------------------------- */

AIOC_API const char* aioc_engine_info(void* context) {
    if (!context) {
        return "Null context";
    }
    InferenceContext* ctx = get_ctx(context);
    return ctx->engine_info_str.c_str();
}

/* ---------------------------------------------------------------------- */

AIOC_API int aioc_has_avx2(void* context) {
    if (!context) {
        return -1;
    }
    InferenceContext* ctx = get_ctx(context);
    return ctx->math_engine.has_avx2_support() ? 1 : 0;
}

/* ===========================================================================
 *  Pipeline Control Functions
 * ========================================================================== */

AIOC_API int aioc_reset_pipeline(void* context) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_reset_pipeline: context is NULL.";
        return AIOC_ERR_NULL_CONTEXT;
    }
    if (!ctx->arena) {
        ctx->last_error = "aioc_reset_pipeline: arena not initialized.";
        return AIOC_ERR_NULL_CONTEXT;
    }

    try {
        // Step 1: Stop the background prefetch thread (if running).
        // The AsyncPrefetcher holds a reference to the SWM, so we must
        // stop it before destroying the SWM.
        if (ctx->prefetcher) {
            ctx->prefetcher->stop();
            ctx->prefetcher.reset();
        }

        // Step 2: Recreate the SlidingWindowManager over the same arena.
        // The SovereignArena (locked RAM) is NOT destroyed — it is reused.
        // This clears all layer slots and resets the write head to offset 0.
        const auto cap = ctx->arena->capacity();
        void* region = ctx->arena->base();
        ctx->window_manager = std::make_unique<ai_os::memory::SlidingWindowManager>(
            region, cap
        );

        // Step 3: Mark as unloaded (caller must call aioc_load_model next).
        ctx->model_loaded = false;

        return AIOC_OK;

    } AIOC_CATCH(ctx)

    return AIOC_ERR_UNKNOWN;
}

/* ---------------------------------------------------------------------- */

AIOC_API int aioc_reload_model(void* context) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_reload_model: context is NULL.";
        return AIOC_ERR_NULL_CONTEXT;
    }
    if (ctx->model_file_path.empty()) {
        ctx->last_error = "aioc_reload_model: no model file path stored. "
                         "Call aioc_load_model() at least once first.";
        return AIOC_ERR_NOT_LOADED;
    }

    // Step 1: Reset the pipeline (stop prefetcher, recreate SWM)
    int reset_err = aioc_reset_pipeline(context);
    if (reset_err != AIOC_OK) {
        return reset_err;
    }

    // Step 2: Load the model using the stored file path
    try {
        ctx->prefetcher = std::make_unique<ai_os::io::AsyncPrefetcher>(
            *ctx->window_manager,
            ctx->model_file_path
        );
        ctx->prefetcher->start();
        ctx->model_loaded = true;
        return AIOC_OK;
    }
    // Memory errors (specific subclasses before base)
    AIOC_CATCH_MEMORY(ctx)
    // Standard errors — invalid_argument → INVALID_ARG, runtime_error → IO
    AIOC_CATCH_STD_CUSTOM(ctx,
        AIOC_ERR_INVALID_ARG,   // std::invalid_argument
        AIOC_ERR_UNKNOWN,       // std::logic_error
        AIOC_ERR_IO)            // std::runtime_error  ← OVERRIDE

    return AIOC_ERR_UNKNOWN;
}

/* ===========================================================================
 *  Engine Configuration Functions
 * ========================================================================== */

AIOC_API int aioc_set_block_size(void* context, uint32_t block_size) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        tl_init_error = "aioc_set_block_size: context is NULL.";
        return AIOC_ERR_NULL_CONTEXT;
    }

    try {
        ctx->math_engine.set_block_size(static_cast<ai_os::compute::TernaryMathEngine::size_type>(block_size));
        // Update cached engine info since block_size changed
        ctx->engine_info_str = ctx->math_engine.engine_info();
        return AIOC_OK;
    } AIOC_CATCH(ctx)

    return AIOC_ERR_UNKNOWN;
}

/* ---------------------------------------------------------------------- */

AIOC_API uint32_t aioc_get_block_size(void* context) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx) {
        return 0;
    }
    return static_cast<uint32_t>(ctx->math_engine.block_size());
}

/* ---------------------------------------------------------------------- */

AIOC_API double aioc_arena_utilization(void* context) {
    InferenceContext* ctx = get_ctx(context);
    if (!ctx || !ctx->arena) {
        return -1.0;
    }
    return ctx->arena->utilization_percent();
}

} /* extern "C" */
