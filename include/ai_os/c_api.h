/**
 * @file c_api.h
 * @brief AI OS Core Phase 5 — C API Header (extern "C" Facade + DLL Export)
 *
 * This header declares the public C interface for the AI OS Core inference
 * system. All functions are declared as extern "C" to prevent C++ name
 * mangling, and decorated with AIOC_API for proper DLL/SO symbol export
 * on Windows (MSVC) and Linux (GCC/Clang).
 *
 * DLL Export Strategy:
 *   - Windows (MSVC): __declspec(dllexport) — exports symbols into the DLL's
 *     export table so that ctypes.LoadLibrary() can resolve them by name.
 *   - Linux/POSIX: __attribute__((visibility("default"))) — ensures symbols
 *     are visible in the shared object (.so) even when compiled with
 *     -fvisibility=hidden.
 *
 * Design Principles:
 *   1. OPAQUE CONTEXT: All internal C++ objects (SovereignArena,
 *      SlidingWindowManager, AsyncPrefetcher, TernaryMathEngine) are
 *      hidden behind a single void* context handle. The caller never
 *      sees C++ types or headers.
 *
 *   2. ERROR CODES: Every function that can fail returns an int error
 *      code (0 = success, negative = error). Detailed error messages
 *      are retrieved via aioc_get_last_error(). C++ exceptions are
 *      caught inside the implementation and never cross the C boundary.
 *
 *   3. OWNERSHIP: The context is created by aioc_init_system() and
 *      destroyed by aioc_teardown(). The caller is responsible for
 *      calling teardown exactly once per context.
 *
 *   4. THREAD MODEL: The context is designed for two concurrent threads:
 *      - Main thread (Python): calls wait_layer, commit, process, retire
 *      - Background thread (internal): managed by AsyncPrefetcher
 *      Query functions (total_layers, etc.) are safe from either thread.
 *
 * Thread Safety:
 *   - Lifecycle functions (init, load, teardown) are NOT thread-safe.
 *     Call them from a single thread before starting inference.
 *   - Pipeline functions (wait, commit, process, retire) are safe for
 *     the main inference thread. The background prefetch thread is
 *     managed internally by AsyncPrefetcher.
 *   - Query functions are read-only and safe from any thread.
 *
 * @author AI OS Core Team
 * @version 1.2.0 (DLL Export Hotfix)
 * @since 2026-04-08
 * @license MIT
 */

#ifndef AIOC_C_API_H
#define AIOC_C_API_H

/* ===========================================================================
 *  Platform Export Macro
 *
 *  Ensures all public C API functions are properly exported from the shared
 *  library (DLL on Windows, .so on Linux). Without this, MSVC applies C++
 *  name mangling AND hides symbols from the export table, causing
 *  ctypes.LoadLibrary() to fail with "function not found".
 *
 *  Usage: AIOC_API is placed before every public function declaration.
 *
 *  Build System Note:
 *    - On Windows/MSVC: AIOS_CORE_EXPORTS is typically defined by CMake via
 *      target_compile_definitions(ai_os_core PRIVATE AIOS_CORE_EXPORTS).
 *    - For simplified builds, this macro always uses dllexport when building
 *      the DLL itself. Consumers (importing) should NOT define AIOS_CORE_EXPORTS.
 * ========================================================================== */

#if defined(_WIN32) || defined(_WIN64)
    #ifdef AIOS_CORE_EXPORTS
        // [FIX #3] نبني الـ DLL نفسها → export الـ symbols
        #define AIOC_API __declspec(dllexport)
    #else
        // [FIX #3] أي تطبيق تاني بيستخدم الـ header → import الـ symbols
        // كان قبل كده dllexport في الاتنين وده غلط
        #define AIOC_API __declspec(dllimport)
    #endif
#else
    #define AIOC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ===========================================================================
 *  Error Codes
 *
 *  All C API functions return one of these codes. Negative values indicate
 *  errors. The special value AIOC_OK (0) indicates success. Detailed
 *  human-readable error messages are available via aioc_get_last_error().
 *
 *  Error categories:
 *    -1x: Context / argument validation errors
 *    -2x: Memory subsystem errors (from Phase 1)
 *    -3x: I/O and prefetcher errors (from Phase 2)
 *    -4x: Computation errors (from Phase 3)
 *    -99: Unknown / unclassified error
 * ========================================================================== */

#define AIOC_OK                   0    ///< Success
#define AIOC_ERR_NULL_CONTEXT    -1    ///< Context pointer is NULL
#define AIOC_ERR_NULL_ARG        -2    ///< Required function argument is NULL
#define AIOC_ERR_INVALID_ARG     -3    ///< Invalid argument value (e.g. zero dim)
#define AIOC_ERR_ALLOCATION      -10   ///< OS-level memory allocation failed
#define AIOC_ERR_LOCK            -11   ///< OS-level memory locking failed
#define AIOC_ERR_CAPACITY        -12   ///< Arena capacity exhausted
#define AIOC_ERR_INVALID_LAYER   -13   ///< Layer size exceeds window / invalid ID
#define AIOC_ERR_WINDOW_OVERFLOW -14   ///< Ring buffer cannot advance
#define AIOC_ERR_IO              -20   ///< File I/O error
#define AIOC_ERR_PREFETCHER      -21   ///< AsyncPrefetcher internal error
#define AIOC_ERR_NOT_STARTED     -22   ///< Prefetcher not started
#define AIOC_ERR_NOT_LOADED      -23   ///< Model not loaded
#define AIOC_ERR_ENGINE          -30   ///< TernaryMathEngine computation error
#define AIOC_ERR_NO_ACTIVE_LAYER -31   ///< No active layer to process
#define AIOC_ERR_UNKNOWN         -99   ///< Unclassified error

/* ===========================================================================
 *  Lifecycle Functions
 * ========================================================================== */

/**
 * @brief Initialize the AI OS Core inference system.
 *
 * Creates a SovereignArena (locked RAM) and a SlidingWindowManager (ring
 * buffer) with the specified memory budget. The TernaryMathEngine is also
 * initialized with AVX2 auto-detection.
 *
 * @param ram_budget_bytes  Amount of physical RAM to lock, in bytes.
 *                         Will be page-aligned internally. Must be > 0.
 *
 * @return Opaque context pointer on success, or NULL on error.
 *         Call aioc_get_last_error(NULL) to retrieve the error message.
 *
 * @note On POSIX systems, the process may need CAP_IPC_LOCK capability
 *       or elevated RLIMIT_MEMLOCK to lock large amounts of memory.
 * @note On Windows, the process needs SeLockMemoryPrivilege.
 */
AIOC_API void* aioc_init_system(size_t ram_budget_bytes);

/**
 * @brief Load an AIOC model file and start background prefetching.
 *
 * Opens the specified AIOC file, validates its header, and starts the
 * background I/O thread that begins loading layers into the ring buffer.
 *
 * @param context    Valid context from aioc_init_system(). Must not be NULL.
 * @param file_path  Path to the .aioc model file. Must not be NULL.
 *
 * @return The same context pointer on success, or NULL on error.
 *         Call aioc_get_last_error(context) to retrieve the error message.
 *
 * @pre aioc_init_system() must have been called successfully.
 * @post The background prefetch thread is running. Layers can be waited for.
 */
AIOC_API void* aioc_load_model(void* context, const char* file_path);

/**
 * @brief Shut down the system and release all resources.
 *
 * Stops the background prefetch thread (if running), then destroys the
 * AsyncPrefetcher, SlidingWindowManager, SovereignArena, and the context
 * itself. All locked RAM is returned to the OS.
 *
 * This function is safe to call with a NULL context (no-op).
 * After this call, the context pointer is invalid and must not be used.
 *
 * @param context  Context to destroy, or NULL.
 */
AIOC_API void aioc_teardown(void* context);

/* ===========================================================================
 *  Inference Pipeline Functions
 *
 *  These functions implement the sequential inference loop:
 *    for each layer:
 *      1. aioc_wait_layer(ctx, layer_id)   — wait for BG thread to load
 *      2. aioc_commit_layer(ctx)           — mark layer as active (readable)
 *      3. aioc_process_layer(ctx, ...)     — compute Y = X ⊛ W
 *      4. aioc_retire_layer(ctx)           — mark layer as done, signal BG
 * ========================================================================== */

/**
 * @brief Block until the specified layer has been loaded into RAM.
 *
 * If the layer has already been fetched, returns immediately (zero latency).
 * Otherwise, blocks the calling thread until the background I/O thread
 * finishes loading the requested layer.
 *
 * @param context        Valid context with a loaded model.
 * @param layer_id       0-indexed layer identifier.
 *
 * @return AIOC_OK on success, negative error code on failure.
 */
AIOC_API int aioc_wait_layer(void* context, int32_t layer_id);

/**
 * @brief Commit the most recently fetched layer, making it the active layer.
 *
 * Transitions the layer from Preparing → Active state in the
 * SlidingWindowManager. After this call, the layer's data is available
 * for inference via aioc_process_layer().
 *
 * @param context  Valid context with a loaded model.
 *
 * @return AIOC_OK on success, negative error code on failure.
 *
 * @pre aioc_wait_layer() must have been called for the desired layer.
 * @post The layer is active and readable. Call aioc_process_layer() next.
 */
AIOC_API int aioc_commit_layer(void* context);

/**
 * @brief Process the active layer using the TernaryMathEngine.
 *
 * Performs ternary matrix-vector multiplication: Y = X ⊛ W
 * (multiply-free: only ADD/SUB, no multiplication).
 *
 * The weight data is read from the active layer in the SlidingWindowManager.
 * The caller provides input and output buffers.
 *
 * @param context     Valid context with an active (committed) layer.
 * @param inputs      Pointer to input activation vector (float32[input_dim]).
 *                    Must not be NULL.
 * @param outputs     Pointer to output activation vector (float32[output_dim]).
 *                    Must not be NULL. Will be overwritten with results.
 * @param input_dim   Number of input features (rows of weight matrix).
 * @param output_dim  Number of output features (columns of weight matrix).
 *
 * @return AIOC_OK on success, negative error code on failure.
 *
 * @pre aioc_commit_layer() must have been called successfully.
 * @post Output buffer contains the computation results.
 */
AIOC_API int aioc_process_layer(void* context, const float* inputs, float* outputs,
                                uint32_t input_dim, uint32_t output_dim);

/**
 * @brief Retire the active layer and signal the background thread.
 *
 * Transitions the layer from Active → Retired in the SlidingWindowManager
 * and notifies the background I/O thread to begin loading the next layer.
 *
 * @param context  Valid context with an active layer.
 *
 * @return AIOC_OK on success, negative error code on failure.
 *
 * @post The active layer's memory is available for reuse. The BG thread
 *       will begin loading the next layer.
 */
AIOC_API int aioc_retire_layer(void* context);

/* ===========================================================================
 *  Query Functions
 * ========================================================================== */

/**
 * @brief Get the path of the currently loaded model file.
 *
 * Returns the file path that was passed to the most recent successful
 * aioc_load_model() call. The returned pointer is valid until the next
 * C API call or until the context is destroyed. Do not free it.
 *
 * @param context  Valid context.
 * @return Model file path string, or empty string if no model is loaded.
 */
AIOC_API const char* aioc_get_model_path(void* context);

/**
 * @brief Get the total number of layers in the loaded model.
 *
 * @param context  Valid context (may or may not have a loaded model).
 * @return Number of layers, or 0 if no model is loaded or context is NULL.
 */
AIOC_API uint32_t aioc_total_layers(void* context);

/**
 * @brief Get the byte size of a specific layer.
 *
 * @param context   Valid context with a loaded model.
 * @param layer_id  0-indexed layer identifier.
 * @return Layer size in bytes, or 0 on error.
 */
AIOC_API uint64_t aioc_layer_size(void* context, uint32_t layer_id);

/**
 * @brief Get the last error message from a context, or from init.
 *
 * If context is non-NULL, returns the error message stored in that context.
 * If context is NULL, returns the error message from the most recent
 * failed aioc_init_system() call (thread-local storage).
 *
 * The returned pointer is valid until the next C API call on the same
 * thread/context. Do not free it.
 *
 * @param context  Context pointer, or NULL for init errors.
 * @return Human-readable error string, or empty string if no error.
 */
AIOC_API const char* aioc_get_last_error(void* context);

/**
 * @brief Get a human-readable description of the math engine's capabilities.
 *
 * Includes AVX2 support status, block size, and platform information.
 *
 * @param context  Valid context.
 * @return Info string (valid until next call). Empty string if context is NULL.
 */
AIOC_API const char* aioc_engine_info(void* context);

/**
 * @brief Check whether AVX2 SIMD is available on the current CPU.
 *
 * @param context  Valid context.
 * @return 1 if AVX2 is supported, 0 if not, -1 if context is NULL.
 */
AIOC_API int aioc_has_avx2(void* context);

/* ===========================================================================
 *  Pipeline Control Functions
 *
 *  These functions manage the inference pipeline lifecycle for multi-token
 *  (autoregressive) generation. After processing all layers once, the
 *  pipeline must be reset before the next forward pass.
 * ========================================================================== */

/**
 * @brief Reset the inference pipeline for a new forward pass.
 *
 * Stops the background prefetch thread (if running), recreates the
 * SlidingWindowManager over the same arena region, and prepares the
 * context for a new aioc_load_model() call.
 *
 * This is required for autoregressive token generation: after processing
 * all layers for one token, call reset_pipeline() before loading the
 * model again for the next token. The model file data in the OS page
 * cache makes subsequent loads significantly faster than the first.
 *
 * @param context  Valid context.
 *
 * @return AIOC_OK on success, negative error code on failure.
 *
 * @note The SovereignArena (locked RAM) is NOT destroyed — it is reused.
 *       Only the SlidingWindowManager slots and AsyncPrefetcher are reset.
 * @note After this call, you MUST call aioc_load_model() before inference.
 */
AIOC_API int aioc_reset_pipeline(void* context);

/**
 * @brief Reload the model and prepare the pipeline in one call.
 *
 * Convenience function that combines aioc_reset_pipeline() + aioc_load_model().
 * Stops the current pipeline, recreates the SWM, and starts prefetching
 * from the same model file that was previously loaded.
 *
 * @param context  Valid context with a previously loaded model.
 *
 * @return AIOC_OK on success, negative error code on failure.
 *
 * @pre aioc_load_model() must have been called at least once before.
 */
AIOC_API int aioc_reload_model(void* context);

/* ===========================================================================
 *  Engine Configuration Functions (used by PerformanceGovernor)
 * ========================================================================== */

/**
 * @brief Set the output-dimension block size for tiled computation.
 *
 * When output_dim > block_size, the TernaryMathEngine divides the
 * computation into L1-cache-friendly tiles. Set to 0 to disable blocking.
 *
 * Gear 1 (conservative) → block_size = 0 (disabled)
 * Gear 5 (balanced)     → block_size = 4096 (default)
 * Gear 10 (greedy)      → block_size = 32768 (maximum speed)
 *
 * @param context    Valid context.
 * @param block_size Block size in output elements (0 = disabled).
 *
 * @return AIOC_OK on success, negative error code on failure.
 */
AIOC_API int aioc_set_block_size(void* context, uint32_t block_size);

/**
 * @brief Get the current block size setting.
 *
 * @param context  Valid context.
 * @return Current block size, or 0 if context is NULL.
 */
AIOC_API uint32_t aioc_get_block_size(void* context);

/**
 * @brief Get the SovereignArena memory utilization as a percentage.
 *
 * @param context  Valid context.
 * @return Value between 0.0 and 100.0, or -1.0 if context is NULL.
 */
AIOC_API double aioc_arena_utilization(void* context);

#ifdef __cplusplus
}
#endif

#endif /* AIOC_C_API_H */
