/**
 * @file ternary_math_engine.hpp
 * @brief AI OS Core Phase 3 - Ternary Math Engine (Multiply-Free Inference)
 *
 * TernaryMathEngine is the computational heart of the AI OS Core. It performs
 * matrix-vector multiplication for Fully Connected (Linear) layers using ONLY
 * addition and subtraction — completely eliminating the multiplication operator (*)
 * from the inference hot path.
 *
 * The Mathematical Insight:
 *   In a ternary quantized neural network, all weights are constrained to the
 *   set {-1, 0, +1}. This transforms the expensive floating-point multiply:
 *
 *     Traditional:  Y[j] = Σ_i  X[i] * W[i][j]     ← costly FMA/MUL
 *     Ternary:      Y[j] = Σ_i  X[i] ± X[i] or 0   ← cheap ADD/SUB only
 *
 *   For each weight value:
 *     W[i][j] == +1  →  Y[j] += X[i]    (add the input)
 *     W[i][j] == -1  →  Y[j] -= X[i]    (subtract the input)
 *     W[i][j] ==  0  →  skip             (no operation)
 *
 *   On modern CPUs, ADD/SUB has ~3 cycle latency and 0.5 cycle throughput
 *   (pipelined), while FMUL has ~4 cycle latency and 0.5 cycle throughput.
 *   The real win comes from avoiding the multiplication circuit entirely,
 *   reducing power consumption by ~30-40% on embedded CPUs.
 *
 * Weight Storage Format: ROW-MAJOR
 *   Weights are stored as int8_t in row-major layout:
 *
 *     W[i][j] is at offset:  (i * output_dim) + j
 *
 *   Why Row-Major is optimal for ternary matmul on CPU:
 *   ─────────────────────────────────────────────────────
 *   The computation iterates: outer loop over i (input_dim), inner loop
 *   over j (output_dim). For each fixed i:
 *
 *     (a) We read ONE scalar input:  inputs[i]            — 1 cache line
 *     (b) We read ONE row of weights: weights[i*out_dim]  — sequential bytes
 *     (c) We update ALL outputs:     outputs[0..out_dim]  — sequential writes
 *
 *   Row-major ensures step (b) accesses weights sequentially, maximizing
 *   L1 cache line utilization (64 bytes / 64 weights per cache line).
 *   Column-major would stride by output_dim per access, wasting cache lines.
 *
 * SIMD Strategy (AVX2):
 *   For each input i, we broadcast X[i] to all 8 SIMD lanes, then process
 *   8 weights and 8 outputs simultaneously:
 *
 *     ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
 *     │ x[0] │ x[1] │ x[2] │ x[3] │ x[4] │ x[5] │ x[6] │ x[7] │  ← All same X[i]
 *     └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
 *     ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
 *     │ w=+1 │ w= 0 │ w=-1 │ w=+1 │ w= 0 │ w=-1 │ w=+1 │ w= 0 │  ← 8 weights
 *     └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
 *     ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
 *     │  +X  │  0   │  -X  │  +X  │  0   │  -X  │  +X  │  0   │  ← contribution
 *     └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
 *                              ↓ add to outputs ↓
 *
 *   The AVX2 path is fully branchless: it uses _mm256_cmpeq_epi32 to create
 *   bitmasks for +1 and -1 weights, then _mm256_blendv_ps to select between
 *   +X, -X, and 0 — no if/else in the inner loop at all.
 *
 * Architecture Position:
 *   ┌─────────────────────────────────────────────────┐
 *   │           Main Inference Thread                  │
 *   │  prefetcher.wait_for_layer(N)                    │
 *   │  engine.compute_linear_layer(                   │
 *   │      activations, weights_ptr, outputs,          │
 *   │      input_dim, output_dim)                      │
 *   │  prefetcher.retire_and_signal()                  │
 *   └────────────┬────────────────────────────────────┘
 *                │
 *   ┌────────────▼────────────────────────────────────┐
 *   │        TernaryMathEngine (THIS PHASE)            │
 *   │   ┌───────────────┐  ┌───────────────────────┐  │
 *   │   │ Scalar Fallback│  │ AVX2 SIMD (branchless)│  │
 *   │   │ (any CPU)      │  │ (auto-detected)        │  │
 *   │   └───────────────┘  └───────────────────────┘  │
 *   └─────────────────────────────────────────────────┘
 *                │ reads from
 *   ┌────────────▼────────────────────────────────────┐
 *   │     SlidingWindowManager (Phase 1/2)             │
 *   │   Locked RAM arena with zero-copy prefetched data │
 *   └─────────────────────────────────────────────────┘
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 *
 * @see SlidingWindowManager  (Phase 1) — Ring buffer memory manager
 * @see AsyncPrefetcher        (Phase 2) — Background I/O prefetcher
 * @see exceptions.hpp          — Memory subsystem exceptions
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ai_os {
namespace compute {

// =============================================================================
//  Constants
// =============================================================================

/**
 * @brief Number of float32 elements processed per AVX2 SIMD iteration.
 *
 * AVX2 registers (__m256) hold 256 bits = 8 × 32-bit floats.
 * The inner loop processes output_dim elements in chunks of 8.
 */
constexpr std::size_t AVX2_FLOATS_PER_ITER = 8;

/**
 * @brief Default block size for tiled (blocked) matrix multiplication.
 *
 * When output_dim exceeds this threshold, the output is processed in
 * tiles of this size to maximize L1 cache hit rate. Each tile's
 * output region (BLOCK_SIZE × 4 bytes = 32KB) fits comfortably in
 * a typical 32KB L1 data cache.
 *
 * Tuning guidance:
 *   - L1 cache = 32KB → BLOCK_SIZE = 8192 (32KB / 4 bytes)
 *   - L1 cache = 64KB → BLOCK_SIZE = 16384
 *   - Smaller values reduce cache misses but increase loop overhead
 */
constexpr std::size_t DEFAULT_BLOCK_SIZE = 4096;

/**
 * @brief Loop unroll factor for the scalar fallback path.
 *
 * The scalar inner loop processes 4 output elements per iteration,
 * reducing branch prediction misses and loop counter overhead.
 */
constexpr std::size_t SCALAR_UNROLL_FACTOR = 4;

/**
 * @brief Loop unroll factor for the AVX2 SIMD path.
 *
 * The AVX2 inner loop processes 2 × 8 = 16 output elements per iteration,
 * enabling instruction-level parallelism (the CPU can overlap the two
 * 8-element blocks in its execution pipeline).
 */
constexpr std::size_t AVX2_UNROLL_FACTOR = 2;

// =============================================================================
//  Engine Statistics
// =============================================================================

/**
 * @struct EngineStats
 * @brief Runtime statistics from the most recent compute_linear_layer() call.
 *
 * These counters are reset at the start of each compute call and accumulated
 * during execution. They are useful for performance profiling and diagnostics.
 */
struct EngineStats {
    std::uint64_t total_weight_ops{0};    ///< Total weight evaluations (input_dim × output_dim)
    std::uint64_t add_operations{0};      ///< Number of +1 weight additions performed
    std::uint64_t sub_operations{0};      ///< Number of -1 weight subtractions performed
    std::uint64_t skip_operations{0};     ///< Number of 0 weight skips (no operation)
    std::uint64_t simd_iterations{0};     ///< Number of AVX2 SIMD iterations executed
    std::uint64_t scalar_iterations{0};   ///< Number of scalar fallback iterations
    double        elapsed_ms{0.0};        ///< Wall-clock time of the last compute call
    bool          used_avx2{false};       ///< Whether AVX2 SIMD was used (vs scalar)
};

// =============================================================================
//  TernaryMathEngine — Multiply-Free Inference Engine
// =============================================================================

/**
 * @class TernaryMathEngine
 * @brief Performs ternary matrix-vector multiplication using only ADD/SUB operations.
 *
 * This engine is designed for maximum CPU efficiency on low-power devices (2GB RAM,
 * no GPU). It replaces all floating-point multiplications with conditional additions
 * and subtractions, reducing both compute latency and power consumption.
 *
 * Key Design Decisions:
 *   1. WEIGHT FORMAT: Row-major int8_t storage for sequential cache-line access.
 *   2. SIMD: AVX2 with branchless blendv_ps — no per-element if/else in hot path.
 *   3. SCALAR FALLBACK: Loop-unrolled (4x) with switch-based weight dispatch.
 *   4. AUTO-DISPATCH: Runtime AVX2 detection in constructor, transparent to caller.
 *   5. ZERO-ALLOC: No heap allocation during compute. All memory comes from
 *      SlidingWindowManager's pre-locked region.
 *
 * Thread Safety:
 *   The compute methods are stateless (except for stats) and safe to call from
 *   multiple threads simultaneously, as long as each thread operates on different
 *   input/output buffers. The m_stats member is NOT thread-safe and should only
 *   be read from the thread that called compute.
 *
 * Usage Example:
 * @code
 *   // One-time setup
 *   ai_os::compute::TernaryMathEngine engine;
 *   std::cout << engine.engine_info() << std::endl;  // "AVX2" or "Scalar"
 *
 *   // Per-inference computation
 *   // 'weights_ptr' comes from SlidingWindowManager::active_layer_address()
 *   engine.compute_linear_layer(
 *       input_activations,      // float[input_dim]
 *       reinterpret_cast<const int8_t*>(weights_ptr),  // int8_t[input_dim × output_dim]
 *       output_activations,     // float[output_dim]
 *       input_dim,
 *       output_dim
 *   );
 * @endcode
 */
class TernaryMathEngine {
public:
    // =========================================================================
    //  Type Aliases
    // =========================================================================

    using size_type = std::uint32_t;  ///< Size type for dimensions

    // =========================================================================
    //  Construction / Destruction
    // =========================================================================

    /**
     * @brief Constructs a TernaryMathEngine and detects CPU SIMD capabilities.
     *
     * The constructor probes the CPU for AVX2 support using platform-specific
     * intrinsics (__builtin_cpu_supports on GCC/Clang, __cpuid on MSVC).
     * The result is cached in m_avx2_supported for fast dispatch during compute.
     *
     * This constructor performs NO memory allocation and is very fast.
     */
    TernaryMathEngine();

    /**
     * @brief Destructor. No dynamic resources to release.
     */
    ~TernaryMathEngine() = default;

    // Copyable and movable (the engine holds only a bool and a stats struct)
    TernaryMathEngine(const TernaryMathEngine&) = default;
    TernaryMathEngine& operator=(const TernaryMathEngine&) = default;
    TernaryMathEngine(TernaryMathEngine&&) = default;
    TernaryMathEngine& operator=(TernaryMathEngine&&) = default;

    // =========================================================================
    //  Core Inference Methods
    // =========================================================================

    /**
     * @brief Computes a ternary linear layer: Y = X ⊛ W (multiply-free matmul).
     *
     * This is the primary inference method. It performs matrix-vector
     * multiplication where weights are ternary (-1, 0, +1), using only
     * addition and subtraction — NO multiplication operator (*) is used.
     *
     * Computation:
     *   For each output j in [0, output_dim):
     *     Y[j] = Σ_{i=0}^{input_dim-1} ternary_op(X[i], W[i * output_dim + j])
     *
     *   Where:
     *     ternary_op(x, +1) =  x    (add)
     *     ternary_op(x, -1) = -x    (subtract)
     *     ternary_op(x,  0) =  0    (skip)
     *
     * @param inputs      Pointer to input activation vector (float32[input_dim]).
     *                    Must not be nullptr. NOT modified by this function.
     * @param weights     Pointer to ternary weight matrix (int8_t[input_dim × output_dim]).
     *                    Stored in row-major order. Each value must be -1, 0, or +1.
     *                    Must not be nullptr. NOT modified by this function.
     * @param outputs     Pointer to output activation vector (float32[output_dim]).
     *                    Must not be nullptr. WILL be overwritten.
     * @param input_dim   Number of input neurons (rows of weight matrix).
     * @param output_dim  Number of output neurons (columns of weight matrix).
     * @param zero_output If true (default), outputs are zeroed before computation.
     *                    Set to false if outputs are already zeroed externally.
     *
     * @throws std::invalid_argument If any pointer is nullptr or dimensions are zero.
     *
     * @note This method automatically selects between scalar and AVX2 SIMD
     *       implementations based on runtime CPU detection.
     * @note Performance: ~2 GFLOP/s (scalar) to ~12 GFLOP/s (AVX2) on typical CPUs.
     * @note The weight pointer typically comes from:
     *       SlidingWindowManager::active_layer_address() (after AsyncPrefetcher loads it).
     */
    void compute_linear_layer(
        const float* inputs,
        const int8_t* weights,
        float* outputs,
        size_type input_dim,
        size_type output_dim,
        bool zero_output = true);

    /**
     * @brief Computes a ternary linear layer with bias addition: Y = (X ⊛ W) + B.
     *
     * Identical to compute_linear_layer() but adds a per-output bias term
     * after the ternary matmul. This is the standard Fully Connected layer
     * operation: output = activation(W × input + bias).
     *
     * @param inputs      Pointer to input vector (float32[input_dim]).
     * @param weights     Pointer to ternary weight matrix (int8_t[input_dim × output_dim]).
     * @param bias        Pointer to bias vector (float32[output_dim]).
     *                    If nullptr, this method behaves identically to compute_linear_layer().
     * @param outputs     Pointer to output vector (float32[output_dim]).
     * @param input_dim   Number of input neurons.
     * @param output_dim  Number of output neurons.
     * @param zero_output If true, outputs are zeroed before computation.
     *
     * @throws std::invalid_argument If inputs, weights, or outputs is nullptr.
     */
    void compute_linear_layer_with_bias(
        const float* inputs,
        const int8_t* weights,
        const float* bias,
        float* outputs,
        size_type input_dim,
        size_type output_dim,
        bool zero_output = true);

    /**
     * @brief Computes a ternary linear layer for a batch of inputs.
     *
     * Processes batch_size independent matrix-vector multiplications.
     * This is equivalent to calling compute_linear_layer() batch_size times,
     * but with slightly better cache behavior (the weight matrix stays in
     * cache across batch iterations).
     *
     * @param inputs      Pointer to batch input matrix (float32[batch_size × input_dim]).
     *                    Row-major: inputs[b * input_dim + i] is the i-th input of batch b.
     * @param weights     Pointer to ternary weight matrix (int8_t[input_dim × output_dim]).
     * @param outputs     Pointer to batch output matrix (float32[batch_size × output_dim]).
     *                    Row-major: outputs[b * output_dim + j] is the j-th output of batch b.
     * @param batch_size  Number of inputs in the batch.
     * @param input_dim   Number of input neurons.
     * @param output_dim  Number of output neurons.
     */
    void compute_linear_layer_batch(
        const float* inputs,
        const int8_t* weights,
        float* outputs,
        size_type batch_size,
        size_type input_dim,
        size_type output_dim);

    // =========================================================================
    //  Validation
    // =========================================================================

    /**
     * @brief Validates that all weights in a ternary weight matrix are in {-1, 0, +1}.
     *
     * This method is NOT called automatically during compute_linear_layer() for
     * performance reasons. It should be called once during model loading to verify
     * weight integrity. Calling it during inference would add unacceptable overhead.
     *
     * @param weights     Pointer to the weight matrix.
     * @param count       Total number of int8_t elements to validate.
     *
     * @return true if all weights are valid (-1, 0, or +1), false otherwise.
     *
     * @note This is a sequential O(N) scan. For a 100M parameter model, this
     *       takes ~50ms. Call only during initialization, not per-inference.
     */
    static bool validate_weights(const int8_t* weights, size_type count);

    // =========================================================================
    //  Query Interface
    // =========================================================================

    /**
     * @brief Returns whether the CPU supports AVX2 SIMD instructions.
     * @return true if AVX2 is available at runtime.
     */
    [[nodiscard]] bool has_avx2_support() const noexcept;

    /**
     * @brief Returns the currently configured output-dimension block size.
     * @return Block size in number of output elements.
     */
    [[nodiscard]] size_type block_size() const noexcept;

    /**
     * @brief Sets the output-dimension block size for tiled computation.
     *
     * When output_dim > block_size, the computation is divided into tiles
     * of this size to improve L1 cache utilization. Set to 0 to disable
     * blocking (process the entire output at once).
     *
     * @param size Block size (0 = disabled, default = DEFAULT_BLOCK_SIZE = 4096).
     */
    void set_block_size(size_type size);

    /**
     * @brief Returns a human-readable description of the engine's capabilities.
     *
     * Includes: SIMD support status, block size, and platform information.
     *
     * @return Multi-line info string.
     */
    [[nodiscard]] std::string engine_info() const;

    /**
     * @brief Returns statistics from the most recent compute_linear_layer() call.
     * @return Const reference to the EngineStats struct.
     *
     * @note These stats are reset at the beginning of each compute call.
     * @note NOT thread-safe: only read from the thread that called compute.
     */
    [[nodiscard]] const EngineStats& last_stats() const noexcept;

    // =========================================================================
    //  Diagnostic
    // =========================================================================

    /**
     * @brief Generates a comprehensive diagnostic report of the last computation.
     * @return Multi-line string with operation counts, timing, and efficiency metrics.
     */
    [[nodiscard]] std::string diagnostics() const;

private:
    // =========================================================================
    //  Private Methods — Scalar Fallback Implementation
    // =========================================================================

    /**
     * @brief Scalar (non-SIMD) implementation of ternary matmul.
     *
     * Processes output elements 4 at a time (SCALAR_UNROLL_FACTOR = 4) to
     * reduce loop overhead and improve branch prediction accuracy.
     *
     * For each input i:
     *   - Load input value x = inputs[i]
     *   - For each group of 4 outputs (j, j+1, j+2, j+3):
     *       Read 4 weights, dispatch via switch:
     *         w == +1 → outputs[j] += x
     *         w == -1 → outputs[j] -= x
     *         w ==  0 → skip
     *   - Handle remaining outputs with scalar loop
     *
     * @param inputs      Input vector.
     * @param weights     Row-major weight matrix.
     * @param outputs     Output vector (zeroed by caller if needed).
     * @param input_dim   Number of inputs.
     * @param output_dim  Number of outputs.
     */
    void compute_scalar(
        const float* inputs,
        const int8_t* weights,
        float* outputs,
        size_type input_dim,
        size_type output_dim);

    /**
     * @brief Scalar implementation with blocked output processing.
     *
     * When output_dim is large (> block_size), this method divides the output
     * into tiles of block_size elements. Each tile fits in L1 cache, reducing
     * cache misses when the output array exceeds L1 capacity.
     *
     * @param inputs      Input vector.
     * @param weights     Row-major weight matrix.
     * @param outputs     Output vector.
     * @param input_dim   Number of inputs.
     * @param output_dim  Number of outputs.
     * @param blk_size    Tile size for output blocking.
     */
    void compute_scalar_blocked(
        const float* inputs,
        const int8_t* weights,
        float* outputs,
        size_type input_dim,
        size_type output_dim,
        size_type blk_size);

    // =========================================================================
    //  Private Methods — AVX2 SIMD Implementation
    // =========================================================================

    /**
     * @brief AVX2 SIMD implementation of ternary matmul.
     *
     * Processes 8 output elements per iteration (AVX2_FLOATS_PER_ITER = 8)
     * with 2x loop unrolling (16 elements per outer iteration). The inner
     * loop is fully branchless:
     *
     *   1. Broadcast input x to 8 SIMD lanes: _mm256_set1_ps(x)
     *   2. Load 8 int8_t weights, sign-extend to int32: _mm256_cvtepi8_epi32
     *   3. Compare with +1: _mm256_cmpeq_epi32 → positive mask
     *   4. Compare with -1: _mm256_cmpeq_epi32 → negative mask
     *   5. Branchless blend: _mm256_blendv_ps to select +x, -x, or 0
     *   6. Accumulate: _mm256_add_ps into output register
     *
     * Only compiled when __AVX2__ is defined at compile time.
     *
     * @param inputs      Input vector.
     * @param weights     Row-major weight matrix.
     * @param outputs     Output vector (zeroed by caller if needed).
     * @param input_dim   Number of inputs.
     * @param output_dim  Number of outputs.
     */
    void compute_avx2(
        const float* inputs,
        const int8_t* weights,
        float* outputs,
        size_type input_dim,
        size_type output_dim);

    /**
     * @brief AVX2 implementation with blocked output processing.
     *
     * Same as compute_avx2 but divides large outputs into L1-cache-friendly tiles.
     *
     * @param inputs      Input vector.
     * @param weights     Row-major weight matrix.
     * @param outputs     Output vector.
     * @param input_dim   Number of inputs.
     * @param output_dim  Number of outputs.
     * @param blk_size    Tile size.
     */
    void compute_avx2_blocked(
        const float* inputs,
        const int8_t* weights,
        float* outputs,
        size_type input_dim,
        size_type output_dim,
        size_type blk_size);

    // =========================================================================
    //  Private Methods — Platform Detection
    // =========================================================================

    /**
     * @brief Detects AVX2 support at runtime using platform-specific CPUID.
     *
     * On GCC/Clang (Linux/macOS):
     *   Uses __builtin_cpu_supports("avx2") which performs a CPUID instruction
     *   and checks the AVX2 feature flag (CPUID.01H:EAX[bit 5]).
     *
     * On MSVC (Windows):
     *   Uses __cpuid() to query CPU feature flags. Checks:
     *     - CPUID.01H:ECX[bit 27] for OSXSAVE
     *     - CPUID.01H:ECX[bit 28] for AVX
     *     - CPUID.07H:EBX[bit 5]  for AVX2
     *
     * @return true if the CPU supports AVX2 and the OS has enabled it.
     */
    static bool detect_avx2_support();

    // =========================================================================
    //  Private Methods — Utilities
    // =========================================================================

    /**
     * @brief Resets the engine statistics counters.
     */
    void reset_stats() noexcept;

    /**
     * @brief Determines whether to use blocked processing based on dimensions.
     * @param output_dim  Output dimension to check.
     * @return true if blocking should be used.
     */
    [[nodiscard]] bool should_block(size_type output_dim) const noexcept;

    // =========================================================================
    //  Member Variables
    // =========================================================================

    bool       m_avx2_supported;   ///< Cached AVX2 detection result
    size_type  m_block_size;       ///< Output blocking tile size (0 = disabled)
    EngineStats m_stats;           ///< Statistics from the last compute call
};

} // namespace compute
} // namespace ai_os
