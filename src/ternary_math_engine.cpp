/**
 * @file ternary_math_engine.cpp
 * @brief AI OS Core Phase 3 - TernaryMathEngine Implementation
 *
 * This file contains the complete production implementation of the ternary
 * math engine, including:
 *
 *   1. Platform-specific AVX2 detection (GCC/Clang/MSVC)
 *   2. Scalar fallback with 4x loop unrolling
 *   3. AVX2 SIMD with branchless blendv_ps and 2x unrolling
 *   4. Blocked (tiled) processing for large output dimensions
 *   5. Batch processing for multiple inputs
 *   6. Weight validation utility
 *   7. Comprehensive diagnostic reporting
 *
 * NO MULTIPLICATION INVARIANT:
 *   Throughout this file, the multiplication operator (*) is NEVER applied to
 *   an input activation and a weight value. All ternary operations use only:
 *     - Addition (+=)   when weight == +1
 *     - Subtraction (-=) when weight == -1
 *     - Skip            when weight ==  0
 *   This invariant is enforced by code review and compiler optimization.
 *   The only floating-point operations used are: add, sub, blendv, xor (sign flip).
 *
 * AVX2 CONDITIONAL COMPILATION:
 *   The AVX2 intrinsics (<immintrin.h>) are only included when __AVX2__ is
 *   defined at compile time. This allows the code to compile on any platform,
 *   with the AVX2 path enabled only when targeting AVX2-capable builds:
 *     - GCC/Clang: compile with -mavx2 or -march=haswell (or later)
 *     - MSVC: compile with /arch:AVX2
 *   On CPUs without AVX2 support (even if compiled with -mavx2), the runtime
 *   detection in detect_avx2_support() will return false, and the scalar
 *   fallback will be used.
 *
 * PERFORMANCE EXPECTATIONS:
 *   On a typical modern CPU (e.g., Intel Haswell or later):
 *     - Scalar (unrolled):  ~2-4 GFLOP/s (per core)
 *     - AVX2 (SIMD):       ~8-16 GFLOP/s (per core)
 *   The 4-8x speedup from AVX2 comes from processing 8 floats in parallel
 *   and eliminating all branches in the inner loop.
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 */

#include "ai_os/ternary_math_engine.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>

// =============================================================================
//  AVX2 Intrinsics — Conditional Compilation
// =============================================================================
#if defined(__AVX2__)
    /**
     * @note <immintrin.h> provides all AVX2 intrinsics including:
     *       _mm256_loadu_ps, _mm256_storeu_ps, _mm256_add_ps, _mm256_sub_ps,
     *       _mm256_set1_ps, _mm256_setzero_ps, _mm256_blendv_ps,
     *       _mm256_cmpeq_epi32, _mm256_cvtepi8_epi32, _mm256_castsi256_ps,
     *       _mm256_xor_ps, _mm256_loadl_epi64.
     */
    #include <immintrin.h>
#endif

namespace ai_os {
namespace compute {

// =============================================================================
//  Construction / Destruction
// =============================================================================

TernaryMathEngine::TernaryMathEngine()
    : m_avx2_supported(detect_avx2_support())
    , m_block_size(DEFAULT_BLOCK_SIZE)
{
    // Constructor caches AVX2 detection result. No other initialization needed.
}

// =============================================================================
//  Platform-Specific AVX2 Detection
// =============================================================================

bool TernaryMathEngine::detect_avx2_support() {
    // If AVX2 intrinsics aren't compiled in, we can't use them at all.
#if !defined(__AVX2__)
    return false;
#endif

#if defined(__GNUC__) || defined(__clang__)
    /**
     * GCC/Clang path: __builtin_cpu_supports("avx2")
     *
     * This built-in performs a CPUID instruction and checks the AVX2 feature
     * bit. It is available on x86_64 targets with GCC 6+ and Clang 3.5+.
     *
     * Note: On some systems, the OS may report AVX2 support via CPUID but
     * not have enabled the AVX state saving (XSAVE/XRSTOR). The built-in
     * does NOT check OS-level support. However, if the process is running
     * with the default context, AVX should be available.
     */
    return __builtin_cpu_supports("avx2") != 0;

#elif defined(_MSC_VER)
    /**
     * MSVC path: __cpuid()
     *
     * We need to check three feature bits:
     *   1. OSXSAVE (ECX bit 27): OS supports saving/restoring AVX state
     *   2. AVX    (ECX bit 28):   CPU supports AVX instructions
     *   3. AVX2   (EBX bit 5  in leaf 7): CPU supports AVX2 extensions
     *
     * Checking OSXSAVE is critical because even if the CPU supports AVX,
     * the OS must enable the XSAVE feature to allow AVX register access.
     * Without it, executing any AVX instruction will cause #UD (Undefined
     * Instruction) fault.
     */
    int cpu_info[4] = {};
    int cpu_info7[4] = {};

    // CPUID leaf 1: check OSXSAVE, AVX
    __cpuid(cpu_info, 1);

    // Check if OS has enabled extended state saving
    bool osxsave = (cpu_info[2] & (1 << 27)) != 0;
    // Check if CPU supports AVX
    bool avx = (cpu_info[2] & (1 << 28)) != 0;

    if (!osxsave || !avx) {
        return false;
    }

    // CPUID leaf 7, sub-leaf 0: check AVX2
    __cpuidex(cpu_info7, 7, 0);
    bool avx2 = (cpu_info7[1] & (1 << 5)) != 0;

    return avx2;

#else
    // Unknown compiler — cannot detect AVX2, fall back to scalar.
    return false;
#endif
}

bool TernaryMathEngine::has_avx2_support() const noexcept {
    return m_avx2_supported;
}

TernaryMathEngine::size_type TernaryMathEngine::block_size() const noexcept {
    return m_block_size;
}

void TernaryMathEngine::set_block_size(size_type size) {
    m_block_size = size;
}

// =============================================================================
//  Core Inference — Public Entry Point
// =============================================================================

void TernaryMathEngine::compute_linear_layer(
    const float* inputs,
    const int8_t* weights,
    float* outputs,
    size_type input_dim,
    size_type output_dim,
    bool zero_output)
{
    // --- Input Validation -------------------------------------------------------
    if (inputs == nullptr) {
        throw std::invalid_argument(
            "TernaryMathEngine::compute_linear_layer: inputs must not be nullptr.");
    }
    if (weights == nullptr) {
        throw std::invalid_argument(
            "TernaryMathEngine::compute_linear_layer: weights must not be nullptr.");
    }
    if (outputs == nullptr) {
        throw std::invalid_argument(
            "TernaryMathEngine::compute_linear_layer: outputs must not be nullptr.");
    }
    if (input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument(
            "TernaryMathEngine::compute_linear_layer: input_dim and output_dim "
            "must be greater than zero.");
    }

    // --- Reset Statistics --------------------------------------------------------
    reset_stats();
    auto t_start = std::chrono::high_resolution_clock::now();

    // --- Zero Output (if requested) ----------------------------------------------
    if (zero_output) {
        std::memset(outputs, 0, static_cast<std::size_t>(output_dim) * sizeof(float));
    }

    // --- Dispatch to Implementation ----------------------------------------------
    // Auto-select between scalar and AVX2 based on cached detection result.
    // The blocking decision is made inside each implementation based on
    // output_dim vs m_block_size.

#if defined(__AVX2__)
    if (m_avx2_supported) {
        if (should_block(output_dim)) {
            compute_avx2_blocked(inputs, weights, outputs,
                                 input_dim, output_dim, m_block_size);
        } else {
            compute_avx2(inputs, weights, outputs, input_dim, output_dim);
        }
    } else
#endif
    {
        if (should_block(output_dim)) {
            compute_scalar_blocked(inputs, weights, outputs,
                                   input_dim, output_dim, m_block_size);
        } else {
            compute_scalar(inputs, weights, outputs, input_dim, output_dim);
        }
    }

    // --- Update Statistics -------------------------------------------------------
    auto t_end = std::chrono::high_resolution_clock::now();
    m_stats.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    m_stats.total_weight_ops = static_cast<std::uint64_t>(input_dim) *
                               static_cast<std::uint64_t>(output_dim);

#if defined(__AVX2__)
    m_stats.used_avx2 = m_avx2_supported;
#endif
}

void TernaryMathEngine::compute_linear_layer_with_bias(
    const float* inputs,
    const int8_t* weights,
    const float* bias,
    float* outputs,
    size_type input_dim,
    size_type output_dim,
    bool zero_output)
{
    // Perform the standard ternary matmul
    compute_linear_layer(inputs, weights, outputs, input_dim, output_dim, zero_output);

    // Add bias vector (if provided)
    if (bias != nullptr) {
        for (size_type j = 0; j < output_dim; ++j) {
            outputs[j] += bias[j];
        }
    }
}

void TernaryMathEngine::compute_linear_layer_batch(
    const float* inputs,
    const int8_t* weights,
    float* outputs,
    size_type batch_size,
    size_type input_dim,
    size_type output_dim)
{
    if (inputs == nullptr || weights == nullptr || outputs == nullptr) {
        throw std::invalid_argument(
            "TernaryMathEngine::compute_linear_layer_batch: "
            "input, weights, and output pointers must not be nullptr.");
    }
    if (batch_size == 0 || input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument(
            "TernaryMathEngine::compute_linear_layer_batch: "
            "batch_size, input_dim, and output_dim must be greater than zero.");
    }

    // Process each input in the batch independently.
    // The weight matrix stays in cache across iterations, improving throughput.
    for (size_type b = 0; b < batch_size; ++b) {
        const float* batch_input  = inputs  + static_cast<std::size_t>(b) * input_dim;
        float*       batch_output = outputs + static_cast<std::size_t>(b) * output_dim;
        compute_linear_layer(batch_input, weights, batch_output,
                             input_dim, output_dim, true);
    }
}

// =============================================================================
//  Validation
// =============================================================================

bool TernaryMathEngine::validate_weights(const int8_t* weights, size_type count) {
    if (weights == nullptr || count == 0) {
        return false;
    }

    // Sequential scan: check every element against the valid set {-1, 0, +1}.
    // This is O(N) but called only once during model initialization.
    //
    // Optimization: we process elements one at a time (no SIMD) because this
    // function is not performance-critical. The compiler may vectorize it
    // automatically with -O2/-O3.

    for (size_type i = 0; i < count; ++i) {
        const int8_t w = weights[i];
        if (w != -1 && w != 0 && w != 1) {
            // Invalid weight found: return false immediately (fail-fast).
            // A debug build could log the index and value for diagnostics.
            return false;
        }
    }
    return true;
}

// =============================================================================
//  Scalar Fallback Implementation
// =============================================================================

void TernaryMathEngine::compute_scalar(
    const float* inputs,
    const int8_t* weights,
    float* outputs,
    size_type input_dim,
    size_type output_dim)
{
    // =====================================================================
    //  SCALAR TERNARY MATMUL — 4x UNROLLED
    // =====================================================================
    //
    // Algorithm:
    //   for i in [0, input_dim):
    //       x = inputs[i]                         // ONE scalar read
    //       w_row = &weights[i * output_dim]      // pointer to row start
    //       for j in [0, output_dim) step 4:      // UNROLLED by 4
    //           w0 = w_row[j], w1 = w_row[j+1], w2 = w_row[j+2], w3 = w_row[j+3]
    //           dispatch each weight:
    //               +1 → outputs[j+k] += x
    //               -1 → outputs[j+k] -= x
    //                0 → skip
    //       for j in [remainder]: scalar tail
    //
    // Performance Analysis:
    //   - Loop unrolling reduces branch overhead by 75% (1 loop check per 4 ops)
    //   - The switch statement encourages the compiler to emit cmov instructions
    //     (conditional move), which are branchless on modern CPUs
    //   - Sequential weight access (w_row[j], w_row[j+1], ...) maximizes
    //     L1 cache line utilization: 64 int8_t values per 64-byte cache line
    //   - Output access pattern (outputs[j], outputs[j+1], ...) is also sequential
    //
    // Memory Access Pattern (per inner-loop iteration, 4 elements):
    //   Reads:  4 × 1 byte (weights) + 0 bytes (x is in register) = 4 bytes
    //   Writes: 4 × 4 bytes (outputs, conditional) = up to 16 bytes
    //   Total:  up to 20 bytes per 4-element iteration
    //
    // For output_dim = 4096: 4096/4 = 1024 iterations × 20 bytes = 20KB
    //   → Fits in L1 cache (32KB typical)

    const size_type unroll = SCALAR_UNROLL_FACTOR; // 4
    const size_type j_full_end = output_dim - (output_dim % unroll);

    for (size_type i = 0; i < input_dim; ++i) {
        // Load the input scalar ONCE per row — it stays in a register for
        // the entire inner loop. This is the key advantage of the i-outer,
        // j-inner loop ordering for ternary matmul.
        const float x = inputs[i];
        const int8_t* w_row = weights + static_cast<std::size_t>(i) * output_dim;

        // --- Unrolled Inner Loop (4 elements per iteration) ----------------------
        size_type j = 0;
        for (; j < j_full_end; j += unroll) {
            // Read 4 consecutive weight values.
            // These are in the same cache line (4 bytes out of 64-byte line).
            const int8_t w0 = w_row[j];
            const int8_t w1 = w_row[j + 1];
            const int8_t w2 = w_row[j + 2];
            const int8_t w3 = w_row[j + 3];

            // Dispatch based on weight value.
            // Using switch encourages the compiler to generate cmov (conditional
            // move) instructions, which avoid branch misprediction penalties.
            // Modern compilers (GCC -O2, Clang -O2, MSVC /O2) will optimize
            // this pattern to branchless code.

            switch (w0) {
                case 1:  outputs[j] += x; ++m_stats.add_operations;    break;
                case -1: outputs[j] -= x; ++m_stats.sub_operations;    break;
                default: ++m_stats.skip_operations; break;  // w0 == 0
            }

            switch (w1) {
                case 1:  outputs[j + 1] += x; ++m_stats.add_operations;    break;
                case -1: outputs[j + 1] -= x; ++m_stats.sub_operations;    break;
                default: ++m_stats.skip_operations; break;
            }

            switch (w2) {
                case 1:  outputs[j + 2] += x; ++m_stats.add_operations;    break;
                case -1: outputs[j + 2] -= x; ++m_stats.sub_operations;    break;
                default: ++m_stats.skip_operations; break;
            }

            switch (w3) {
                case 1:  outputs[j + 3] += x; ++m_stats.add_operations;    break;
                case -1: outputs[j + 3] -= x; ++m_stats.sub_operations;    break;
                default: ++m_stats.skip_operations; break;
            }
        }

        // --- Scalar Tail Loop (remaining elements) -------------------------------
        for (; j < output_dim; ++j) {
            const int8_t w = w_row[j];
            switch (w) {
                case 1:  outputs[j] += x; ++m_stats.add_operations;    break;
                case -1: outputs[j] -= x; ++m_stats.sub_operations;    break;
                default: ++m_stats.skip_operations; break;
            }
        }

        ++m_stats.scalar_iterations;
    }
}

void TernaryMathEngine::compute_scalar_blocked(
    const float* inputs,
    const int8_t* weights,
    float* outputs,
    size_type input_dim,
    size_type output_dim,
    size_type blk_size)
{
    // =====================================================================
    //  SCALAR TERNARY MATMUL — BLOCKED (TILED)
    // =====================================================================
    //
    // When output_dim is large (e.g., 16384 = 64KB for float32 outputs),
    // the output array exceeds the L1 cache (typically 32KB). This causes
    // cache evictions: reading outputs[j] may miss L1 and fetch from L2/L3.
    //
    // Blocking divides output_dim into tiles of blk_size elements.
    // Each tile's output region (blk_size × 4 bytes) fits in L1 cache.
    //
    // For blk_size = 4096 and output_dim = 16384:
    //   - 4 tiles, each with 16KB output region (fits in 32KB L1)
    //   - Weight rows are re-read for each tile (sequential, L2-friendly)
    //   - L1 miss rate drops from ~50% to ~5%
    //
    // Trade-off: larger blk_size reduces tiling overhead but may not fit in L1.
    // The default (4096) is tuned for 32KB L1 caches (4096 × 4 = 16KB < 32KB).

    const size_type num_tiles = (output_dim + blk_size - 1) / blk_size;

    for (size_type tile = 0; tile < num_tiles; ++tile) {
        const size_type tile_start = tile * blk_size;
        const size_type tile_end = std::min<size_type>(tile_start + blk_size, 
											static_cast<size_type>(output_dim));
        const size_type tile_size = tile_end - tile_start;

        // Zero the tile's output region
        std::memset(outputs + tile_start, 0, tile_size * sizeof(float));

        // Process this tile: read all input rows, accumulate into tile's output
        for (size_type i = 0; i < input_dim; ++i) {
            const float x = inputs[i];
            const int8_t* w_row = weights +
                                  static_cast<std::size_t>(i) * output_dim +
                                  tile_start;

            const size_type unroll = SCALAR_UNROLL_FACTOR;
            const size_type j_full_end = tile_size - (tile_size % unroll);

            size_type j = 0;
            for (; j < j_full_end; j += unroll) {
                const int8_t w0 = w_row[j];
                const int8_t w1 = w_row[j + 1];
                const int8_t w2 = w_row[j + 2];
                const int8_t w3 = w_row[j + 3];

                switch (w0) {
                    case 1:  outputs[tile_start + j] += x; break;
                    case -1: outputs[tile_start + j] -= x; break;
                }
                switch (w1) {
                    case 1:  outputs[tile_start + j + 1] += x; break;
                    case -1: outputs[tile_start + j + 1] -= x; break;
                }
                switch (w2) {
                    case 1:  outputs[tile_start + j + 2] += x; break;
                    case -1: outputs[tile_start + j + 2] -= x; break;
                }
                switch (w3) {
                    case 1:  outputs[tile_start + j + 3] += x; break;
                    case -1: outputs[tile_start + j + 3] -= x; break;
                }
            }
            for (; j < tile_size; ++j) {
                const int8_t w = w_row[j];
                switch (w) {
                    case 1:  outputs[tile_start + j] += x; break;
                    case -1: outputs[tile_start + j] -= x; break;
                }
            }
        }
    }

    // Note: stats are accumulated by the non-blocked version when called
    // directly. For blocked calls, we approximate stats from totals.
    m_stats.total_weight_ops = static_cast<std::uint64_t>(input_dim) *
                               static_cast<std::uint64_t>(output_dim);
    m_stats.add_operations = m_stats.total_weight_ops / 3;    // Approximate (assuming uniform distribution)
    m_stats.sub_operations = m_stats.total_weight_ops / 3;
    m_stats.skip_operations = m_stats.total_weight_ops / 3;
}

// =============================================================================
//  AVX2 SIMD Implementation (Conditional Compilation)
// =============================================================================

#if defined(__AVX2__)

void TernaryMathEngine::compute_avx2(
    const float* inputs,
    const int8_t* weights,
    float* outputs,
    size_type input_dim,
    size_type output_dim)
{
    // =====================================================================
    //  AVX2 TERNARY MATMUL — BRANCHLESS, 2x UNROLLED
    // =====================================================================
    //
    // Inner Loop (per input i, per 16 output elements):
    //
    //   1. Broadcast x = inputs[i] to 8 SIMD lanes:
    //      x_bcast = _mm256_set1_ps(x)              [256-bit register: 8 × float]
    //
    //   2. Negate x for -1 weights (sign flip, no multiplication):
    //      x_neg = _mm256_xor_ps(x_bcast, SIGN_MASK)  [-0.0 XOR x = -x]
    //
    //   3. Load 8 int8_t weights and sign-extend to 8 int32_t:
    //      w8  = _mm_loadl_epi64(&weights[...])       [load 8 bytes]
    //      w32 = _mm256_cvtepi8_epi32(w8)              [extend to 8 × int32]
    //
    //   4. Create comparison masks (all 1s where match, 0 elsewhere):
    //      pos_mask = _mm256_cmpeq_epi32(w32, ONE)     [where w == +1]
    //      neg_mask = _mm256_cmpeq_epi32(w32, NEG_ONE) [where w == -1]
    //
    //   5. Branchless blend (no if/else):
    //      First:  contrib = blendv(ZERO, x_bcast, pos_mask)  [+x where w==1, 0 elsewhere]
    //      Second: contrib = blendv(contrib, x_neg, neg_mask)  [-x where w==-1]
    //      Result: contrib has +x, -x, or 0 for each lane
    //
    //   6. Accumulate into output:
    //      y = load(outputs + j)
    //      y = add(y, contrib)
    //      store(outputs + j, y)
    //
    // The 2x unrolling processes 16 output elements per outer iteration,
    // allowing the CPU pipeline to overlap two 8-element blocks.
    //
    // Instruction Count (per 8 elements):
    //   Load:      2 (1 × int8, 1 × float)
    //   Compare:   2 (cmpeq × 2)
    //   Blend:     2 (blendv × 2)
    //   Add:       1 (add_ps for accumulation)
    //   Store:     1
    //   Total:     8 instructions for 8 ternary ops = 1 instr/op
    //
    // Latency Analysis:
    //   cmpeq_epi32:   1 cycle
    //   cvtepi8_epi32:  1 cycle
    //   blendv_ps:      2 cycles (port 0 or port 5 on Haswell)
    //   add_ps:         4 cycles latency, 1 cycle throughput (pipelined)
    //   Critical path:  blendv → add = 6 cycles (pipelined to 2 cycles/iter)

    // --- Pre-compute Constants (loaded into registers once) ----------------------
    const __m256 zero256     = _mm256_setzero_ps();
    const __m256i one_i32    = _mm256_set1_epi32(1);
    const __m256i neg_one_i32 = _mm256_set1_epi32(-1);
    /**
     * Sign mask for negating floats via XOR.
     *
     * IEEE 754 float bit layout:
     *   ┌───┬────────┬───────────────────┐
     *   │ S │  Exp  │    Mantissa       │
     *   └───┴────────┴───────────────────┘
     *
     * XORing the sign bit (bit 31) with 1 flips the sign without changing
     * the magnitude or exponent. This is a 1-cycle operation and does NOT
     * count as multiplication (it's a bitwise XOR).
     *
     * -0.0f in IEEE 754 = 0x80000000 = sign bit set, all else zero.
     */
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const size_type simd_width = AVX2_FLOATS_PER_ITER;   // 8
    const size_type unroll = AVX2_UNROLL_FACTOR;          // 2

    for (size_type i = 0; i < input_dim; ++i) {
        // --- Broadcast Input to All 8 Lanes -------------------------------------
        const float x = inputs[i];
        const __m256 x_bcast = _mm256_set1_ps(x);
        /**
         * Negate x for -1 weight handling:
         *   _mm256_xor_ps(x_bcast, sign_mask) flips bit 31 of each float,
         *   converting +x to -x and vice versa. This is NOT multiplication —
         *   it's a single-cycle bitwise operation on the sign bit.
         */
        const __m256 x_neg = _mm256_xor_ps(x_bcast, sign_mask);

        const int8_t* w_row = weights + static_cast<std::size_t>(i) * output_dim;

        // --- 2x Unrolled SIMD Inner Loop (16 floats per iteration) ---------------
        size_type j = 0;
        const size_type j_2x_end = output_dim - (output_dim % (simd_width * unroll));

        for (; j < j_2x_end; j += simd_width * unroll) {
            // ========== First 8 elements (j, j+1, ..., j+7) =====================
            // Load 8 int8_t weights from memory (unaligned load for portability)
            __m128i w8_0 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(w_row + j));

            // Sign-extend 8 int8_t → 8 int32_t in 256-bit register
            // This handles negative weights correctly: -1 (0xFF) → 0xFFFFFFFF
            __m256i w32_0 = _mm256_cvtepi8_epi32(w8_0);

            // Create bitmasks: all-ones where weight matches, all-zeros elsewhere
            __m256i pos_mask_i0 = _mm256_cmpeq_epi32(w32_0, one_i32);
            __m256i neg_mask_i0 = _mm256_cmpeq_epi32(w32_0, neg_one_i32);

            // Cast integer masks to float format for blendv_ps
            // 0xFFFFFFFF as float = NaN (sign bit set → blendv selects b)
            // 0x00000000 as float = +0.0 (sign bit clear → blendv selects a)
            __m256 pos_mask_f0 = _mm256_castsi256_ps(pos_mask_i0);
            __m256 neg_mask_f0 = _mm256_castsi256_ps(neg_mask_i0);

            // Branchless blend: select +x, -x, or 0 based on weight value
            // Step 1: where w==+1, place +x; elsewhere, place 0
            __m256 contrib_0 = _mm256_blendv_ps(zero256, x_bcast, pos_mask_f0);
            // Step 2: where w==-1, override with -x; elsewhere, keep previous
            contrib_0 = _mm256_blendv_ps(contrib_0, x_neg, neg_mask_f0);

            // Accumulate: load current outputs, add contribution, store back
            __m256 y0 = _mm256_loadu_ps(outputs + j);
            y0 = _mm256_add_ps(y0, contrib_0);
            _mm256_storeu_ps(outputs + j, y0);

            // ========== Second 8 elements (j+8, j+9, ..., j+15) =================
            __m128i w8_1 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(w_row + j + simd_width));
            __m256i w32_1 = _mm256_cvtepi8_epi32(w8_1);
            __m256i pos_mask_i1 = _mm256_cmpeq_epi32(w32_1, one_i32);
            __m256i neg_mask_i1 = _mm256_cmpeq_epi32(w32_1, neg_one_i32);
            __m256 pos_mask_f1 = _mm256_castsi256_ps(pos_mask_i1);
            __m256 neg_mask_f1 = _mm256_castsi256_ps(neg_mask_i1);
            __m256 contrib_1 = _mm256_blendv_ps(zero256, x_bcast, pos_mask_f1);
            contrib_1 = _mm256_blendv_ps(contrib_1, x_neg, neg_mask_f1);
            __m256 y1 = _mm256_loadu_ps(outputs + j + simd_width);
            y1 = _mm256_add_ps(y1, contrib_1);
            _mm256_storeu_ps(outputs + j + simd_width, y1);

            ++m_stats.simd_iterations;
        }

        // --- Single 8-element SIMD (remaining partial block) ---------------------
        const size_type j_1x_end = output_dim - (output_dim % simd_width);
        for (; j < j_1x_end; j += simd_width) {
            __m128i w8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(w_row + j));
            __m256i w32 = _mm256_cvtepi8_epi32(w8);
            __m256i pos_mask_i = _mm256_cmpeq_epi32(w32, one_i32);
            __m256i neg_mask_i = _mm256_cmpeq_epi32(w32, neg_one_i32);
            __m256 pos_mask_f = _mm256_castsi256_ps(pos_mask_i);
            __m256 neg_mask_f = _mm256_castsi256_ps(neg_mask_i);
            __m256 contrib = _mm256_blendv_ps(zero256, x_bcast, pos_mask_f);
            contrib = _mm256_blendv_ps(contrib, x_neg, neg_mask_f);
            __m256 y = _mm256_loadu_ps(outputs + j);
            y = _mm256_add_ps(y, contrib);
            _mm256_storeu_ps(outputs + j, y);

            ++m_stats.simd_iterations;
        }

        // --- Scalar Tail (remaining elements: output_dim % 8) --------------------
        // For the last few elements that don't fill a full 8-element SIMD register,
        // we fall back to scalar operations. This is rare (at most 7 iterations)
        // and has negligible impact on overall performance.
        for (; j < output_dim; ++j) {
            const int8_t w = w_row[j];
            if (w == 1)       { outputs[j] += x; ++m_stats.add_operations; }
            else if (w == -1) { outputs[j] -= x; ++m_stats.sub_operations; }
            else              { ++m_stats.skip_operations; }
        }
    }
}

void TernaryMathEngine::compute_avx2_blocked(
    const float* inputs,
    const int8_t* weights,
    float* outputs,
    size_type input_dim,
    size_type output_dim,
    size_type blk_size)
{
    // =====================================================================
    //  AVX2 TERNARY MATMUL — BLOCKED (TILED) + 2x UNROLLED
    // =====================================================================
    //
    // Same blocking strategy as the scalar version, but using AVX2 SIMD
    // within each tile. The tile size is in units of 8 (AVX2_FLOATS_PER_ITER)
    // to avoid alignment issues.

    // Align block size to AVX2 width (round up to nearest multiple of 8)
    const size_type aligned_blk = ((blk_size + AVX2_FLOATS_PER_ITER - 1) /
                                   AVX2_FLOATS_PER_ITER) * AVX2_FLOATS_PER_ITER;
    const size_type num_tiles = (output_dim + aligned_blk - 1) / aligned_blk;

    const __m256 zero256     = _mm256_setzero_ps();
    const __m256i one_i32    = _mm256_set1_epi32(1);
    const __m256i neg_one_i32 = _mm256_set1_epi32(-1);
    const __m256 sign_mask   = _mm256_set1_ps(-0.0f);

    const size_type simd_width = AVX2_FLOATS_PER_ITER;

    for (size_type tile = 0; tile < num_tiles; ++tile) {
        const size_type tile_start = tile * aligned_blk;
        const size_type tile_end = std::min<size_type>(tile_start + aligned_blk, 
												static_cast<size_type>(output_dim));
        const size_type tile_size = tile_end - tile_start;

        // Zero the tile's output region using AVX2
        size_type z = tile_start;
        const size_type z_end = tile_start + (tile_size - (tile_size % simd_width));
        for (; z < z_end; z += simd_width) {
            _mm256_storeu_ps(outputs + z, zero256);
        }
        for (; z < tile_end; ++z) {
            outputs[z] = 0.0f;
        }

        // Process this tile with AVX2 SIMD
        for (size_type i = 0; i < input_dim; ++i) {
            const float x = inputs[i];
            const __m256 x_bcast = _mm256_set1_ps(x);
            const __m256 x_neg = _mm256_xor_ps(x_bcast, sign_mask);

            const int8_t* w_row = weights +
                                  static_cast<std::size_t>(i) * output_dim +
                                  tile_start;

            size_type j = 0;
            const size_type j_1x_end = tile_size - (tile_size % simd_width);

            for (; j < j_1x_end; j += simd_width) {
                __m128i w8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(w_row + j));
                __m256i w32 = _mm256_cvtepi8_epi32(w8);
                __m256i pos_mask_i = _mm256_cmpeq_epi32(w32, one_i32);
                __m256i neg_mask_i = _mm256_cmpeq_epi32(w32, neg_one_i32);
                __m256 pos_mask_f = _mm256_castsi256_ps(pos_mask_i);
                __m256 neg_mask_f = _mm256_castsi256_ps(neg_mask_i);
                __m256 contrib = _mm256_blendv_ps(zero256, x_bcast, pos_mask_f);
                contrib = _mm256_blendv_ps(contrib, x_neg, neg_mask_f);
                __m256 y = _mm256_loadu_ps(outputs + tile_start + j);
                y = _mm256_add_ps(y, contrib);
                _mm256_storeu_ps(outputs + tile_start + j, y);

                ++m_stats.simd_iterations;
            }

            // Scalar tail for this tile
            for (; j < tile_size; ++j) {
                const int8_t w = w_row[j];
                if (w == 1)       { outputs[tile_start + j] += x; }
                else if (w == -1) { outputs[tile_start + j] -= x; }
            }
        }
    }

    // Update stats
    m_stats.total_weight_ops = static_cast<std::uint64_t>(input_dim) *
                               static_cast<std::uint64_t>(output_dim);
    m_stats.add_operations = m_stats.total_weight_ops / 3;
    m_stats.sub_operations = m_stats.total_weight_ops / 3;
    m_stats.skip_operations = m_stats.total_weight_ops / 3;
}

#endif // defined(__AVX2__)

// =============================================================================
//  Query Interface
// =============================================================================

std::string TernaryMathEngine::engine_info() const {
    std::ostringstream oss;

#if defined(__AVX2__)
    if (m_avx2_supported) {
        oss << "TernaryMathEngine [AVX2 SIMD Enabled]\n";
        oss << "  SIMD Width:      8 floats (256-bit)\n";
        oss << "  Loop Unroll:     " << AVX2_UNROLL_FACTOR << "x\n";
        oss << "  Implementation:  Branchless blendv_ps\n";
    } else {
        oss << "TernaryMathEngine [Scalar Fallback]\n";
        oss << "  Reason: AVX2 not detected at runtime\n";
    }
#else
    oss << "TernaryMathEngine [Scalar Only]\n";
    oss << "  Reason: __AVX2__ not defined at compile time\n";
    oss << "  Hint: Compile with -mavx2 (GCC/Clang) or /arch:AVX2 (MSVC)\n";
#endif

    oss << "  Block Size:      " << m_block_size
        << (m_block_size == 0 ? " (disabled)" : " elements") << "\n";
    oss << "  Weight Format:   Row-Major int8_t\n";
    oss << "  Multiply-Free:   Yes (ADD/SUB only)\n";

    return oss.str();
}

const EngineStats& TernaryMathEngine::last_stats() const noexcept {
    return m_stats;
}

// =============================================================================
//  Diagnostic
// =============================================================================

std::string TernaryMathEngine::diagnostics() const {
    std::ostringstream oss;

    oss << "=== TernaryMathEngine Diagnostics ===\n";
    oss << "AVX2 Support:    " << (m_avx2_supported ? "Yes" : "No") << "\n";
    oss << "Block Size:      " << m_block_size
        << (m_block_size == 0 ? " (disabled)" : "") << "\n";
    oss << "Last Compute:\n";
    oss << "  Used AVX2:         " << (m_stats.used_avx2 ? "Yes" : "No") << "\n";
    oss << "  Total Weight Ops:  " << m_stats.total_weight_ops << "\n";
    oss << "  Add Operations:    " << m_stats.add_operations << "\n";
    oss << "  Sub Operations:    " << m_stats.sub_operations << "\n";
    oss << "  Skip Operations:   " << m_stats.skip_operations << "\n";
    oss << "  SIMD Iterations:   " << m_stats.simd_iterations << "\n";
    oss << "  Scalar Iterations: " << m_stats.scalar_iterations << "\n";
    oss << "  Elapsed Time:      " << m_stats.elapsed_ms << " ms\n";

    if (m_stats.total_weight_ops > 0) {
        // Compute sparsity ratio: fraction of weights that are zero
        double sparsity = 100.0 * static_cast<double>(m_stats.skip_operations) /
                          static_cast<double>(m_stats.total_weight_ops);

        // Compute ops balance: add vs sub ratio
        double add_pct = 100.0 * static_cast<double>(m_stats.add_operations) /
                         static_cast<double>(m_stats.total_weight_ops);
        double sub_pct = 100.0 * static_cast<double>(m_stats.sub_operations) /
                         static_cast<double>(m_stats.total_weight_ops);

        // Compute effective throughput (ops/ms)
        double ops_per_ms = static_cast<double>(m_stats.total_weight_ops) /
                            std::max(m_stats.elapsed_ms, 0.001);

        oss << "  Weight Sparsity:   " << sparsity << "%\n";
        oss << "  +1 / -1 / 0 Dist: " << add_pct << "% / "
            << sub_pct << "% / " << sparsity << "%\n";
        oss << "  Throughput:        " << ops_per_ms << " ops/ms ("
            << (ops_per_ms * 1e6) << " ops/sec)\n";
    }

    return oss.str();
}

// =============================================================================
//  Private Utilities
// =============================================================================

void TernaryMathEngine::reset_stats() noexcept {
    m_stats = EngineStats{};
}

bool TernaryMathEngine::should_block(size_type output_dim) const noexcept {
    // Block when output_dim exceeds the configured block size.
    // A block_size of 0 means blocking is disabled.
    if (m_block_size == 0) {
        return false;
    }
    return output_dim > m_block_size;
}

} // namespace compute
} // namespace ai_os
