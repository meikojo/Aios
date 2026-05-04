/**
 * @file exceptions.hpp
 * @brief AI OS Core - Custom Exception Hierarchy for Memory Subsystem
 *
 * This header defines a comprehensive hierarchy of exception types for the
 * AI OS Core memory management layer. Each exception type corresponds to a
 * specific failure mode, enabling precise error handling and diagnostics.
 *
 * Hierarchy:
 *   std::runtime_error
 *     └── MemoryError (base for all memory subsystem errors)
 *           ├── MemoryAllocationError     - OS-level allocation failure
 *           ├── MemoryLockError           - OS-level memory pinning failure
 *           ├── ArenaCapacityExhaustedError - Attempt to exceed hard limit
 *           ├── ArenaNotInitializedError  - Arena used before initialization
 *           ├── InvalidLayerSizeError     - Layer size exceeds window capacity
 *           ├── WindowOverflowError       - Sliding window cannot advance
 *           └── AlignmentError            - Allocation alignment violation
 *
 * Design Rationale:
 *   - Granular exception types allow callers to catch specific failures
 *     without inspecting error messages (which may be localized later).
 *   - All exceptions carry a human-readable what() string AND a numeric
 *     error code for programmatic handling.
 *   - The base class provides a common interface for logging frameworks.
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 */

#pragma once

#include <stdexcept>
#include <string>
#include <cstdint>
#include <utility>

namespace ai_os {
namespace memory {

// =============================================================================
//  Base Exception
// =============================================================================

/**
 * @class MemoryError
 * @brief Base class for all AI OS Core memory subsystem exceptions.
 *
 * Extends std::runtime_error with an additional integer error code field.
 * All specific memory exceptions inherit from this class, enabling callers
 * to catch either specific types or the entire family with a single handler.
 *
 * Usage Example:
 * @code
 *   try {
 *       SovereignArena arena(500 * 1024 * 1024);
 *   } catch (const MemoryError& e) {
 *       std::cerr << "Memory subsystem error [" << e.code() << "]: " << e.what() << "\n";
 *   }
 * @endcode
 */
class MemoryError : public std::runtime_error {
public:
    /**
     * @brief Constructs a MemoryError with a descriptive message and error code.
     * @param message Human-readable description of the error.
     * @param code    Numeric error code for programmatic handling.
     */
    explicit MemoryError(const std::string& message, std::int32_t code = 0)
        : std::runtime_error(message), m_code(code) {}

    /**
     * @brief Constructs a MemoryError with message and code via move semantics.
     * @param message Rvalue reference to the descriptive message.
     * @param code    Numeric error code for programmatic handling.
     */
    explicit MemoryError(std::string&& message, std::int32_t code = 0)
        : std::runtime_error(std::move(message)), m_code(code) {}

    /**
     * @brief Destructor. Virtual to ensure proper cleanup in derived classes.
     */
    ~MemoryError() override = default;

    /**
     * @brief Returns the numeric error code associated with this exception.
     * @return Error code integer (0 means unspecified).
     */
    [[nodiscard]] std::int32_t code() const noexcept { return m_code; }

private:
    std::int32_t m_code; ///< Numeric error code for programmatic dispatch
};

// =============================================================================
//  Allocation Errors
// =============================================================================

/**
 * @class MemoryAllocationError
 * @brief Thrown when the operating system fails to allocate the requested memory.
 *
 * This indicates a fundamental resource exhaustion condition. On Windows, this
 * corresponds to VirtualAlloc returning NULL. On POSIX, this corresponds to
 * mmap returning MAP_FAILED. The operating system error code (GetLastError()
 * or errno) is captured in the error code field.
 */
class MemoryAllocationError : public MemoryError {
public:
    /**
     * @brief Constructs with the requested size and OS error code.
     * @param requested_bytes The number of bytes that were requested.
     * @param os_error_code   The OS-level error code (GetLastError/errno).
     */
    explicit MemoryAllocationError(std::size_t requested_bytes, std::int32_t os_error_code = 0)
        : MemoryError(
            "Failed to allocate " + std::to_string(requested_bytes) +
            " bytes from the operating system. OS error code: " + std::to_string(os_error_code),
            os_error_code)
    {}
};

// =============================================================================
//  Locking Errors
// =============================================================================

/**
 * @class MemoryLockError
 * @brief Thrown when the OS fails to pin (lock) memory in physical RAM.
 *
 * Memory locking prevents the operating system's virtual memory manager from
 * paging the allocated region to disk. This is critical for AI inference
 * workloads where page faults would cause unacceptable latency spikes.
 *
 * Common causes:
 *   - The process exceeds its RLIMIT_MEMLOCK limit (POSIX)
 *   - Insufficient privileges (CAP_IPC_LOCK on Linux)
 *   - The requested region exceeds the system's lockable memory quota
 *
 * Recovery suggestion: The caller may attempt to reduce the requested arena
 * size or elevate process privileges before retrying.
 */
class MemoryLockError : public MemoryError {
public:
    /**
     * @brief Constructs with the size of the region that failed to lock.
     * @param lock_bytes     The number of bytes that were to be locked.
     * @param os_error_code  The OS-level error code (GetLastError/errno).
     */
    explicit MemoryLockError(std::size_t lock_bytes, std::int32_t os_error_code = 0)
        : MemoryError(
            "Failed to lock " + std::to_string(lock_bytes) +
            " bytes in physical RAM. The OS denied the memory pinning request. "
            "Possible causes: exceeded RLIMIT_MEMLOCK, insufficient privileges, "
            "or system lockable memory quota exhausted. OS error code: " +
            std::to_string(os_error_code),
            os_error_code)
    {}
};

// =============================================================================
//  Arena Errors
// =============================================================================

/**
 * @class ArenaCapacityExhaustedError
 * @brief Thrown when an allocation request exceeds the arena's remaining capacity.
 *
 * The SovereignArena enforces a hard upper bound on total memory usage. Once
 * this limit is reached, no further allocations are permitted. This is by
 * design: the arena must never exceed its pre-committed physical memory budget,
 * regardless of runtime conditions.
 *
 * The error message includes the requested size, the remaining capacity, and
 * the total arena size to assist with debugging and capacity planning.
 */
class ArenaCapacityExhaustedError : public MemoryError {
public:
    /**
     * @brief Constructs with detailed capacity breakdown.
     * @param requested_bytes  The allocation size that was rejected.
     * @param remaining_bytes  The bytes still available in the arena.
     * @param total_capacity   The total capacity of the arena.
     */
    explicit ArenaCapacityExhaustedError(
        std::size_t requested_bytes,
        std::size_t remaining_bytes,
        std::size_t total_capacity)
        : MemoryError(
            "Arena capacity exhausted: requested " + std::to_string(requested_bytes) +
            " bytes but only " + std::to_string(remaining_bytes) +
            " bytes remain (arena total: " + std::to_string(total_capacity) + " bytes).",
            static_cast<std::int32_t>(total_capacity - remaining_bytes))
    {}
};

/**
 * @class ArenaNotInitializedError
 * @brief Thrown when operations are attempted on an uninitialized arena.
 *
 * This guards against use-after-move, use-after-release, and programming
 * errors where the arena is accessed before successful construction.
 */
class ArenaNotInitializedError : public MemoryError {
public:
    /**
     * @brief Constructs with a description of the invalid operation.
     * @param operation The name of the operation that was attempted.
     */
    explicit ArenaNotInitializedError(const std::string& operation)
        : MemoryError(
            "Operation '" + operation + "' attempted on an uninitialized or "
            "released SovereignArena. Ensure the arena was constructed successfully.",
            -1)
    {}
};

// =============================================================================
//  Sliding Window Errors
// =============================================================================

/**
 * @class InvalidLayerSizeError
 * @brief Thrown when a layer's size exceeds the total window capacity.
 *
 * A single layer must fit within the SlidingWindowManager's managed region.
 * If a layer is larger than the entire arena, it cannot be loaded at all,
 * regardless of how many other layers have been freed. This is an unrecoverable
 * configuration error that the caller must address (e.g., by using a larger
 * arena or a more quantized model).
 */
class InvalidLayerSizeError : public MemoryError {
public:
    /**
     * @brief Constructs with the layer size and available capacity.
     * @param layer_size  The size of the layer that was rejected.
     * @param capacity    The total capacity of the sliding window.
     */
    explicit InvalidLayerSizeError(std::size_t layer_size, std::size_t capacity)
        : MemoryError(
            "Layer size (" + std::to_string(layer_size) +
            " bytes) exceeds total sliding window capacity (" +
            std::to_string(capacity) + " bytes). Cannot load this layer.",
            static_cast<std::int32_t>(layer_size))
    {}
};

/**
 * @class WindowOverflowError
 * @brief Thrown when the sliding window cannot advance without overwriting active data.
 *
 * This occurs when the ring buffer has wrapped around and the next write position
 * would collide with data that is still being read (the active layer). The caller
 * must wait for the active layer to be retired before advancing, or increase the
 * arena size to accommodate more concurrent layers.
 */
class WindowOverflowError : public MemoryError {
public:
    /**
     * @brief Constructs with details about the overflow condition.
     * @param active_layer_id  The ID of the layer that is blocking advancement.
     * @param required_offset  The offset where the next write needs to occur.
     * @param active_offset    The offset where the active layer begins.
     * @param active_size      The size of the active layer that is blocking.
     */
    explicit WindowOverflowError(
        std::uint32_t active_layer_id,
        std::size_t required_offset,
        std::size_t active_offset,
        std::size_t active_size)
        : MemoryError(
            "Window overflow: cannot advance write head to offset " +
            std::to_string(required_offset) + " because active layer " +
            std::to_string(active_layer_id) + " still occupies offset " +
            std::to_string(active_offset) + " (size: " +
            std::to_string(active_size) + " bytes).",
            static_cast<std::int32_t>(active_layer_id))
    {}
};

// =============================================================================
//  Alignment Errors
// =============================================================================

/**
 * @class AlignmentError
 * @brief Thrown when an allocation request has an invalid alignment.
 *
 * Alignment must be a power of two and greater than zero. Non-power-of-two
 * alignments are not supported by standard aligners and would lead to
 * undefined behavior.
 */
class AlignmentError : public MemoryError {
public:
    /**
     * @brief Constructs with the invalid alignment value.
     * @param alignment The alignment value that was rejected.
     */
    explicit AlignmentError(std::size_t alignment)
        : MemoryError(
            "Invalid alignment: " + std::to_string(alignment) +
            ". Alignment must be a power of two and greater than zero.",
            static_cast<std::int32_t>(alignment))
    {}
};

} // namespace memory
} // namespace ai_os
