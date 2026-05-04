/**
 * @file sovereign_arena.hpp
 * @brief AI OS Core Phase 1 - Sovereign Memory Arena
 *
 * SovereignArena implements a "sovereign memory arena" — a contiguous region of
 * physical memory that is allocated from the operating system and permanently
 * pinned (locked) in RAM. Once locked, the OS virtual memory manager is
 * forbidden from paging any portion of this region to disk, guaranteeing
 * deterministic access times critical for real-time AI inference.
 *
 * Architectural Role:
 *   SovereignArena is the foundational memory primitive of the AI OS Core.
 *   It sits at the lowest level of the memory hierarchy:
 *
 *     OS Physical RAM
 *       └── SovereignArena (locked, contiguous)
 *             └── SlidingWindowManager (ring buffer)
 *                   └── Layer allocations (circular overwrite)
 *
 * Key Properties:
 *   1. HARD LIMIT: After construction, the arena will never hold more than
 *      `capacity()` bytes. Any allocation that would exceed this limit is
 *      rejected with ArenaCapacityExhaustedError — no exceptions.
 *
 *   2. ZERO PAGING: The entire region is locked via mlock/VirtualLock. This
 *      is verified during construction; if locking fails, the arena is not
 *      created and MemoryLockError is thrown.
 *
 *   3. BUMP ALLOCATION: Internal allocation uses a simple bump pointer. This
 *      is O(1) per allocation with zero fragmentation. Deallocation is not
 *      supported (the arena is designed for single-use then reset semantics).
 *
 *   4. RAII COMPLIANT: The arena acquires resources in the constructor and
 *      releases them in the destructor. It is non-copyable and non-movable
 *      to prevent accidental double-free or use-after-move bugs.
 *
 * Thread Safety:
 *   This class is NOT thread-safe by design. External synchronization (e.g.,
 *   std::mutex) is required if multiple threads access the same arena. This
 *   is intentional: the overhead of internal locking would penalize the
 *   single-threaded inference path.
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 *
 * @see SlidingWindowManager - The ring buffer layer built on top of this arena.
 * @see exceptions.hpp     - Exception types thrown by this class.
 */

#pragma once

#include "ai_os/platform.hpp"
#include "ai_os/exceptions.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace ai_os {
namespace memory {

class SovereignArena {
public:
    // =========================================================================
    //  Type Aliases
    // =========================================================================

    using size_type       = std::size_t;   ///< Unsigned size type for byte counts
    using pointer         = void*;         ///< Opaque pointer to allocated memory
    using const_pointer   = const void*;   ///< Const opaque pointer to memory

    // =========================================================================
    //  Lifetime Management
    // =========================================================================

    /**
     * @brief Constructs a SovereignArena and locks the specified capacity in physical RAM.
     *
     * This constructor performs the following operations atomically:
     *   1. Validates that `capacity_bytes` is non-zero and page-aligned.
     *   2. Allocates `capacity_bytes` of contiguous virtual memory from the OS
     *      using VirtualAlloc (Windows) or mmap (POSIX).
     *   3. Locks the entire allocation in physical RAM using VirtualLock (Windows)
     *      or mlock (POSIX) to prevent paging.
     *   4. Initializes the internal bump allocator to offset 0.
     *
     * If step 2 or step 3 fails, all previously acquired resources are released
     * and the appropriate exception is thrown (MemoryAllocationError or
     * MemoryLockError respectively). The arena is guaranteed to be in a clean
     * state (no leaked resources) if the constructor throws.
     *
     * @param capacity_bytes The total number of bytes to reserve and lock.
     *                       Must be greater than 0. Will be rounded up to the
     *                       nearest page boundary automatically.
     *
     * @throws MemoryAllocationError If the OS cannot allocate the requested memory.
     * @throws MemoryLockError       If the OS cannot lock the memory in physical RAM.
     * @throws std::invalid_argument If capacity_bytes is zero.
     *
     * @note On POSIX systems, the process may need CAP_IPC_LOCK capability or
     *       an elevated RLIMIT_MEMLOCK to lock large amounts of memory.
     * @note On Windows, the process must have the SeLockMemoryPrivilege.
     */
    explicit SovereignArena(size_type capacity_bytes);

    /**
     * @brief Destroys the arena and returns all memory to the operating system.
     *
     * This destructor performs the following cleanup:
     *   1. Unlocks the memory region (VirtualUnlock / munlock).
     *   2. Releases the virtual memory allocation (VirtualFree / munmap).
     *
     * After destruction, all pointers previously returned by allocate() are
     * invalidated. Accessing them is undefined behavior.
     *
     * The destructor is noexcept: it will not throw even if the OS calls fail.
     * Any failures are silently ignored because throwing from a destructor
     * during stack unwinding would cause std::terminate().
     */
    ~SovereignArena();

    // Non-copyable (prevents double-free)
    SovereignArena(const SovereignArena&) = delete;
    SovereignArena& operator=(const SovereignArena&) = delete;

    // Non-movable (arena address stability is a core invariant)
    SovereignArena(SovereignArena&&) = delete;
    SovereignArena& operator=(SovereignArena&&) = delete;

    // =========================================================================
    //  Allocation Interface
    // =========================================================================

    /**
     * @brief Allocates a block of memory from the arena's internal bump allocator.
     *
     * The allocation is O(1) — it simply advances the internal write offset.
     * There is no per-allocation overhead (no headers, footers, or metadata).
     * The returned pointer is guaranteed to be aligned to at least `alignment`
     * bytes. Extra padding bytes inserted for alignment are counted against
     * the arena's capacity.
     *
     * Important: This allocator does NOT support individual deallocation.
     * The entire arena is released at once when the SovereignArena is destroyed.
     * For sub-region reuse, use the SlidingWindowManager class instead.
     *
     * @param size      The number of bytes to allocate. Must be > 0.
     * @param alignment The alignment requirement in bytes. Must be a power of two.
     *                  Defaults to alignof(std::max_align_t), which is typically 16.
     *
     * @return A pointer to the start of the allocated block, aligned as requested.
     *
     * @throws ArenaCapacityExhaustedError If `size` (plus alignment padding)
     *         exceeds the arena's remaining free space.
     * @throws ArenaNotInitializedError   If the arena was not properly initialized.
     * @throws AlignmentError             If `alignment` is not a power of two.
     * @throws std::invalid_argument      If `size` is zero.
     *
     * @note The returned pointer remains valid for the lifetime of the SovereignArena.
     * @note Multiple allocations are contiguous in memory with only alignment gaps.
     */
    [[nodiscard]] pointer allocate(size_type size, size_type alignment = alignof(std::max_align_t));

    /**
     * @brief Allocates and zero-initializes a block of memory from the arena.
     *
     * Equivalent to allocate() followed by std::memset(ptr, 0, size).
     * This is useful for weight matrices and activation buffers that must
     * start from a known state.
     *
     * @param size      The number of bytes to allocate and zero-fill.
     * @param alignment The alignment requirement in bytes (must be power of two).
     *
     * @return A pointer to the zero-initialized block.
     *
     * @throws Same exceptions as allocate().
     */
    [[nodiscard]] pointer allocate_zeroed(size_type size, size_type alignment = alignof(std::max_align_t));

    /**
     * @brief Resets the internal bump allocator to offset zero.
     *
     * All previously allocated memory becomes available for reuse. Existing
     * pointers are NOT invalidated (they still point to valid memory within
     * the arena), but the space they occupy may be returned by subsequent
     * allocate() calls.
     *
     * This is a logical reset only — no OS-level operations are performed
     * (the memory remains locked and allocated). This is O(1).
     *
     * @warning The caller is responsible for ensuring that no in-flight
     *          operations reference memory that will be overwritten after reset.
     */
    void reset() noexcept;

    // =========================================================================
    //  Query Interface
    // =========================================================================

    /**
     * @brief Returns the base address of the arena's memory region.
     * @return Pointer to the first byte of the locked memory region.
     *         Returns nullptr if the arena was not properly initialized.
     */
    [[nodiscard]] pointer base() noexcept;

    /** @brief Const-qualified overload of base(). */
    [[nodiscard]] const_pointer base() const noexcept;

    /**
     * @brief Returns the total capacity of the arena in bytes.
     *
     * This is the maximum number of bytes the arena can hold. It is fixed
     * at construction time and never changes for the lifetime of the arena.
     *
     * @return Total capacity in bytes.
     */
    [[nodiscard]] size_type capacity() const noexcept;

    /**
     * @brief Returns the number of bytes currently allocated (used) by the bump allocator.
     * @return Bytes allocated so far.
     */
    [[nodiscard]] size_type used() const noexcept;

    /**
     * @brief Returns the number of bytes remaining (free) in the arena.
     *
     * Calculated as capacity() - used(). When this reaches zero, subsequent
     * allocate() calls will throw ArenaCapacityExhaustedError.
     *
     * @return Bytes available for future allocations.
     */
    [[nodiscard]] size_type available() const noexcept;

    /**
     * @brief Checks whether the arena's memory is successfully locked in physical RAM.
     * @return true if the memory region is pinned (no paging), false otherwise.
     */
    [[nodiscard]] bool is_locked() const noexcept;

    /**
     * @brief Checks whether the arena was successfully initialized.
     * @return true if the arena holds a valid memory region, false otherwise.
     */
    [[nodiscard]] bool is_initialized() const noexcept;

    /**
     * @brief Calculates the arena's memory utilization as a percentage.
     * @return Value between 0.0 and 100.0 representing used / capacity * 100.
     */
    [[nodiscard]] double utilization_percent() const noexcept;

    // =========================================================================
    //  Diagnostic Interface
    // =========================================================================

    /**
     * @brief Returns the total number of allocations performed since construction (or last reset).
     * @return Allocation count.
     */
    [[nodiscard]] std::uint64_t allocation_count() const noexcept;

    /**
     * @brief Returns the peak memory usage (high-water mark) in bytes.
     *
     * This tracks the maximum value of `used()` observed over the arena's
     * lifetime. It is useful for capacity planning and debugging.
     *
     * @return Peak used bytes.
     */
    [[nodiscard]] size_type peak_used() const noexcept;

private:
    // =========================================================================
    //  Private Implementation Helpers
    // =========================================================================

    /**
     * @brief Validates that an alignment value is a non-zero power of two.
     * @param alignment The value to validate.
     * @throws AlignmentError If alignment is zero or not a power of two.
     */
    static void validate_alignment(size_type alignment);

    /**
     * @brief Validates that a size value is non-zero.
     * @param size The value to validate.
     * @throws std::invalid_argument If size is zero.
     */
    static void validate_size(size_type size);

    /**
     * @brief Performs OS-level memory allocation (VirtualAlloc or mmap).
     * @param size Number of bytes to allocate (must be page-aligned).
     * @return Pointer to the allocated region, or nullptr on failure.
     */
    static pointer os_allocate(size_type size);

    /**
     * @brief Performs OS-level memory locking (VirtualLock or mlock).
     * @param ptr  Pointer to the region to lock.
     * @param size Size of the region in bytes.
     * @return true on success, false on failure.
     */
    static bool os_lock(pointer ptr, size_type size);

    /**
     * @brief Performs OS-level memory unlocking (VirtualUnlock or munlock).
     * @param ptr  Pointer to the region to unlock.
     * @param size Size of the region in bytes.
     */
    static void os_unlock(pointer ptr, size_type size) noexcept;

    /**
     * @brief Performs OS-level memory deallocation (VirtualFree or munmap).
     * @param ptr  Pointer to the region to release.
     * @param size Size of the region in bytes (needed for POSIX munmap).
     */
    static void os_deallocate(pointer ptr, size_type size) noexcept;

    /**
     * @brief Aligns the internal write offset up to the specified alignment.
     * @param alignment The target alignment (must be power of two).
     * @return The aligned offset.
     */
    size_type align_up(size_type offset, size_type alignment) const noexcept;

    // =========================================================================
    //  Member Variables
    // =========================================================================

    pointer     m_base;       ///< Base address of the locked memory region
    size_type   m_capacity;   ///< Total capacity in bytes (page-aligned)
    size_type   m_used;       ///< Current bump allocator offset (bytes used)
    size_type   m_peak_used;  ///< High-water mark of m_used
    bool        m_locked;     ///< True if memory was successfully pinned
    std::uint64_t m_alloc_count; ///< Total number of allocate() calls
};

} // namespace memory
} // namespace ai_os
