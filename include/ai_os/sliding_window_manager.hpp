/**
 * @file sliding_window_manager.hpp
 * @brief AI OS Core Phase 1 - Sliding Window Memory Manager
 *
 * SlidingWindowManager implements a ring buffer (circular overwrite) allocator
 * that operates within a memory region provided by SovereignArena. It is the
 * second layer of the AI OS Core memory hierarchy, responsible for managing
 * sequential model layer data in a fixed-size buffer without any dynamic
 * memory allocation or garbage collection.
 *
 * Architectural Role:
 *   The manager treats the provided memory region as a circular ring. Model
 *   layers are loaded sequentially into this ring. When the write cursor
 *   wraps around, it overwrites the oldest layer data — achieving O(1)
 *   memory complexity regardless of how many layers are processed.
 *
 *   Memory Layout (conceptual, with 3 layers in a ring):
 *
 *     ┌───────────────────────────────────────────────┐
 *     │  Layer 3  │       Layer 1       │   Layer 2   │
 *     │ (writing) │    (active/read)    │ (stale)     │
 *     └───────────────────────────────────────────────┘
 *      ▲           ▲                     ▲
 *   write_head  read_head             retired
 *
 *   When Layer 1 finishes processing:
 *     - write_head advances, Layer 2 becomes the new active layer
 *     - If write_head wraps, Layer 3's space is reused for Layer 4
 *
 * Key Properties:
 *   1. ZERO ALLOCATION: No malloc, new, or any OS-level allocation occurs.
 *      All memory comes from the pre-locked SovereignArena region.
 *
 *   2. ZERO GARBAGE: No free/delete/cleanup is needed. Old layers are simply
 *      overwritten by new layers in a deterministic circular pattern.
 *
 *   3. O(1) MEMORY: The total memory footprint is exactly the size of the
 *      provided region. Processing 10 layers or 10,000 layers consumes the
 *      same amount of memory.
 *
 *   4. VARIABLE-SIZE LAYERS: Each layer can have a different size. The ring
 *      buffer tracks offsets and sizes per layer slot.
 *
 * Thread Safety:
 *   NOT thread-safe by default. Use the guard() method to obtain a scoped
 *   lock when operating from multiple threads. The internal mutex is a
 *   std::recursive_mutex to allow re-entrant locking within the same thread.
 *
 * Usage Pattern (Model Inference):
 * @code
 *   SovereignArena arena(500 * 1024 * 1024);  // 500MB locked
 *   void* region = arena.allocate(500 * 1024 * 1024);
 *
 *   SlidingWindowManager swm(region, 500 * 1024 * 1024);
 *
 *   // Load and process each layer sequentially
 *   for (uint32_t i = 0; i < model.num_layers(); ++i) {
 *       auto [ptr, offset, size] = swm.prepare_layer(model.layer_size(i));
 *       model.load_layer_weights(i, ptr);        // Load weights into ptr
 *       swm.commit_layer();
 *       inference_engine.process(swm.active_layer());
 *       swm.retire_layer();
 *   }
 * @endcode
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 *
 * @see SovereignArena - The memory arena that provides the underlying region.
 * @see exceptions.hpp  - Exception types thrown by this class.
 */

#pragma once

#include "ai_os/platform.hpp"
#include "ai_os/exceptions.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ai_os {
namespace memory {

// =============================================================================
//  LayerSlot - Descriptor for a single layer within the ring buffer
// =============================================================================

/**
 * @struct LayerSlot
 * @brief Describes the position, size, and state of a single layer in the ring buffer.
 *
 * Each layer loaded into the SlidingWindowManager is tracked by a LayerSlot.
 * The slot stores the byte offset within the managed region, the layer's
 * size, its sequential ID (which model layer it represents), and its current
 * lifecycle state.
 */
struct LayerSlot {
    std::size_t   offset{0};     ///< Byte offset from the region's base address
    std::size_t   size{0};       ///< Size of this layer in bytes
    std::uint32_t layer_id{0};   ///< Sequential identifier (0-indexed model layer number)

    /**
     * @enum State
     * @brief Lifecycle state of a layer slot within the ring buffer.
     */
    enum class State : std::uint8_t {
        Empty,      ///< Slot is unused / available for overwrite
        Preparing,  ///< Memory is being written (layer data being loaded)
        Active,     ///< Layer is currently being processed (read-only)
        Retired     ///< Layer processing is complete; safe to overwrite
    };

    State         state{State::Empty};  ///< Current lifecycle state

    /**
     * @brief Computes the absolute memory address of this layer's data.
     * @param base The base address of the managed memory region.
     * @return Pointer to the first byte of this layer's data.
     */
    [[nodiscard]] void* address(void* base) const noexcept {
        return static_cast<std::byte*>(base) + offset;
    }

    /**
     * @brief Const-qualified overload of address().
     */
    [[nodiscard]] const void* address(const void* base) const noexcept {
        return static_cast<const std::byte*>(base) + offset;
    }

    /**
     * @brief Returns a human-readable string for the slot's current state.
     */
    [[nodiscard]] const char* state_string() const noexcept {
        switch (state) {
            case State::Empty:     return "Empty";
            case State::Preparing: return "Preparing";
            case State::Active:    return "Active";
            case State::Retired:   return "Retired";
            default:               return "Unknown";
        }
    }
};

// =============================================================================
//  LayerHandle - Opaque handle returned by prepare_layer()
// =============================================================================

/**
 * @struct LayerHandle
 * @brief A handle to a layer that has been prepared for writing.
 *
 * Returned by prepare_layer(), this handle provides the caller with the
 * memory address and size for writing layer data. The handle must be
 * passed to commit_layer() to finalize the layer.
 */
struct LayerHandle {
    void*        address{nullptr};  ///< Writable pointer to the layer's memory region
    std::size_t  size{0};           ///< Size of the allocated layer region in bytes
    std::size_t  offset{0};         ///< Byte offset from the region base
    std::uint32_t layer_id{0};      ///< Sequential layer identifier

    /**
     * @brief Convenience cast operator for passing the address to C APIs.
     */
    explicit operator void*() const noexcept { return address; }

    /**
     * @brief Checks if the handle is valid (non-null address).
     */
    [[nodiscard]] bool is_valid() const noexcept { return address != nullptr; }
};

// =============================================================================
//  SlidingWindowManager - Ring Buffer Layer Manager
// =============================================================================

/**
 * @class SlidingWindowManager
 * @brief Manages sequential model layers in a circular ring buffer with zero allocation.
 *
 * The SlidingWindowManager treats a pre-allocated memory region as a ring buffer.
 * Model layers are loaded one at a time; when the ring wraps around, the oldest
 * layer's memory is reused. This ensures that memory consumption is always O(1)
 * — bounded by the size of the managed region — regardless of the total number
 * of layers processed.
 *
 * Lifecycle of a Layer:
 *   1. prepare_layer(size) → allocates space, returns LayerHandle (state: Preparing)
 *   2. Caller writes layer data to handle.address
 *   3. commit_layer() → marks the layer as ready (state: Active)
 *   4. Caller processes the layer (inference computation)
 *   5. retire_layer() → marks the layer as done (state: Retired)
 *   6. The next prepare_layer() may reuse the retired slot's memory
 *
 * Invariant:
 *   At most ONE layer is in the Active state at any time. This is enforced
 *   by the manager and ensures that the write cursor never overwrites data
 *   that is still being read.
 */
class SlidingWindowManager {
public:
    // =========================================================================
    //  Type Aliases
    // =========================================================================

    using size_type = std::size_t;

    // =========================================================================
    //  Construction / Destruction
    // =========================================================================

    /**
     * @brief Constructs a SlidingWindowManager over a pre-allocated memory region.
     *
     * The manager does NOT take ownership of the memory region. The caller
     * is responsible for ensuring that the region remains valid for the
     * entire lifetime of this manager. Typically, the region comes from
     * a SovereignArena allocation.
     *
     * @param region_base  Pointer to the start of the managed memory region.
     *                     Must be non-null.
     * @param region_size  Size of the managed memory region in bytes.
     *                     Must be greater than 0.
     *
     * @throws std::invalid_argument If region_base is nullptr or region_size is 0.
     */
    SlidingWindowManager(void* region_base, size_type region_size);

    /**
     * @brief Destructor. Does not free the managed memory region.
     *
     * The managed region is owned externally (typically by SovereignArena).
     * This destructor only cleans up internal bookkeeping structures.
     */
    ~SlidingWindowManager() = default;

    // Copyable and movable are intentionally deleted because the ring buffer
    // contains absolute offsets into a specific memory region.
    SlidingWindowManager(const SlidingWindowManager&) = delete;
    SlidingWindowManager& operator=(const SlidingWindowManager&) = delete;
    SlidingWindowManager(SlidingWindowManager&&) = delete;
    SlidingWindowManager& operator=(SlidingWindowManager&&) = delete;

    // =========================================================================
    //  Layer Lifecycle Operations
    // =========================================================================

    /**
     * @brief Prepares a memory slot for loading the next model layer.
     *
     * This method finds or creates a slot within the ring buffer large enough
     * to hold `layer_size` bytes. The returned LayerHandle provides a writable
     * pointer where the caller should load the layer's weight data.
     *
     * Slot Selection Strategy:
     *   - First, attempt to allocate at the current write_head position.
     *   - If the write_head would collide with the active layer, attempt to
     *     wrap around to the beginning of the region.
     *   - If wrapping also collides, throw WindowOverflowError.
     *   - Reuse Retired or Empty slots wherever possible.
     *
     * The newly prepared slot enters the Preparing state and will be promoted
     * to Active when commit_layer() is called.
     *
     * @param layer_size The size of the upcoming layer in bytes.
     *
     * @return A LayerHandle containing the writable address, size, and metadata.
     *
     * @throws InvalidLayerSizeError    If layer_size exceeds total region capacity.
     * @throws WindowOverflowError      If no contiguous free space is available.
     *
     * @pre No other layer must be in the Preparing state.
     * @post A slot is reserved and the caller must call commit_layer() next.
     */
    [[nodiscard]] LayerHandle prepare_layer(size_type layer_size);

    /**
     * @brief Commits the most recently prepared layer, making it the active layer.
     *
     * Transitions the preparing slot from Preparing → Active. After this call,
     * the layer data should be treated as read-only (it is now being used
     * for inference computation).
     *
     * @pre prepare_layer() must have been called successfully and no subsequent
     *      commit_layer() without an intervening retire_layer().
     *
     * @throws std::logic_error If no layer is in the Preparing state.
     */
    void commit_layer();

    /**
     * @brief Retires the currently active layer, marking its memory as reusable.
     *
     * Transitions the active slot from Active → Retired. The memory occupied
     * by the retired layer becomes available for future prepare_layer() calls.
     * The actual data is NOT zeroed or modified — it will be overwritten when
     * the next layer needs that space.
     *
     * @pre A layer must be in the Active state.
     *
     * @throws std::logic_error If no layer is currently active.
     */
    void retire_layer();

    /**
     * @brief Convenience method: commit + retire in a single call.
     *
     * Atomically commits the preparing layer and retires the current active
     * layer (if any). This is the typical pattern in sequential inference
     * where layers are processed one at a time.
     */
    void advance();

    // =========================================================================
    //  Query Interface
    // =========================================================================

    /**
     * @brief Returns the active layer's slot descriptor, if any.
     * @return Pointer to the active LayerSlot, or nullptr if no layer is active.
     */
    [[nodiscard]] const LayerSlot* active_layer() const noexcept;

    /**
     * @brief Returns the active layer's memory address.
     * @return Pointer to the active layer's data, or nullptr if no layer is active.
     */
    [[nodiscard]] void* active_layer_address() noexcept;

    /**
     * @brief Const-qualified overload of active_layer_address().
     */
    [[nodiscard]] const void* active_layer_address() const noexcept;

    /**
     * @brief Returns the base address of the managed memory region.
     */
    [[nodiscard]] void* region_base() noexcept;

    /**
     * @brief Const-qualified overload of region_base().
     */
    [[nodiscard]] const void* region_base() const noexcept;

    /**
     * @brief Returns the total size of the managed memory region.
     */
    [[nodiscard]] size_type region_size() const noexcept;

    /**
     * @brief Returns the current write head offset (where next write will occur).
     */
    [[nodiscard]] size_type write_head() const noexcept;

    /**
     * @brief Returns the total number of layers that have been committed.
     */
    [[nodiscard]] std::uint64_t total_layers_committed() const noexcept;

    /**
     * @brief Returns the total number of layers that have been retired.
     */
    [[nodiscard]] std::uint64_t total_layers_retired() const noexcept;

    /**
     * @brief Returns the number of slot entries currently in the ring buffer history.
     *
     * This is the count of all LayerSlot records (including Empty, Retired, etc.).
     * It grows as new slots are created but old slots are reused, so this
     * represents the maximum ring occupancy, not the current active count.
     */
    [[nodiscard]] std::size_t slot_count() const noexcept;

    /**
     * @brief Calculates current memory utilization as a percentage.
     * @return Value between 0.0 and 100.0.
     */
    [[nodiscard]] double utilization_percent() const noexcept;

    // =========================================================================
    //  Thread Safety
    // =========================================================================

    /**
     * @brief RAII lock guard type for thread-safe access.
     *
     * Usage:
     * @code
     *   auto lock = swm.guard();
     *   auto handle = swm.prepare_layer(size);
     *   // ... write data ...
     *   swm.commit_layer();
     * @endcode
     */
    using LockGuard = std::lock_guard<std::recursive_mutex>;

    /**
     * @brief Returns a scoped lock that protects all operations on this manager.
     * @return A std::lock_guard that locks the internal mutex.
     */
    [[nodiscard]] LockGuard guard() const;

    // =========================================================================
    //  Diagnostic
    // =========================================================================

    /**
     * @brief Returns a human-readable summary of the ring buffer state.
     *
     * Useful for debugging and logging. Includes all slot states, write head
     * position, and utilization metrics.
     *
     * @return Multi-line string describing the current state.
     */
    [[nodiscard]] std::string diagnostics() const;

private:
    // =========================================================================
    //  Private Helpers
    // =========================================================================

    /**
     * @brief Finds a contiguous free region of at least `required_size` bytes.
     *
     * Searches forward from the write head, with wrap-around. Returns the
     * offset where the layer should be placed.
     *
     * @param required_size  The number of bytes needed.
     * @param out_reuse_slot [out] Index of an existing slot to reuse, or -1 if new.
     *
     * @return Byte offset where the new layer should be placed.
     *
     * @throws WindowOverflowError If no contiguous free region is available.
     */
    size_type find_free_region(size_type required_size, std::ptrdiff_t& out_reuse_slot);

    /**
     * @brief Computes the amount of free contiguous space from a given offset.
     *
     * Scans forward from `start_offset` until hitting an occupied slot or
     * the end of the region (with wrap-around).
     *
     * @param start_offset  The offset to start scanning from.
     * @return Number of contiguous free bytes.
     */
    size_type contiguous_free_from(size_type start_offset) const noexcept;

    /**
     * @brief Finds the index of the active slot, if any.
     * @return Slot index, or -1 if no slot is active.
     */
    [[nodiscard]] std::ptrdiff_t find_active_slot() const noexcept;

    /**
     * @brief Checks if two byte ranges [a, a+a_len) and [b, b+b_len) overlap
     *        in the circular ring buffer.
     *
     * @param a_start Start of first range.
     * @param a_len   Length of first range.
     * @param b_start Start of second range.
     * @param b_len   Length of second range.
     * @return true if the ranges overlap in the circular buffer.
     */
    [[nodiscard]] static bool ranges_overlap_circular(
        size_type a_start, size_type a_len,
        size_type b_start, size_type b_len,
        size_type ring_size) noexcept;

    // =========================================================================
    //  Member Variables
    // =========================================================================

    void*                m_base;             ///< Base address of the managed region
    size_type            m_size;             ///< Total size of the managed region
    size_type            m_write_head;       ///< Current write cursor offset
    std::vector<LayerSlot> m_slots;          ///< Ring buffer slot descriptors
    std::uint64_t        m_layers_committed; ///< Total layers transitioned to Active
    std::uint64_t        m_layers_retired;   ///< Total layers transitioned to Retired
    std::uint32_t        m_next_layer_id;    ///< Auto-incrementing layer ID counter

    mutable std::recursive_mutex m_mutex;    ///< Thread safety mutex
};

} // namespace memory
} // namespace ai_os
