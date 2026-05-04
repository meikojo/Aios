/**
 * @file sliding_window_manager.cpp
 * @brief AI OS Core Phase 1 - SlidingWindowManager Implementation
 *
 * This file contains the complete implementation of the SlidingWindowManager
 * class, which provides a ring buffer (circular overwrite) allocator for
 * sequential model layer data. The implementation enforces strict invariants:
 *
 *   - At most ONE layer is Active at any time.
 *   - At most ONE layer is Preparing at any time.
 *   - Retired/Empty layers may be overwritten freely.
 *   - The write cursor never advances past the active layer's boundary.
 *
 * Ring Buffer Strategy:
 *   The managed region is treated as a linear address space with a wrap-around
 *   point at (m_size). Layers are placed at increasing offsets. When the write
 *   head reaches or exceeds m_size, it wraps to offset 0. Before writing, the
 *   manager verifies that the target range does not overlap with the active
 *   layer's range using the ranges_overlap_circular() method.
 *
 * Variable-Size Support:
 *   Each layer can have a different size. The manager tracks per-slot sizes
 *   and offsets, enabling heterogeneous layer dimensions (common in transformer
 *   models where embedding layers differ from attention layers).
 *
 * Performance:
 *   - find_free_region():  O(N) where N = number of slots (typically small).
 *   - contiguous_free_from(): O(N * log N) due to sorting occupied ranges.
 *   - All other operations: O(1).
 *   - No dynamic memory allocation occurs during normal operation (slots are
 *     pre-reserved via vector::reserve).
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 */

#include "ai_os/sliding_window_manager.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace ai_os {
namespace memory {

// =============================================================================
//  Construction / Destruction
// =============================================================================

SlidingWindowManager::SlidingWindowManager(void* region_base, size_type region_size)
    : m_base(region_base)
    , m_size(region_size)
    , m_write_head(0)
    , m_layers_committed(0)
    , m_layers_retired(0)
    , m_next_layer_id(0)
{
    if (region_base == nullptr) {
        throw std::invalid_argument(
            "SlidingWindowManager: region_base must not be nullptr.");
    }
    if (region_size == 0) {
        throw std::invalid_argument(
            "SlidingWindowManager: region_size must be greater than zero.");
    }

    // Pre-reserve a reasonable capacity for the slot vector to avoid
    // reallocation during typical operation. For a 96-layer transformer
    // model, we need at most 96 slots. Reserve 128 to provide headroom.
    m_slots.reserve(128);
}

// =============================================================================
//  Layer Lifecycle Operations
// =============================================================================

LayerHandle SlidingWindowManager::prepare_layer(size_type layer_size) {
    // --- Validation ----------------------------------------------------------------

    // A single layer must fit entirely within the managed region.
    // If a layer is larger than the entire arena, no amount of circular
    // reuse can accommodate it — this is a hard configuration error.
    if (layer_size > m_size) {
        throw InvalidLayerSizeError(layer_size, m_size);
    }

    // Enforce: only one layer can be in the Preparing state at a time.
    // This prevents ambiguous state where multiple prepare_layer() calls
    // without intervening commit_layer() would create conflicting slots.
    for (const auto& slot : m_slots) {
        if (slot.state == LayerSlot::State::Preparing) {
            throw std::logic_error(
                "SlidingWindowManager::prepare_layer: a layer is already in the "
                "Preparing state. Call commit_layer() before preparing another.");
        }
    }

    // --- Find Free Region ----------------------------------------------------------
    // The core algorithm: locate a contiguous free region of at least
    // `layer_size` bytes that does not overlap with the active layer.
    std::ptrdiff_t reuse_slot_index = -1;
    const size_type target_offset = find_free_region(layer_size, reuse_slot_index);

    // --- Handle Slot Reuse or Creation ---------------------------------------------
    std::size_t slot_idx;

    if (reuse_slot_index >= 0) {
        // Reuse an existing slot (Retired or Empty) that already occupies
        // the target offset region. This is the common fast path — it avoids
        // growing the slot vector unnecessarily.
        slot_idx = static_cast<std::size_t>(reuse_slot_index);
        m_slots[slot_idx].offset   = target_offset;
        m_slots[slot_idx].size     = layer_size;
        m_slots[slot_idx].layer_id = m_next_layer_id;
        m_slots[slot_idx].state    = LayerSlot::State::Preparing;
    } else {
        // No existing slot covers this region — create a new one.
        // The slot vector was pre-reserved, so this push_back should not
        // trigger a reallocation in typical usage.
        slot_idx = m_slots.size();
        m_slots.push_back({
            target_offset,
            layer_size,
            m_next_layer_id,
            LayerSlot::State::Preparing
        });
    }

    // --- Advance Write Head --------------------------------------------------------
    // Move the write head past the newly prepared layer.
    // If the write head would exceed the region size, it wraps to 0 to
    // implement the circular (ring buffer) behavior.
    m_write_head = target_offset + layer_size;
    if (m_write_head >= m_size) {
        m_write_head = 0; // Wrap around for circular behavior
    }

    // --- Construct and Return Handle -----------------------------------------------
    // The handle provides the caller with a typed pointer to the layer's
    // memory region, along with metadata for debugging and bookkeeping.
    void* layer_address = static_cast<std::byte*>(m_base) + target_offset;

    // Increment the layer ID for the next prepare_layer() call.
    ++m_next_layer_id;

    return LayerHandle{
        layer_address,
        layer_size,
        target_offset,
        static_cast<std::uint32_t>(m_next_layer_id - 1)
    };
}

void SlidingWindowManager::commit_layer() {
    // Find the slot in the Preparing state and transition it to Active.
    //
    // State Machine Transition:
    //   Preparing ──commit_layer()──▶ Active
    //
    // Once Active, the layer's memory is considered read-only and is being
    // used for inference computation. It cannot be overwritten until
    // retire_layer() is called.
    bool found = false;
    for (auto& slot : m_slots) {
        if (slot.state == LayerSlot::State::Preparing) {
            slot.state = LayerSlot::State::Active;
            found = true;
            ++m_layers_committed;
            break;
        }
    }

    if (!found) {
        throw std::logic_error(
            "SlidingWindowManager::commit_layer: no layer is in the Preparing "
            "state. Call prepare_layer() before committing.");
    }
}

void SlidingWindowManager::retire_layer() {
    // Find the slot in the Active state and transition it to Retired.
    //
    // State Machine Transition:
    //   Active ──retire_layer()──▶ Retired
    //
    // Once Retired, the layer's memory is available for reuse. The data
    // is NOT cleared — it will be overwritten by the next prepare_layer()
    // call that targets this region, ensuring O(1) cleanup cost.
    bool found = false;
    for (auto& slot : m_slots) {
        if (slot.state == LayerSlot::State::Active) {
            slot.state = LayerSlot::State::Retired;
            found = true;
            ++m_layers_retired;
            break;
        }
    }

    if (!found) {
        throw std::logic_error(
            "SlidingWindowManager::retire_layer: no layer is in the Active "
            "state. Commit a layer before retiring it.");
    }
}

void SlidingWindowManager::advance() {
    // Convenience operation: retire the active layer and commit the preparing
    // layer in a single call. This is the most common pattern for sequential
    // model inference:
    //
    //   for each layer:
    //     swm.prepare_layer(size)     // allocate space
    //     load_weights(handle)        // write data
    //     swm.advance()               // retire old + commit new
    //     process(active_layer())     // run inference
    //
    // This is equivalent to calling retire_layer() followed by commit_layer(),
    // but slightly more efficient as it avoids scanning the slot vector twice.
    //
    // State Machine Transition:
    //   Active    ──▶ Retired   (if active exists)
    //   Preparing ──▶ Active    (if preparing exists)

    // Step 1: Retire the current active layer (if any).
    for (auto& slot : m_slots) {
        if (slot.state == LayerSlot::State::Active) {
            slot.state = LayerSlot::State::Retired;
            ++m_layers_retired;
            break; // At most one Active slot exists (class invariant)
        }
    }

    // Step 2: Commit the preparing layer (if any).
    for (auto& slot : m_slots) {
        if (slot.state == LayerSlot::State::Preparing) {
            slot.state = LayerSlot::State::Active;
            ++m_layers_committed;
            break; // At most one Preparing slot exists (class invariant)
        }
    }
}

// =============================================================================
//  Query Interface
// =============================================================================

const LayerSlot* SlidingWindowManager::active_layer() const noexcept {
    for (const auto& slot : m_slots) {
        if (slot.state == LayerSlot::State::Active) {
            return &slot;
        }
    }
    return nullptr;
}

void* SlidingWindowManager::active_layer_address() noexcept {
    const auto* active = active_layer();
    if (active == nullptr) return nullptr;
    return static_cast<std::byte*>(m_base) + active->offset;
}

const void* SlidingWindowManager::active_layer_address() const noexcept {
    const auto* active = active_layer();
    if (active == nullptr) return nullptr;
    return static_cast<const std::byte*>(m_base) + active->offset;
}

void* SlidingWindowManager::region_base() noexcept {
    return m_base;
}

const void* SlidingWindowManager::region_base() const noexcept {
    return m_base;
}

SlidingWindowManager::size_type SlidingWindowManager::region_size() const noexcept {
    return m_size;
}

SlidingWindowManager::size_type SlidingWindowManager::write_head() const noexcept {
    return m_write_head;
}

std::uint64_t SlidingWindowManager::total_layers_committed() const noexcept {
    return m_layers_committed;
}

std::uint64_t SlidingWindowManager::total_layers_retired() const noexcept {
    return m_layers_retired;
}

std::size_t SlidingWindowManager::slot_count() const noexcept {
    return m_slots.size();
}

double SlidingWindowManager::utilization_percent() const noexcept {
    if (m_size == 0) return 0.0;

    // Calculate total bytes occupied by non-Empty slots.
    // Note: in the typical inference pattern, at most one slot is Active
    // and one is Preparing, so utilization is usually low (2 layers worth
    // of data out of the total arena size).
    size_type occupied = 0;
    for (const auto& slot : m_slots) {
        if (slot.state != LayerSlot::State::Empty) {
            occupied += slot.size;
        }
    }

    return (static_cast<double>(occupied) / static_cast<double>(m_size)) * 100.0;
}

// =============================================================================
//  Thread Safety
// =============================================================================

SlidingWindowManager::LockGuard SlidingWindowManager::guard() const {
    return LockGuard(m_mutex);
}

// =============================================================================
//  Diagnostic
// =============================================================================

std::string SlidingWindowManager::diagnostics() const {
    std::ostringstream oss;

    oss << "=== SlidingWindowManager Diagnostics ===\n";
    oss << "Region Base:    " << m_base << "\n";
    oss << "Region Size:    " << m_size << " bytes ("
        << (m_size / (1024 * 1024)) << " MB)\n";
    oss << "Write Head:     " << m_write_head << " bytes\n";
    oss << "Total Committed:" << m_layers_committed << " layers\n";
    oss << "Total Retired:  " << m_layers_retired << " layers\n";
    oss << "Next Layer ID:  " << m_next_layer_id << "\n";
    oss << "Slot Count:     " << m_slots.size() << "\n";
    oss << "Utilization:    " << utilization_percent() << "%\n";
    oss << "------------------------------------------\n";

    for (std::size_t i = 0; i < m_slots.size(); ++i) {
        const auto& slot = m_slots[i];
        oss << "  Slot[" << i << "]: "
            << "offset=" << slot.offset
            << ", size=" << slot.size
            << ", layer_id=" << slot.layer_id
            << ", state=" << slot.state_string()
            << "\n";
    }

    return oss.str();
}

// =============================================================================
//  Private Helpers
// =============================================================================

SlidingWindowManager::size_type SlidingWindowManager::find_free_region(
    size_type required_size,
    std::ptrdiff_t& out_reuse_slot)
{
    out_reuse_slot = -1;

    const std::ptrdiff_t active_idx = find_active_slot();
    bool has_active = (active_idx >= 0);

    // --- Collect active layer range for overlap checking ---------------------------
    size_type active_offset = 0;
    size_type active_size   = 0;
    if (has_active) {
        active_offset = m_slots[static_cast<std::size_t>(active_idx)].offset;
        active_size   = m_slots[static_cast<std::size_t>(active_idx)].size;
    }

    // Lambda: check if placing a layer at [offset, offset+size) would overlap
    // with the currently active layer in the circular buffer.
    auto is_safe = [&](size_type offset, size_type size) -> bool {
        if (!has_active) return true; // No active layer → always safe
        return !ranges_overlap_circular(
            offset, size, active_offset, active_size, m_size);
    };

    // Lambda: try to find a reusable slot at the given offset.
    auto try_reuse_at = [&](size_type offset) -> bool {
        for (std::size_t i = 0; i < m_slots.size(); ++i) {
            if ((m_slots[i].state == LayerSlot::State::Retired ||
                 m_slots[i].state == LayerSlot::State::Empty)) {

                // A reusable slot must start at or before the target offset,
                // and its range must fully cover [offset, offset + required_size).
                const auto& slot = m_slots[i];
                if (slot.offset <= offset &&
                    (slot.offset + slot.size) >= (offset + required_size)) {
                    // Also verify the slot's range doesn't exceed the region
                    // boundary in a way that would make the coverage invalid.
                    if (slot.offset + required_size <= m_size) {
                        out_reuse_slot = static_cast<std::ptrdiff_t>(i);
                        return true;
                    }
                }
            }
        }
        return false;
    };

    // --- Attempt 1: Linear placement at write_head (fast path) ---------------------
    // This is the most common case: the next layer is placed right after the
    // previous one. No wrapping, no searching.
    if (m_write_head + required_size <= m_size) {
        size_type free_here = contiguous_free_from(m_write_head);
        if (free_here >= required_size && is_safe(m_write_head, required_size)) {
            if (try_reuse_at(m_write_head)) {
                return m_write_head;
            }
            return m_write_head;
        }
    }

    // --- Attempt 2: Wrap-around to offset 0 ----------------------------------------
    // When write_head is near the end of the region and there's not enough
    // space for the new layer before the boundary, try starting from offset 0.
    if (m_write_head + required_size > m_size) {
        size_type free_at_zero = contiguous_free_from(0);
        if (free_at_zero >= required_size && is_safe(0, required_size)) {
            if (try_reuse_at(0)) {
                return 0;
            }
            return 0;
        }
    }

    // --- Attempt 3: Scan all Retired/Empty slots for direct reuse -----------------
    // An existing retired slot might be large enough to hold the new layer
    // without any additional searching. This is efficient when layers have
    // variable sizes and a previous layer happened to be larger.
    for (std::size_t i = 0; i < m_slots.size(); ++i) {
        const auto& slot = m_slots[i];
        if (slot.state == LayerSlot::State::Retired ||
            slot.state == LayerSlot::State::Empty) {

            if (slot.size >= required_size && is_safe(slot.offset, required_size)) {
                // Verify the slot fits without wrapping past the region end
                if (slot.offset + required_size <= m_size) {
                    out_reuse_slot = static_cast<std::ptrdiff_t>(i);
                    return slot.offset;
                }
            }
        }
    }

    // --- Attempt 4: Exhaustive scan — check gaps between all occupied slots --------
    // Build a sorted list of all occupied ranges and find the largest gap.
    // If any gap is large enough and safe (no active layer overlap), use it.
    struct Range { size_type start; size_type end; };
    std::vector<Range> occupied;
    occupied.reserve(m_slots.size());

    for (const auto& slot : m_slots) {
        if (slot.state != LayerSlot::State::Empty && slot.size > 0) {
            occupied.push_back({slot.offset, slot.offset + slot.size});
        }
    }
    std::sort(occupied.begin(), occupied.end(),
              [](const Range& a, const Range& b) { return a.start < b.start; });

    // Check gap before the first occupied range (from offset 0 to first.start)
    if (!occupied.empty()) {
        size_type gap_size = occupied.front().start;
        if (gap_size >= required_size && is_safe(0, required_size)) {
            return 0;
        }

        // Check gaps between consecutive occupied ranges
        for (std::size_t i = 0; i + 1 < occupied.size(); ++i) {
            size_type gap_start = occupied[i].end;
            size_type gap_end   = occupied[i + 1].start;
            size_type inner_gap_size  = gap_end - gap_start;
            if (inner_gap_size >= required_size && is_safe(gap_start, required_size)) {
                if (gap_start + required_size <= m_size) {
                    return gap_start;
                }
            }
        }

        // Check gap after the last occupied range (from last.end to m_size)
        size_type last_end = occupied.back().end;
        if (last_end < m_size) {
            size_type tail_gap_size = m_size - last_end;
            if (tail_gap_size >= required_size && is_safe(last_end, required_size)) {
                return last_end;
            }
        }
    } else {
        // No occupied slots at all — entire region is free
        return 0;
    }

    // --- All strategies failed -----------------------------------------------------
    if (has_active) {
        throw WindowOverflowError(
            m_slots[static_cast<std::size_t>(active_idx)].layer_id,
            m_write_head,
            active_offset,
            active_size);
    } else {
        throw WindowOverflowError(0, m_write_head, 0, m_size);
    }
}

SlidingWindowManager::size_type SlidingWindowManager::contiguous_free_from(
    size_type start_offset) const noexcept
{
    // Compute the number of contiguous free bytes starting from `start_offset`,
    // scanning forward without wrapping.
    //
    // Algorithm:
    //   1. Collect all occupied (non-Empty) ranges from the slot list.
    //   2. Sort them by start offset.
    //   3. Find the first occupied range whose start >= start_offset.
    //   4. The free space is: that range's start - start_offset.
    //   5. If no such range exists, the free space extends to m_size.
    //
    // Complexity: O(N log N) due to sorting, where N = number of slots.

    // Collect occupied ranges
    struct Range { size_type start; size_type end; };
    std::vector<Range> occupied;
    occupied.reserve(m_slots.size());

    for (const auto& slot : m_slots) {
        if (slot.state != LayerSlot::State::Empty && slot.size > 0) {
            occupied.push_back({slot.offset, slot.offset + slot.size});
        }
    }

    if (occupied.empty()) {
        // No occupied slots → entire remaining region is free
        return m_size - start_offset;
    }

    // Sort by start offset
    std::sort(occupied.begin(), occupied.end(),
              [](const Range& a, const Range& b) { return a.start < b.start; });

    // Find the first occupied range that starts at or after start_offset.
    // Use binary search for O(log N) lookup.
    auto it = std::lower_bound(occupied.begin(), occupied.end(), start_offset,
        [](const Range& r, size_type val) { return r.start < val; });

    if (it != occupied.end()) {
        // Found an occupied range at or after start_offset.
        // But we also need to check if start_offset falls INSIDE an earlier
        // range that overlaps with it (i.e., an earlier range whose end > start_offset).
        if (it != occupied.begin()) {
            auto prev = it - 1;
            if (prev->end > start_offset) {
                // start_offset is inside an occupied range → no free space
                return 0;
            }
        }
        return it->start - start_offset;
    } else {
        // No occupied range at or after start_offset.
        // Check if start_offset falls inside the last occupied range.
        auto& last = occupied.back();
        if (last.start <= start_offset && last.end > start_offset) {
            return 0; // Inside an occupied range
        }
        return m_size - start_offset;
    }
}

bool SlidingWindowManager::ranges_overlap_circular(
    size_type a_start, size_type a_len,
    size_type b_start, size_type b_len,
    size_type ring_size) noexcept
{
    // Check if two byte ranges overlap within a circular ring buffer.
    //
    // A range [start, start + len) in a ring buffer of size `ring_size` may
    // wrap around. We need to handle four cases:
    //
    //   Case 1: Neither range wraps.
    //     Standard interval overlap: overlap iff a_start < b_end && b_start < a_end
    //
    //   Case 2: Range A wraps, Range B does not.
    //     A occupies [a_start, ring_size) ∪ [0, a_start + a_len - ring_size)
    //     B occupies [b_start, b_start + b_len)
    //     Overlap if B intersects either segment of A.
    //
    //   Case 3: Range A does not wrap, Range B wraps.
    //     Symmetric to Case 2.
    //
    //   Case 4: Both ranges wrap.
    //     Both occupy [start, ring_size) ∪ [0, remainder)
    //     Always overlap (they both include the beginning of the ring).
    //
    // For simplicity and correctness, we normalize: a range wraps if
    // start + len > ring_size. We check overlap by testing both the
    // "tail" segment [start, ring_size) and the "head" segment [0, remainder).

    size_type a_end = a_start + a_len;
    size_type b_end = b_start + b_len;

    // Normalize: compute the two segments for each range.
    // Segment 1: [start, min(end, ring_size))
    // Segment 2: [0, max(0, end - ring_size)) — only if end > ring_size

    auto seg1_start = a_start;
    auto seg1_end   = (a_end > ring_size) ? ring_size : a_end;
    auto seg2_start = size_type(0);
    auto seg2_end   = (a_end > ring_size) ? (a_end - ring_size) : size_type(0);
    bool a_wraps    = (a_end > ring_size);

    auto b_seg1_start = b_start;
    auto b_seg1_end   = (b_end > ring_size) ? ring_size : b_end;
    auto b_seg2_start = size_type(0);
    auto b_seg2_end   = (b_end > ring_size) ? (b_end - ring_size) : size_type(0);
    bool b_wraps      = (b_end > ring_size);

    // Helper: check overlap of two non-wrapping intervals
    auto intervals_overlap = [](size_type s1, size_type e1,
                                size_type s2, size_type e2) -> bool {
        if (e1 <= s1 || e2 <= s2) return false; // Empty intervals
        return s1 < e2 && s2 < e1;
    };

    // Check all segment pair combinations
    if (intervals_overlap(seg1_start, seg1_end, b_seg1_start, b_seg1_end)) {
        return true;
    }
    if (b_wraps && intervals_overlap(seg1_start, seg1_end, b_seg2_start, b_seg2_end)) {
        return true;
    }
    if (a_wraps && intervals_overlap(seg2_start, seg2_end, b_seg1_start, b_seg1_end)) {
        return true;
    }
    if (a_wraps && b_wraps && intervals_overlap(seg2_start, seg2_end, b_seg2_start, b_seg2_end)) {
        return true;
    }

    return false;
}

std::ptrdiff_t SlidingWindowManager::find_active_slot() const noexcept {
    for (std::size_t i = 0; i < m_slots.size(); ++i) {
        if (m_slots[i].state == LayerSlot::State::Active) {
            return static_cast<std::ptrdiff_t>(i);
        }
    }
    return -1;
}

} // namespace memory
} // namespace ai_os
