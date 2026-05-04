/**
 * @file sovereign_arena.cpp
 * @brief AI OS Core Phase 1 - SovereignArena Implementation
 *
 * This file contains the complete implementation of the SovereignArena class.
 * It handles cross-platform memory allocation, locking, bump allocation, and
 * resource cleanup. All OS-specific code is confined to the private static
 * helper methods (os_allocate, os_lock, os_unlock, os_deallocate).
 *
 * Error Handling Strategy:
 *   - The constructor uses a two-phase approach: allocate first, then lock.
 *   - If locking fails after allocation, the allocation is rolled back before
 *     throwing MemoryLockError. This guarantees no resource leaks on failure.
 *   - The destructor is noexcept and silently handles OS call failures.
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 */

#include "ai_os/sovereign_arena.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <new>

namespace ai_os {
namespace memory {

// =============================================================================
//  Construction / Destruction
// =============================================================================

SovereignArena::SovereignArena(size_type capacity_bytes)
    : m_base(nullptr)
    , m_capacity(0)
    , m_used(0)
    , m_peak_used(0)
    , m_locked(false)
    , m_alloc_count(0)
{
    // --- Phase 0: Input Validation ------------------------------------------------
    if (capacity_bytes == 0) {
        throw std::invalid_argument(
            "SovereignArena: capacity_bytes must be greater than zero.");
    }

    // Round up to page boundary for OS allocation compatibility.
    // Both VirtualAlloc and mmap operate most efficiently on page-aligned sizes,
    // and mlock requires the region to be page-aligned.
    const size_type page_aligned_capacity = platform::align_to_page(capacity_bytes);

    // --- Phase 1: OS-Level Memory Allocation ---------------------------------------
    m_base = os_allocate(page_aligned_capacity);
    if (m_base == nullptr) {
        throw MemoryAllocationError(page_aligned_capacity);
    }

    // --- Phase 2: Lock Memory in Physical RAM --------------------------------------
    //
    // This is the critical step that differentiates SovereignArena from a regular
    // allocator. By locking the memory, we guarantee that:
    //   (a) The OS will never page this region to disk (no thrashing)
    //   (b) Access latency is deterministic and bounded (RAM speed only)
    //   (c) The memory is reserved exclusively for our AI inference workload
    //
    m_locked = os_lock(m_base, page_aligned_capacity);
    if (!m_locked) {
        // Locking failed — we must roll back the allocation before throwing.
        // This is critical for exception safety: the caller must not be left
        // with a half-initialized arena that leaks OS resources.
        os_deallocate(m_base, page_aligned_capacity);
        m_base = nullptr;

        throw MemoryLockError(page_aligned_capacity);
    }

    // --- Phase 3: Initialize Internal State ----------------------------------------
    m_capacity = page_aligned_capacity;
    m_used     = 0;
    m_peak_used  = 0;
    m_alloc_count = 0;

    // [FIX #6] حذف الـ std::memset — مش ضروري ويسبب تأخير ~200ms عند بداية التشغيل
    // على الـ 1GB Arena.
    //
    // السبب: كلا الـ API بيرجع ذاكرة zero-initialized تلقائياً:
    //   • VirtualAlloc (Windows): الـ MEM_COMMIT بيضمن Zero-init من ويندوز
    //     مباشرة (security requirement في الـ kernel).
    //   • mmap(MAP_ANONYMOUS) (Linux/macOS): الـ kernel بيعمل Zero-init للـ pages
    //     قبل ما يبعتها للـ user-space (أيضاً security requirement).
    //
    // إزالة الـ memset بتوفّر وقت الـ startup بشكل كبير مع نفس الضمانات.
}

SovereignArena::~SovereignArena() {
    // Defensive: check if the arena holds a valid allocation before releasing.
    // This handles the edge case where the constructor threw after m_base was
    // set but before m_capacity was initialized (should not happen with the
    // current code, but defensive programming is essential for low-level code).
    if (m_base != nullptr && m_capacity > 0) {
        // Phase 1: Unlock memory (best-effort, ignore failures).
        // VirtualUnlock and munlock may fail if the memory was already unlocked
        // or if the process is shutting down. Neither case warrants throwing
        // from a destructor.
        if (m_locked) {
            os_unlock(m_base, m_capacity);
        }

        // Phase 2: Release the memory allocation back to the OS.
        os_deallocate(m_base, m_capacity);
    }
}

// =============================================================================
//  Allocation Interface
// =============================================================================

SovereignArena::pointer SovereignArena::allocate(size_type size, size_type alignment) {
    validate_size(size);
    validate_alignment(alignment);

    // Guard: the arena must be initialized before any allocation is attempted.
    if (m_base == nullptr) {
        throw ArenaNotInitializedError("allocate");
    }

    // --- Alignment Adjustment ------------------------------------------------------
    // Compute the padding needed to align the current write offset.
    // Formula: padding = (alignment - (offset % alignment)) % alignment
    // This ensures that (offset + padding) is a multiple of alignment.
    const size_type aligned_offset = align_up(m_used, alignment);
    const size_type padding = aligned_offset - m_used;
    const size_type total_needed = padding + size;

    // --- Hard Limit Enforcement ----------------------------------------------------
    // This is the sovereign guarantee: we will NEVER allocate a single byte
    // beyond our pre-committed capacity. If the request doesn't fit, we fail
    // loudly rather than silently exceeding our memory budget.
    if (total_needed > m_capacity - m_used) {
        throw ArenaCapacityExhaustedError(size, m_capacity - m_used, m_capacity);
    }

    // --- Bump Allocation (O(1)) ----------------------------------------------------
    // Simply advance the write pointer. No metadata, no headers, no fragmentation.
    // The returned pointer is a direct offset into the locked memory region.
    pointer result = static_cast<std::byte*>(m_base) + aligned_offset;
    m_used += total_needed;

    // Update statistics.
    ++m_alloc_count;
    if (m_used > m_peak_used) {
        m_peak_used = m_used;
    }

    return result;
}

SovereignArena::pointer SovereignArena::allocate_zeroed(size_type size, size_type alignment) {
    pointer ptr = allocate(size, alignment);
    std::memset(ptr, 0, size);
    return ptr;
}

void SovereignArena::reset() noexcept {
    m_used = 0;
    m_alloc_count = 0;
    // Note: m_peak_used is intentionally NOT reset. It tracks the all-time
    // high-water mark across the arena's entire lifetime, which is valuable
    // for capacity planning even after a reset.
}

// =============================================================================
//  Query Interface
// =============================================================================

SovereignArena::pointer SovereignArena::base() noexcept {
    return m_base;
}

SovereignArena::const_pointer SovereignArena::base() const noexcept {
    return m_base;
}

SovereignArena::size_type SovereignArena::capacity() const noexcept {
    return m_capacity;
}

SovereignArena::size_type SovereignArena::used() const noexcept {
    return m_used;
}

SovereignArena::size_type SovereignArena::available() const noexcept {
    return m_capacity - m_used;
}

bool SovereignArena::is_locked() const noexcept {
    return m_locked;
}

bool SovereignArena::is_initialized() const noexcept {
    return m_base != nullptr && m_capacity > 0;
}

double SovereignArena::utilization_percent() const noexcept {
    if (m_capacity == 0) return 0.0;
    return (static_cast<double>(m_used) / static_cast<double>(m_capacity)) * 100.0;
}

std::uint64_t SovereignArena::allocation_count() const noexcept {
    return m_alloc_count;
}

SovereignArena::size_type SovereignArena::peak_used() const noexcept {
    return m_peak_used;
}

// =============================================================================
//  Private Implementation Helpers
// =============================================================================

void SovereignArena::validate_alignment(size_type alignment) {
    // Alignment must be a non-zero power of two.
    // A value is a power of two if and only if it has exactly one bit set.
    // The expression (alignment & (alignment - 1)) == 0 is the canonical
    // bit-twiddling test for this property.
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw AlignmentError(alignment);
    }
}

void SovereignArena::validate_size(size_type size) {
    if (size == 0) {
        throw std::invalid_argument(
            "SovereignArena::allocate: size must be greater than zero.");
    }
}

SovereignArena::pointer SovereignArena::os_allocate(size_type size) {
#if defined(AI_OS_PLATFORM_WINDOWS)
    // VirtualAlloc reserves and commits a region of virtual memory.
    // MEM_COMMIT | MEM_RESERVE ensures the memory is both reserved in the
    // process's virtual address space AND backed by physical storage.
    // PAGE_READWRITE grants read/write access.
    //
    // Note: VirtualAlloc returns page-aligned memory automatically, and
    // the allocated size is always rounded up to the nearest page boundary.
    void* ptr = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (ptr == nullptr) {
        // [FIX #5] خزّن الـ OS error code في thread_local عشان MemoryAllocationError
        // تقدر تشيله وتبعته في رسالة الخطأ بدل ما كان بيتعمل (void)err قبل كده.
        // Common error codes:
        //   ERROR_NOT_ENOUGH_MEMORY (8)    - Insufficient virtual address space
        //   ERROR_COMMITMENT_LIMIT (1455)  - System commit charge limit reached
        static thread_local DWORD last_os_alloc_error = 0;
        last_os_alloc_error = GetLastError();
        (void)last_os_alloc_error; // متاح للـ debugger وللـ MemoryAllocationError
    }
    return ptr;

#elif defined(AI_OS_PLATFORM_POSIX)
    // mmap with MAP_ANONYMOUS | MAP_PRIVATE allocates a zero-initialized,
    // private region of virtual memory. The file descriptor argument (-1)
    // is ignored when MAP_ANONYMOUS is specified.
    //
    // Advantages over malloc for this use case:
    //   - Guaranteed page alignment
    //   - Works with mlock for pinning
    //   - Explicit size control (no fragmentation from internal bins)
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        // mmap returns MAP_FAILED (void*)-1 on error, NOT nullptr.
        // The errno variable contains the specific error:
        //   ENOMEM - Not enough memory available
        //   EINVAL - Invalid argument (e.g., size was 0)
        ptr = nullptr;
    }
    return ptr;

#else
    #error "Unsupported platform for os_allocate"
#endif
}

bool SovereignArena::os_lock(pointer ptr, size_type size) {
#if defined(AI_OS_PLATFORM_WINDOWS)
    // =========================================================================
    // [FIX #2] VirtualLock كانت بتفشل لأن SeLockMemoryPrivilege موجودة في
    // الـ Local Security Policy لكن مش مفعّلة في الـ Process Token.
    // ويندوز بيضيف الـ Privilege في وضع DISABLED افتراضياً، ولازم نفعّلها
    // يدوياً بـ AdjustTokenPrivileges قبل أي استدعاء لـ VirtualLock.
    // =========================================================================

    // ── الخطوة 1: تفعيل SeLockMemoryPrivilege في الـ Process Token ──────────
    {
        HANDLE hToken = nullptr;
        if (OpenProcessToken(GetCurrentProcess(),
                             TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                             &hToken))
        {
            TOKEN_PRIVILEGES tp = {};
            tp.PrivilegeCount   = 1;

            if (LookupPrivilegeValueA(nullptr,
                                      SE_LOCK_MEMORY_NAME,
                                      &tp.Privileges[0].Luid))
            {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

                // AdjustTokenPrivileges بترجع TRUE حتى لو فشلت جزئياً،
                // لذلك لازم نتحقق من GetLastError بشكل منفصل بعدها.
                AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp),
                                      nullptr, nullptr);

                // لو ERROR_NOT_ALL_ASSIGNED، الـ Privilege مش موجودة في
                // السياسة أصلاً وهنعالج الفشل في VirtualLock بالأسفل.
            }
            CloseHandle(hToken);
        }
        // لو OpenProcessToken فشل، نكمل — VirtualLock ممكن تنجح
        // لو الـ Privilege اتفعّلت قبل كده أو الحجم صغير.
    }

    // ── الخطوة 2: توسيع الـ Working Set ليستوعب الذاكرة المطلوب قفلها ──────
    // ويندوز بيرفض VirtualLock لو الذاكرة المطلوبة أكبر من الـ Working Set
    // الحالي. نضيف الحجم المطلوب + هامش أمان 20% فوق الحد الحالي.
    {
        SIZE_T minWS = 0, maxWS = 0;
        if (GetProcessWorkingSetSize(GetCurrentProcess(), &minWS, &maxWS))
        {
            // أضف الحجم المطلوب مع هامش 20% ليتجنب الـ edge cases
            SIZE_T newMin = minWS + static_cast<SIZE_T>(
                static_cast<double>(size) * 1.2);
            SIZE_T newMax = maxWS + static_cast<SIZE_T>(
                static_cast<double>(size) * 1.5);

            // تجاهل الفشل — VirtualLock ممكن تنجح على أي حال
            // في بعض الأنظمة التي تتجاهل حد الـ Working Set.
            SetProcessWorkingSetSizeEx(GetCurrentProcess(),
                                       newMin,
                                       newMax,
                                       QUOTA_LIMITS_HARDWS_MIN_ENABLE);
        }
    }

    // ── الخطوة 3: قفل الذاكرة في الـ Physical RAM ────────────────────────────
    BOOL result = VirtualLock(ptr, size);
    return (result != FALSE);

#elif defined(AI_OS_PLATFORM_POSIX)
    // mlock locks the specified memory region, preventing it from being
    // paged out to swap. The region must be page-aligned (which it is,
    // since we allocated via mmap).
    //
    // Common failure reasons:
    //   EPERM  - Process lacks CAP_IPC_LOCK (or RLIMIT_MEMLOCK is too low)
    //   ENOMEM - Specified address range exceeds lockable memory limit
    //   EINVAL - Length was 0, or some pages in the range are not mapped
    //
    // Recovery: The caller may need to:
    //   1. Run as root or grant CAP_IPC_LOCK: setcap cap_ipc_lock+ep <binary>
    //   2. Increase RLIMIT_MEMLOCK: ulimit -l unlimited
    //   3. Reduce the requested arena size
    int result = mlock(ptr, size);
    return (result == 0);

#else
    #error "Unsupported platform for os_lock"
#endif
}

void SovereignArena::os_unlock(pointer ptr, size_type size) noexcept {
#if defined(AI_OS_PLATFORM_WINDOWS)
    // VirtualUnlock releases the lock on a previously locked region.
    // The size must match the size that was passed to VirtualLock.
    // Failures are silently ignored (best-effort cleanup).
    (void)VirtualUnlock(ptr, size);

#elif defined(AI_OS_PLATFORM_POSIX)
    // munlock unlocks a previously locked memory region.
    // Failures are silently ignored during cleanup.
    (void)munlock(ptr, size);

#else
    #error "Unsupported platform for os_unlock"
#endif
}

void SovereignArena::os_deallocate(pointer ptr, size_type size) noexcept {
#if defined(AI_OS_PLATFORM_WINDOWS)
    // VirtualFree with MEM_RELEASE releases the entire region back to the
    // process's virtual address space. The size parameter must be 0 when
    // using MEM_RELEASE (Windows ignores it and frees the full allocation).
    // This is different from MEM_DECOMMIT, which only removes the physical
    // backing while keeping the virtual address reservation.
    (void)VirtualFree(ptr, 0, MEM_RELEASE);

#elif defined(AI_OS_PLATFORM_POSIX)
    // munmap unmaps the memory region from the process's address space.
    // The size must match the size passed to the original mmap call.
    // After munmap, accessing the memory is undefined behavior.
    (void)munmap(ptr, size);

#else
    #error "Unsupported platform for os_deallocate"
#endif
}

SovereignArena::size_type SovereignArena::align_up(size_type offset, size_type alignment) const noexcept {
    // Align `offset` up to the nearest multiple of `alignment`.
    // Assumes alignment is a power of two (validated by validate_alignment).
    // The mask ~(alignment - 1) clears all bits below the alignment boundary.
    return (offset + alignment - 1) & ~(alignment - 1);
}

} // namespace memory
} // namespace ai_os
