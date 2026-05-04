/**
 * @file platform.hpp
 * @brief AI OS Core - Platform Detection and Abstraction Layer
 *
 * This header provides unified macros and type definitions for cross-platform
 * memory management operations. It abstracts the differences between Windows
 * (VirtualAlloc/VirtualLock) and POSIX (mmap/mlock) APIs so that the rest
 * of the codebase can operate agnostically of the host operating system.
 *
 * Design Rationale:
 *   - Low-level OS calls are isolated behind macros and inline wrappers
 *   - All conditional compilation is confined to this single translation unit
 *   - Downstream headers and source files use AI_OS_PLATFORM_* exclusively
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 */

#pragma once

///////////////////////////////////////////////////////////////////////////////
//  Compiler & Standard Library Detection
///////////////////////////////////////////////////////////////////////////////

#include <cstddef>

// Ensure C++17 or later
#if __cplusplus < 201703L
    #error "AI OS Core requires C++17 or later. Compile with -std=c++17 or higher."
#endif

///////////////////////////////////////////////////////////////////////////////
//  Platform Identification Macros
///////////////////////////////////////////////////////////////////////////////

#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
    /**
     * @def AI_OS_PLATFORM_WINDOWS
     * @brief Defined when compiling for Microsoft Windows (32-bit or 64-bit).
     */
    #define AI_OS_PLATFORM_WINDOWS 1

    // Pull in Windows system headers for memory management
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>

    // Windows-specific memory operation macros
    #define AI_OS_MEM_ALLOC(size)          VirtualAlloc(nullptr, (size), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    #define AI_OS_MEM_FREE(ptr, size)      VirtualFree((ptr), 0, MEM_RELEASE)
    #define AI_OS_MEM_LOCK(ptr, size)      VirtualLock((ptr), (size))
    #define AI_OS_MEM_UNLOCK(ptr, size)    VirtualUnlock((ptr), (size))

#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    /**
     * @def AI_OS_PLATFORM_POSIX
     * @brief Defined when compiling for any POSIX-compliant system (Linux, macOS, BSD, etc.).
     */
    #define AI_OS_PLATFORM_POSIX 1

    // Pull in POSIX system headers for memory management
    #include <sys/mman.h>
    #include <unistd.h>

    // POSIX-specific memory operation macros
    #define AI_OS_MEM_ALLOC(size)          mmap(nullptr, (size), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    #define AI_OS_MEM_FREE(ptr, size)      munmap((ptr), (size))
    #define AI_OS_MEM_LOCK(ptr, size)      mlock((ptr), (size))
    #define AI_OS_MEM_UNLOCK(ptr, size)    munlock((ptr), (size))

#else
    #error "AI OS Core: Unsupported platform. Only Windows and POSIX-compliant systems are supported."
#endif

///////////////////////////////////////////////////////////////////////////////
//  System Page Size Query
///////////////////////////////////////////////////////////////////////////////

namespace ai_os {
namespace platform {

/**
 * @brief Returns the system's virtual memory page size in bytes.
 *
 * On Windows, this delegates to GetSystemInfo().dwPageSize.
 * On POSIX systems, this delegates to sysconf(_SC_PAGESIZE).
 *
 * The page size is needed to correctly align memory allocations
 * for optimal performance and to satisfy mlock/VirtualLock requirements
 * (which operate on page-aligned boundaries).
 *
 * @return System page size in bytes (typically 4096 on most architectures).
 */
inline std::size_t page_size() noexcept {
#if defined(AI_OS_PLATFORM_WINDOWS)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<std::size_t>(si.dwPageSize);
#else
    static const long ps = sysconf(_SC_PAGESIZE);
    return static_cast<std::size_t>(ps > 0 ? ps : 4096);
#endif
}

/**
 * @brief Aligns a byte size up to the nearest system page boundary.
 *
 * Many low-level memory operations (mlock, VirtualLock) require or perform
 * optimally when sizes are page-aligned. This utility rounds up any
 * arbitrary byte count to the next multiple of the system page size.
 *
 * @param bytes The unaligned byte count.
 * @return The page-aligned byte count (>= bytes).
 */
inline std::size_t align_to_page(std::size_t bytes) noexcept {
    const std::size_t ps = page_size();
    return (bytes + ps - 1) & ~(ps - 1);
}

} // namespace platform
} // namespace ai_os
