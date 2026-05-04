/**
 * @file async_prefetcher.hpp
 * @brief AI OS Core Phase 2 - Asynchronous Layer Prefetcher
 *
 * AsyncPrefetcher implements a background I/O worker that reads model layer
 * data from disk directly into the SlidingWindowManager's memory region,
 * operating in parallel with the main inference thread. This enables I/O
 * latency hiding: while the CPU processes layer N, the disk preloads layer N+1
 * into RAM, achieving near-zero pipeline stalls.
 *
 * Architectural Position in AI OS Core:
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │                    Main Inference Thread                      │
 *   │  wait_for_layer(N) → commit → process → retire → signal       │
 *   └──────────────┬──────────────────────────────▲────────────────┘
 *                  │ m_cv_fetched               │ m_cv_consumed
 *                  │ (layer ready)               │ (layer consumed)
 *   ┌──────────────▼──────────────────────────────┴────────────────┐
 *   │                   Background I/O Thread                       │
 *   │  prefetch_loop: prepare_layer(N) → ifstream.read() → signal  │
 *   └──────────────┬───────────────────────────────────────────────┘
 *                  │ Zero-Copy Read (ifstream → SlidingWindow RAM)
 *   ┌──────────────▼───────────────────────────────────────────────┐
 *   │              SlidingWindowManager (Phase 1)                   │
 *   │         Ring buffer within SovereignArena locked RAM           │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * Key Design Principles:
 *
 *   1. ZERO-COPY: Data flows from ifstream directly into the address returned
 *      by SlidingWindowManager::prepare_layer(). No intermediate std::vector,
 *      no temporary buffers. The only buffering is the ifstream's internal
 *      4MB read-ahead buffer (set via pubsetbuf) for sequential HDD optimization.
 *
 *   2. PIPELINE DEPTH = 1: The background thread stays exactly one layer ahead
 *      of the main thread. While the CPU processes layer N, the disk loads N+1.
 *      This minimizes memory usage while hiding I/O latency.
 *
 *   3. BACKPRESSURE: If the SlidingWindowManager's ring buffer is full
 *      (WindowOverflowError), the background thread blocks and waits for the
 *      main thread to retire layers. This prevents unbounded memory growth and
 *      ensures the hard memory budget is never exceeded.
 *
 *   4. GRACEFUL SHUTDOWN: The destructor safely joins the background thread,
 *      even if it is waiting on a condition variable. All CVs are notified
 *      during stop(), ensuring no deadlocks during destruction.
 *
 * AIOC Model File Format:
 *   The prefetcher expects a binary model file in the AIOC format:
 *
 *     Offset  Size      Field               Description
 *     0x00    4 bytes   magic               0x41494F43 ("AIOC" in ASCII)
 *     0x04    4 bytes   version             File format version (currently 1)
 *     0x08    4 bytes   num_layers          Total number of layers
 *     0x0C    N×8 bytes layer_sizes         uint64_t array of per-layer byte sizes
 *     0x0C+   variable  layer_data          Concatenated raw layer weight data
 *            N×8
 *
 *   Header size = 12 + (num_layers × 8) bytes.
 *   Layer data follows immediately after the header, with each layer's data
 *   exactly layer_sizes[i] bytes long, concatenated in order.
 *
 * Thread Safety:
 *   The class is designed for exactly TWO concurrent threads:
 *     - Thread A (Main): calls start(), wait_for_layer(), commit_fetched_layer(),
 *       retire_and_signal(), stop()
 *     - Thread B (Background): runs prefetch_loop() internally
 *   The internal mutex (m_mutex) protects shared state. The SlidingWindowManager
 *   has its own internal mutex (accessed via swm.guard()).
 *
 *   Lock ordering to prevent deadlock:
 *     1. m_mutex (AsyncPrefetcher's own mutex)
 *     2. SlidingWindowManager's mutex (via swm.guard())
 *     These are NEVER held simultaneously across threads.
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 *
 * @see SlidingWindowManager  (Phase 1) — Ring buffer layer manager
 * @see SovereignArena       (Phase 1) — Locked memory arena
 * @see exceptions.hpp         — WindowOverflowError, InvalidLayerSizeError
 */

#pragma once

#include "ai_os/sliding_window_manager.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ai_os {
namespace io {

// =============================================================================
//  AIOC Model File Format Constants
// =============================================================================

/**
 * @def AIOC_FILE_MAGIC
 * @brief Magic number identifying an AIOC model file (0x41494F43 = "AIOC").
 *
 * This 4-byte value appears at offset 0x00 of every valid AIOC file.
 * Tooling and model converters must write this magic number to produce
 * compatible files.
 */
constexpr std::uint32_t AIOC_FILE_MAGIC = 0x41494F43;

/**
 * @def AIOC_FILE_VERSION
 * @brief Current file format version (1).
 *
 * Future versions may add fields to the header. Readers should validate
 * this version and reject files with unsupported versions.
 */
constexpr std::uint32_t AIOC_FILE_VERSION = 1;

/**
 * @def AIOC_HEADER_BASE_SIZE
 * @brief Size of the fixed portion of the AIOC header in bytes (magic + version + num_layers).
 */
constexpr std::size_t AIOC_HEADER_BASE_SIZE = 12;

// =============================================================================
//  ModelFileHeader — Parsed representation of the AIOC file header
// =============================================================================

/**
 * @struct ModelFileHeader
 * @brief Stores the parsed header information from an AIOC model file.
 *
 * This struct is populated by validate_and_parse_header() during construction
 * and provides quick access to file metadata without re-reading the file.
 */
struct ModelFileHeader {
    std::uint32_t magic{0};       ///< File magic number (should be AIOC_FILE_MAGIC)
    std::uint32_t version{0};     ///< File format version
    std::uint32_t num_layers{0};  ///< Total number of model layers
};

// =============================================================================
//  LayerMetadata — Per-layer information for diagnostics and query
// =============================================================================

/**
 * @struct LayerMetadata
 * @brief Stores metadata about a single model layer.
 *
 * Includes the layer's size in bytes and its byte offset within the
 * AIOC file's data section (after the header).
 */
struct LayerMetadata {
    std::uint32_t layer_id{0};   ///< Sequential layer identifier (0-indexed)
    std::uint64_t size{0};       ///< Size of this layer's data in bytes
    std::uint64_t file_offset{0}; ///< Byte offset of this layer's data in the file
};

// =============================================================================
//  PrefetcherState — Lifecycle state of the AsyncPrefetcher
// =============================================================================

/**
 * @enum PrefetcherState
 * @brief Represents the current operational state of the AsyncPrefetcher.
 *
 * State Machine:
 *
 *   Idle ──start()──▶ Running ──pause()──▶ Paused
 *     ▲                  │    ▲              │
 *     │            stop()│    │resume()       │stop()
 *     │                  │    │              │
 *     │                  ▼    │              ▼
 *     └────────────── Stopped ◀───────────────┘
 *     │
 *     └──── (error during operation) ──── Error
 *     │
 *     └──── (all layers fetched) ──────── Completed
 */
enum class PrefetcherState : std::uint8_t {
    Idle,       ///< Constructed but not started
    Running,    ///< Background thread is actively fetching
    Paused,     ///< Background thread is suspended (backpressure)
    Stopped,    ///< Background thread has been stopped
    Error,      ///< An I/O or synchronization error occurred
    Completed   ///< All layers have been successfully fetched
};

/**
 * @brief Returns a human-readable string for a PrefetcherState value.
 * @param state The state to convert.
 * @return Static C-string description.
 */
inline const char* prefetcher_state_string(PrefetcherState state) noexcept {
    switch (state) {
        case PrefetcherState::Idle:      return "Idle";
        case PrefetcherState::Running:   return "Running";
        case PrefetcherState::Paused:    return "Paused";
        case PrefetcherState::Stopped:   return "Stopped";
        case PrefetcherState::Error:     return "Error";
        case PrefetcherState::Completed: return "Completed";
        default:                         return "Unknown";
    }
}

// =============================================================================
//  AsyncPrefetcher — Background I/O Prefetch Worker
// =============================================================================

/**
 * @class AsyncPrefetcher
 * @brief Reads model layer data from disk in a background thread, directly
 *        into SlidingWindowManager's memory region (zero-copy).
 *
 * The AsyncPrefetcher is the I/O subsystem of the AI OS Core. It bridges
 * the gap between slow disk storage (HDD/SSD) and the fast locked RAM arena
 * by preloading layers in parallel with inference computation.
 *
 * Typical Usage Pattern:
 * @code
 *   // Setup (Phase 1)
 *   SovereignArena arena(500 * 1024 * 1024);
 *   void* region = arena.allocate(500 * 1024 * 1024);
 *   SlidingWindowManager swm(region, 500 * 1024 * 1024);
 *
 *   // Phase 2: Start prefetching
 *   AsyncPrefetcher prefetcher(swm, "model.aioc");
 *   prefetcher.start();
 *
 *   // Inference loop
 *   for (uint32_t i = 0; i < prefetcher.total_layers(); ++i) {
 *       // Wait for background thread to load layer i into RAM
 *       prefetcher.wait_for_layer(i);
 *
 *       // Commit: Preparing → Active (makes the layer readable)
 *       prefetcher.commit_fetched_layer();
 *
 *       // Process the layer (inference computation)
 *       inference_engine.process(swm.active_layer_address());
 *
 *       // Retire and signal background thread to load next layer
 *       prefetcher.retire_and_signal();
 *   }
 *
 *   // Cleanup
 *   prefetcher.stop();
 * @endcode
 *
 * Synchronization Protocol:
 *   The two threads communicate via a handshake mechanism:
 *
 *   1. BG Thread prepares a layer slot (Preparing state)
 *   2. BG Thread reads data directly into the slot's memory (zero-copy)
 *   3. BG Thread sets m_latest_fetched_layer and notifies m_cv_fetched
 *   4. BG Thread waits on m_cv_consumed for the main thread's signal
 *   5. Main Thread wakes up from wait_for_layer() when notified
 *   6. Main Thread commits the layer (Preparing → Active)
 *   7. Main Thread processes the layer
 *   8. Main Thread retires the layer (Active → Retired)
 *   9. Main Thread increments m_consumed_up_to and notifies m_cv_consumed
 *   10. BG Thread wakes up and prepares the next layer
 *   11. Repeat from step 2
 *
 * HDD Optimization:
 *   A 4MB internal buffer is set on the ifstream via pubsetbuf(). This forces
 *   the OS to issue large sequential reads, which is critical for HDD performance
 *   (100+ MB/s sequential vs <1 MB/s random). The buffer also reduces system
 *   call overhead by coalescing many small reads into fewer large ones.
 */
class AsyncPrefetcher {
public:
    // =========================================================================
    //  Type Aliases
    // =========================================================================

    using size_type = std::size_t;

    // =========================================================================
    //  Construction / Destruction
    // =========================================================================

    /**
     * @brief Constructs an AsyncPrefetcher bound to a SlidingWindowManager and model file.
     *
     * This constructor:
     *   1. Stores a reference to the SlidingWindowManager (non-owning).
     *   2. Opens the model file in binary mode with a 4MB internal I/O buffer.
     *   3. Reads and validates the AIOC file header (magic, version, num_layers).
     *   4. Parses the per-layer size table to prepare for sequential reads.
     *
     * The background thread is NOT started automatically. Call start() to begin.
     *
     * @param swm       Reference to a SlidingWindowManager (must outlive this object).
     * @param file_path Path to the AIOC model file on disk.
     *
     * @throws std::invalid_argument  If file_path is empty.
     * @throws std::runtime_error    If the file cannot be opened.
     * @throws std::runtime_error    If the file header is invalid (wrong magic, version, etc.).
     * @throws std::runtime_error    If the file is too small to contain the declared layers.
     */
    explicit AsyncPrefetcher(memory::SlidingWindowManager& swm,
                             const std::string& file_path);

    /**
     * @brief Destructor. Stops the background thread and releases resources.
     *
     * If the background thread is running, it is stopped and joined before
     * the object is destroyed. This ensures no dangling thread references.
     * The destructor is safe to call from any state.
     */
    ~AsyncPrefetcher();

    // Non-copyable (owns a thread and file handle)
    AsyncPrefetcher(const AsyncPrefetcher&) = delete;
    AsyncPrefetcher& operator=(const AsyncPrefetcher&) = delete;

    // Non-movable (thread and file handle cannot be safely transferred)
    AsyncPrefetcher(AsyncPrefetcher&&) = delete;
    AsyncPrefetcher& operator=(AsyncPrefetcher&&) = delete;

    // =========================================================================
    //  Control Interface
    // =========================================================================

    /**
     * @brief Starts the background prefetch thread.
     *
     * Spawns a std::thread that runs prefetch_loop(). The thread immediately
     * begins loading the first layer from disk into the SlidingWindowManager.
     *
     * @throws std::logic_error If the prefetcher is already running or completed.
     * @throws std::runtime_error If the thread fails to spawn.
     */
    void start();

    /**
     * @brief Stops the background prefetch thread gracefully.
     *
     * Sets the internal running flag to false, notifies all condition variables
     * to wake the background thread from any wait state, and joins the thread.
     * After this call, the prefetcher is in the Stopped state and cannot be
     * restarted (create a new AsyncPrefetcher instead).
     *
     * This method is safe to call from any thread and from any state. If the
     * thread is not running, it is a no-op.
     */
    void stop();

    /**
     * @brief Pauses the background prefetch thread (backpressure).
     *
     * The background thread will stop fetching after completing its current
     * layer read. It enters the Paused state and waits on a condition variable.
     * Call resume() to allow it to continue.
     *
     * This is useful when the inference engine needs to throttle I/O to
     * reduce memory pressure or CPU cache contention.
     *
     * @throws std::logic_error If the prefetcher is not running.
     */
    void pause();

    /**
     * @brief Resumes a paused background prefetch thread.
     *
     * Wakes the background thread from its paused state, allowing it to
     * continue fetching layers from where it left off.
     *
     * @throws std::logic_error If the prefetcher is not paused.
     */
    void resume();

    // =========================================================================
    //  Main Thread Interface (Inference Thread)
    // =========================================================================

    /**
     * @brief Blocks the calling thread until the specified layer has been fetched.
     *
     * This is the primary synchronization point between the main inference
     * thread and the background I/O thread. The calling thread is suspended
     * via condition_variable.wait() until:
     *   (a) The background thread has finished loading the target layer
 *         (m_latest_fetched_layer >= target_layer_id), OR
     *   (b) The prefetcher has been stopped, OR
     *   (c) The prefetcher has completed all layers.
     *
     * If the target layer has ALREADY been fetched, this function returns
     * immediately (zero latency) — it only blocks when the disk is slower
     * than the CPU.
     *
     * @param target_layer_id The 0-indexed layer ID to wait for.
     *
     * @throws std::runtime_error If the prefetcher was stopped before the
     *         target layer could be fetched.
     * @throws std::runtime_error If all layers were fetched but the target
     *         layer was not among them (indicates a logic error).
     */
    void wait_for_layer(std::int32_t target_layer_id);

    /**
     * @brief Commits the most recently fetched layer, making it the active layer.
     *
     * Delegates to SlidingWindowManager::commit_layer(), transitioning the
     * fetched layer from Preparing → Active. After this call, the layer's
     * data is available for read-only inference computation via
     * SlidingWindowManager::active_layer_address().
     *
     * Must be called after wait_for_layer() and before any processing.
     *
     * @throws std::logic_error If no layer is in the Preparing state.
     */
    void commit_fetched_layer();

    /**
     * @brief Retires the active layer and signals the background thread to continue.
     *
     * This method combines two operations:
     *   1. Retires the current active layer (Active → Retired) in the SWM.
     *   2. Increments m_consumed_up_to and notifies the background thread
     *      via m_cv_consumed and m_cv_state_change.
     *
     * After this call, the background thread wakes up and begins fetching
     * the next layer. This should be called after inference processing is
     * complete.
     *
     * @note If no layer is currently active (e.g., for the very first call
     *       before any commit), the retire step is silently skipped. Only
     *       the signal is sent.
     */
    void retire_and_signal();

    // =========================================================================
    //  Query Interface
    // =========================================================================

    /**
     * @brief Returns the current operational state of the prefetcher.
     * @return Current PrefetcherState value.
     */
    [[nodiscard]] PrefetcherState state() const noexcept;

    /**
     * @brief Returns the ID of the most recently fetched layer.
     * @return Layer ID (0-indexed), or -1 if no layer has been fetched yet.
     */
    [[nodiscard]] std::int32_t latest_fetched_layer() const noexcept;

    /**
     * @brief Returns the ID of the most recently consumed (retired + signaled) layer.
     * @return Layer ID (0-indexed), or -1 if no layer has been consumed yet.
     */
    [[nodiscard]] std::int32_t consumed_up_to() const noexcept;

    /**
     * @brief Returns the total number of layers declared in the model file.
     * @return Layer count.
     */
    [[nodiscard]] std::uint32_t total_layers() const noexcept;

    /**
     * @brief Returns the byte size of a specific layer.
     * @param layer_id 0-indexed layer identifier.
     * @return Layer size in bytes.
     * @throws std::out_of_range If layer_id >= total_layers().
     */
    [[nodiscard]] std::uint64_t layer_size(std::uint32_t layer_id) const;

    /**
     * @brief Returns the byte offset of a specific layer within the file.
     * @param layer_id 0-indexed layer identifier.
     * @return Byte offset from the start of the file.
     * @throws std::out_of_range If layer_id >= total_layers().
     */
    [[nodiscard]] std::uint64_t layer_file_offset(std::uint32_t layer_id) const;

    /**
     * @brief Returns the total size of all layer data combined.
     * @return Total bytes of layer data (excluding header).
     */
    [[nodiscard]] std::uint64_t total_data_size() const noexcept;

    /**
     * @brief Returns the byte offset where layer data begins in the file.
     * @return Offset of the first byte of layer data.
     */
    [[nodiscard]] std::uint64_t data_start_offset() const noexcept;

    /**
     * @brief Checks whether a prefetch error has occurred.
     * @return true if an error was encountered during background fetching.
     */
    [[nodiscard]] bool has_error() const noexcept;

    /**
     * @brief Returns the error message, if any.
     * @return Error description string, or empty if no error.
     */
    [[nodiscard]] const std::string& error_message() const noexcept;

    /**
     * @brief Returns whether all layers have been fetched.
     * @return true if the background thread has completed all fetches.
     */
    [[nodiscard]] bool is_fetch_complete() const noexcept;

    /**
     * @brief Returns whether the background thread is currently running.
     * @return true if the thread is active.
     */
    [[nodiscard]] bool is_running() const noexcept;

    // =========================================================================
    //  Diagnostic Interface
    // =========================================================================

    /**
     * @brief Returns the file path of the model being prefetched.
     * @return Const reference to the file path string.
     */
    [[nodiscard]] const std::string& file_path() const noexcept;

    /**
     * @brief Returns the size of the internal I/O buffer in bytes.
     * @return Buffer size (default: 4MB = 4,194,304 bytes).
     */
    [[nodiscard]] std::size_t io_buffer_size() const noexcept;

    /**
     * @brief Generates a comprehensive diagnostic report of the prefetcher's state.
     * @return Multi-line string with all state details, layer metadata, and statistics.
     */
    [[nodiscard]] std::string diagnostics() const;

private:
    // =========================================================================
    //  Private Methods — Background Thread
    // =========================================================================

    /**
     * @brief The main loop executed by the background I/O thread.
     *
     * This function runs in a separate std::thread and is responsible for:
     *   1. Checking pause/stop conditions.
     *   2. Calling swm.prepare_layer() to reserve memory for the next layer.
     *   3. Reading layer data from the file directly into the reserved memory
     *      (zero-copy via ifstream::read()).
     *   4. Updating m_latest_fetched_layer and notifying the main thread.
     *   5. Waiting for the main thread to consume the layer before continuing.
     *
     * The loop terminates when: all layers are fetched, stop() is called,
     * or an unrecoverable I/O error occurs.
     */
    void prefetch_loop();

    // =========================================================================
    //  Private Methods — File Operations
    // =========================================================================

    /**
     * @brief Opens the model file, reads and validates the AIOC header.
     *
     * Reads the first 12 bytes (magic + version + num_layers), validates
     * them against expected values, then reads the layer size array.
     *
     * @throws std::runtime_error On any validation failure.
     */
    void validate_and_parse_header();

    /**
     * @brief Seeks the file read position to the start of a specific layer's data.
     * @param layer_id 0-indexed layer identifier.
     * @throws std::out_of_range If layer_id is invalid.
     * @throws std::runtime_error If the seek operation fails.
     */
    void seek_to_layer(std::uint32_t layer_id);

    // =========================================================================
    //  Private Methods — State Management
    // =========================================================================

    /**
     * @brief Atomically updates the prefetcher's state and notifies waiters.
     * @param new_state The new state to transition to.
     */
    void set_state(PrefetcherState new_state);

    // =========================================================================
    //  Member Variables
    // =========================================================================

    // --- External References (non-owning) ---
    memory::SlidingWindowManager& m_swm;  ///< Reference to the ring buffer manager

    // --- File I/O ---
    std::string      m_file_path;   ///< Path to the AIOC model file
    std::ifstream    m_file;        ///< Binary input file stream
    std::vector<char> m_io_buffer;  ///< Large internal buffer for sequential HDD reads

    // --- File Metadata (parsed from header) ---
    ModelFileHeader          m_header;         ///< Parsed AIOC file header
    std::vector<std::uint64_t> m_layer_sizes;  ///< Per-layer byte sizes
    std::vector<std::uint64_t> m_layer_offsets;///< Per-layer byte offsets in file
    std::uint64_t            m_data_start_offset; ///< Byte offset where data section begins

    // --- Threading ---
    std::thread       m_worker;       ///< Background I/O thread handle
    mutable std::mutex m_mutex;       ///< Protects condition variable waits

    // --- Condition Variables ---
    std::condition_variable m_cv_fetched;       ///< BG→Main: "a new layer is ready"
    std::condition_variable m_cv_consumed;      ///< Main→BG: "I'm done, continue"
    std::condition_variable m_cv_state_change;  ///< General: pause/resume/stop signals

    // --- Atomic Synchronization Primitives ---
    /**
     * @brief ID of the most recently fetched layer (0-indexed), or -1 if none.
     *
     * Written by BG thread after each successful read.
     * Read by Main thread in wait_for_layer().
     * Uses release/acquire semantics to ensure the layer data write is
     * visible before the flag is set.
     */
    std::atomic<std::int32_t> m_latest_fetched_layer{-1};

    /**
     * @brief ID of the most recently consumed (retired) layer, or -1 if none.
     *
     * Written by Main thread after retiring a layer.
     * Read by BG thread to determine when it's safe to prepare the next layer.
     */
    std::atomic<std::int32_t> m_consumed_up_to{-1};

    /** @brief True while the background thread should continue running. */
    std::atomic<bool> m_running{false};

    /** @brief True when the background thread is paused (backpressure). */
    std::atomic<bool> m_paused{false};

    /** @brief True when all layers have been fetched (normal completion). */
    std::atomic<bool> m_fetch_complete{false};

    // --- Error State ---
    // [FIX #6] m_error كان plain bool يُكتب من BG thread ويُقرأ من Main thread
    // بدون أي حماية → Data Race مضمون. الحل: std::atomic<bool> للـ flag،
    // و m_error_msg محمي بـ m_mutex عند الكتابة والقراءة.
    std::atomic<bool> m_error{false}; ///< True if an error occurred (atomic - thread-safe)
    std::string m_error_msg;          ///< Human-readable error description (protected by m_mutex)

    // --- Lifecycle State ---
    std::atomic<PrefetcherState> m_state{PrefetcherState::Idle};

    // --- Constants ---
    static constexpr std::size_t IO_BUFFER_SIZE = 4 * 1024 * 1024; ///< 4MB I/O buffer
};

} // namespace io
} // namespace ai_os
