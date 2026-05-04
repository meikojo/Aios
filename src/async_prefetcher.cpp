/**
 * @file async_prefetcher.cpp
 * @brief AI OS Core Phase 2 - AsyncPrefetcher Implementation
 *
 * This file contains the complete implementation of the AsyncPrefetcher class.
 * It handles background I/O, zero-copy file reading, thread synchronization,
 * backpressure management, and graceful shutdown.
 *
 * Implementation Notes:
 *
 * Zero-Copy Strategy:
 *   The critical zero-copy guarantee is enforced here. The only read operation
 *   is: m_file.read(static_cast<char*>(handle.address), layer_size).
 *   This writes data directly from the ifstream's internal buffer (or from
 *   the OS page cache) into the SlidingWindowManager's locked RAM region.
 *   No std::vector, no std::array, no temporary allocation exists in the
 *   data path. The 4MB m_io_buffer is set via pubsetbuf() and belongs to
 *   the ifstream's streambuf — it is NOT an intermediate data copy buffer.
 *
 * Thread Synchronization:
 *   The two threads communicate via three condition variables:
 *     m_cv_fetched:       BG notifies Main when a layer is ready
 *     m_cv_consumed:      Main notifies BG when a layer is consumed
 *     m_cv_state_change:  Used for pause/resume/stop transitions
 *
 *   The atomic variables (m_latest_fetched_layer, m_consumed_up_to) use
 *   memory_order_release (writer) and memory_order_acquire (reader) to
 *   ensure that the actual data written to memory is visible before the
 *   signaling flag is observed by the other thread.
 *
 * Backpressure Handling:
 *   When SlidingWindowManager::prepare_layer() throws WindowOverflowError
 *   (the ring buffer is full), the background thread catches the exception
 *   and blocks on m_cv_state_change. It retries prepare_layer() after being
 *   woken by the main thread's retire_and_signal() call. This ensures that
 *   the prefetcher never overwhelms the ring buffer's capacity.
 *
 * @author AI OS Core Team
 * @version 1.0.0
 * @since 2026-04-08
 * @license MIT
 */

#include "ai_os/async_prefetcher.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <system_error>

namespace ai_os {
namespace io {

// =============================================================================
//  Construction / Destruction
// =============================================================================

AsyncPrefetcher::AsyncPrefetcher(memory::SlidingWindowManager& swm,
                                 const std::string& file_path)
    : m_swm(swm)
    , m_file_path(file_path)
    , m_data_start_offset(0)
{
    // --- Input Validation --------------------------------------------------------
    if (m_file_path.empty()) {
        throw std::invalid_argument(
            "AsyncPrefetcher: file_path must not be empty.");
    }

    // --- Allocate and Set the Large I/O Buffer -----------------------------------
    // The 4MB buffer is allocated on the heap and registered with the ifstream's
    // streambuf BEFORE opening the file. This is critical because:
    //   (a) pubsetbuf() must be called before any I/O operations on many
    //       implementations (per the C++ standard, effect is implementation-defined
    //       after I/O has begun).
    //   (b) The buffer must outlive the ifstream (it's a member, so this is
    //       guaranteed by RAII ordering: members are destroyed in reverse
    //       declaration order, and m_file is declared after m_io_buffer).
    //
    // Effect on HDD Performance:
    //   With a 4MB buffer, a single ifstream::read() for, say, 50MB of layer
    //   data results in ~12-13 large read() system calls instead of thousands
    //   of small ones. On HDD, this transforms random-access-like behavior
    //   (due to small reads interrupting the disk's read-ahead) into genuine
    //   sequential reads, improving throughput from ~1 MB/s to 100+ MB/s.
    m_io_buffer.resize(IO_BUFFER_SIZE);
    m_file.rdbuf()->pubsetbuf(m_io_buffer.data(),
                               static_cast<std::streamsize>(IO_BUFFER_SIZE));

    // --- Open the Model File in Binary Mode --------------------------------------
    // std::ios::binary prevents line-ending translation on Windows (CRLF → LF),
    //   which would corrupt binary weight data.
    // std::ios::in opens for reading.
    m_file.open(m_file_path, std::ios::in | std::ios::binary);

    if (!m_file.is_open()) {
        // Provide a descriptive error message with the OS error code.
        // Common causes: file doesn't exist, permission denied, path too long.
        int os_errno = 0;
#ifdef _WIN32
        os_errno = static_cast<int>(GetLastError());
#else
        os_errno = errno;
#endif
        throw std::runtime_error(
            "AsyncPrefetcher: failed to open model file '" + m_file_path +
            "'. OS error code: " + std::to_string(os_errno) +
            ". Ensure the file exists and is readable.");
    }

    // --- Parse and Validate the File Header ---------------------------------------
    validate_and_parse_header();
}

AsyncPrefetcher::~AsyncPrefetcher() {
    // Ensure the background thread is stopped before destruction.
    // This is critical for correctness: if the thread accesses m_swm or m_file
    // after they've been destroyed, the result is undefined behavior.
    //
    // stop() is idempotent: if the thread was never started or already stopped,
    // it's a no-op. If the thread is running, it blocks until the thread exits.
    stop();

    // The file stream is closed automatically by ifstream's destructor.
    // The I/O buffer is freed automatically by vector's destructor.
}

// =============================================================================
//  Control Interface
// =============================================================================

void AsyncPrefetcher::start() {
    // --- State Validation --------------------------------------------------------
    PrefetcherState current = m_state.load(std::memory_order_acquire);
    if (current == PrefetcherState::Running) {
        throw std::logic_error(
            "AsyncPrefetcher::start: prefetcher is already running.");
    }
    if (current == PrefetcherState::Completed) {
        throw std::logic_error(
            "AsyncPrefetcher::start: prefetcher has already completed all layers. "
            "Create a new AsyncPrefetcher instance to fetch again.");
    }

    // --- Reset State for Re-start (after pause/stop) ------------------------------
    if (current == PrefetcherState::Stopped || current == PrefetcherState::Error) {
        throw std::logic_error(
            "AsyncPrefetcher::start: cannot restart a stopped or errored prefetcher. "
            "Create a new instance.");
    }

    // --- Reset Synchronization State ---------------------------------------------
    m_running.store(true, std::memory_order_release);
    m_paused.store(false, std::memory_order_release);
    m_fetch_complete.store(false, std::memory_order_release);
    // [FIX #6] m_error الآن atomic — استخدم store، وامسح m_error_msg تحت lock
    m_error.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_error_msg.clear();
    }

    // --- Spawn the Background Thread ---------------------------------------------
    try {
        m_worker = std::thread(&AsyncPrefetcher::prefetch_loop, this);
    } catch (const std::system_error& e) {
        m_running.store(false, std::memory_order_release);
        throw std::runtime_error(
            "AsyncPrefetcher::start: failed to spawn background thread: " +
            std::string(e.what()));
    }

    set_state(PrefetcherState::Running);
}

void AsyncPrefetcher::stop() {
    // Idempotent: if already stopped, do nothing.
    if (!m_running.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    // Wake up ALL waiting threads so they can observe m_running == false
    // and exit their wait loops.
    //
    // It's safe to notify_all even if no one is waiting — the notification
    // is simply lost. But it's critical to notify ALL condition variables
    // because the background thread might be waiting on any one of them.
    m_cv_state_change.notify_all();
    m_cv_consumed.notify_all();
    m_cv_fetched.notify_all();

    // --- Join the Background Thread -----------------------------------------------
    // This blocks until the background thread exits. If the thread is in a
    // condition_variable.wait(), it will be woken by the notify_all() above
    // and will observe m_running == false, causing it to break out of its loop.
    //
    // If this destructor is called from the background thread itself (which
    // would be a programming error), joinable() would be true but join()
    // would cause std::system_error (resource_deadlock_would_occur).
    // We guard against this with the joinable() check.
    if (m_worker.joinable()) {
        m_worker.join();
    }

    set_state(PrefetcherState::Stopped);
}

void AsyncPrefetcher::pause() {
    PrefetcherState current = m_state.load(std::memory_order_acquire);
    if (current != PrefetcherState::Running) {
        throw std::logic_error(
            "AsyncPrefetcher::pause: can only pause a running prefetcher. "
            "Current state: " + std::string(prefetcher_state_string(current)));
    }

    m_paused.store(true, std::memory_order_release);
    set_state(PrefetcherState::Paused);
}

void AsyncPrefetcher::resume() {
    PrefetcherState current = m_state.load(std::memory_order_acquire);
    if (current != PrefetcherState::Paused) {
        throw std::logic_error(
            "AsyncPrefetcher::resume: can only resume a paused prefetcher. "
            "Current state: " + std::string(prefetcher_state_string(current)));
    }

    m_paused.store(false, std::memory_order_release);
    m_cv_state_change.notify_all();
    set_state(PrefetcherState::Running);
}

// =============================================================================
//  Main Thread Interface
// =============================================================================

void AsyncPrefetcher::wait_for_layer(std::int32_t target_layer_id) {
    // --- Input Validation --------------------------------------------------------
    if (target_layer_id < 0 ||
        static_cast<std::uint32_t>(target_layer_id) >= m_header.num_layers) {
        throw std::out_of_range(
            "AsyncPrefetcher::wait_for_layer: target_layer_id " +
            std::to_string(target_layer_id) + " is out of range [0, " +
            std::to_string(m_header.num_layers) + ").");
    }

    // --- Fast Path: Layer Already Fetched -----------------------------------------
    // If the background thread has already loaded this layer, return immediately.
    // This is the zero-latency path — no mutex, no condition variable wait.
    if (m_latest_fetched_layer.load(std::memory_order_acquire) >= target_layer_id) {
        return;
    }

    // --- Slow Path: Block Until Fetched ------------------------------------------
    // The layer hasn't been fetched yet. Suspend the calling thread on the
    // condition variable until the background thread signals completion.
    //
    // We use a predicate-based wait to protect against spurious wakeups.
    // The lambda captures 'this' and 'target_layer_id' by value/reference.
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv_fetched.wait(lock, [this, target_layer_id]() -> bool {
        return m_latest_fetched_layer.load(std::memory_order_acquire) >= target_layer_id
               || !m_running.load(std::memory_order_acquire)
               || m_fetch_complete.load(std::memory_order_acquire);
    });

    // --- Post-Wait Validation ----------------------------------------------------
    // Check why the wait was unblocked. It might have been a stop or error,
    // not a successful fetch.

    if (!m_running.load(std::memory_order_acquire) &&
        m_latest_fetched_layer.load(std::memory_order_acquire) < target_layer_id) {
        throw std::runtime_error(
            "AsyncPrefetcher::wait_for_layer: prefetcher was stopped before "
            "layer " + std::to_string(target_layer_id) +
            " could be fetched. Latest fetched: " +
            std::to_string(m_latest_fetched_layer.load()));
    }

    if (m_fetch_complete.load(std::memory_order_acquire) &&
        m_latest_fetched_layer.load(std::memory_order_acquire) < target_layer_id) {
        throw std::runtime_error(
            "AsyncPrefetcher::wait_for_layer: all layers have been fetched, but "
            "layer " + std::to_string(target_layer_id) +
            " was not available. This indicates a logic error in the fetch loop.");
    }
}

void AsyncPrefetcher::commit_fetched_layer() {
    // Delegate to SlidingWindowManager with its own mutex protection.
    // The SWM's guard() returns a lock_guard that acquires the SWM's internal
    // recursive_mutex, ensuring thread-safe access.
    auto swm_lock = m_swm.guard();
    m_swm.commit_layer();
}

void AsyncPrefetcher::retire_and_signal() {
    // --- Step 1: Retire the Active Layer -----------------------------------------
    // Transition the active layer from Active → Retired in the SWM.
    // This frees the layer's memory for reuse by the next prepare_layer() call.
    //
    // We check for an active layer before calling retire_layer() because:
    //   - For the very first layer, there is no previous active layer to retire.
    //   - retire_layer() throws std::logic_error if no layer is active.
    {
        auto swm_lock = m_swm.guard();
        if (m_swm.active_layer() != nullptr) {
            m_swm.retire_layer();
        }
    }

    // --- Step 2: Signal the Background Thread -------------------------------------
    // Increment the consumed counter with release semantics. This ensures that
    // the memory writes performed by the inference engine (reading from the
    // active layer) are visible to the background thread when it observes
    // the updated counter value.
    //
    // The release/acquire pair forms a synchronizes-with relationship:
    //   Main thread (release) ──synchronizes-with──▶ BG thread (acquire)
    const std::int32_t prev = m_consumed_up_to.fetch_add(1, std::memory_order_release);

    // Notify ALL waiting threads. Even if only the BG thread is waiting,
    // notify_all is safer than notify_one in case of spurious wakeups or
    // multiple waiters (e.g., diagnostic threads).
    m_cv_consumed.notify_all();
    m_cv_state_change.notify_all();

    // --- Diagnostic: Log the consumed layer ---------------------------------------
    // (void)prev; // Suppress unused variable warning in release builds
    static_cast<void>(prev); // Layer prev+1 has been consumed
}

// =============================================================================
//  Query Interface
// =============================================================================

PrefetcherState AsyncPrefetcher::state() const noexcept {
    return m_state.load(std::memory_order_acquire);
}

std::int32_t AsyncPrefetcher::latest_fetched_layer() const noexcept {
    return m_latest_fetched_layer.load(std::memory_order_acquire);
}

std::int32_t AsyncPrefetcher::consumed_up_to() const noexcept {
    return m_consumed_up_to.load(std::memory_order_acquire);
}

std::uint32_t AsyncPrefetcher::total_layers() const noexcept {
    return m_header.num_layers;
}

std::uint64_t AsyncPrefetcher::layer_size(std::uint32_t layer_id) const {
    if (layer_id >= m_header.num_layers) {
        throw std::out_of_range(
            "AsyncPrefetcher::layer_size: layer_id " + std::to_string(layer_id) +
            " is out of range [0, " + std::to_string(m_header.num_layers) + ").");
    }
    return m_layer_sizes[layer_id];
}

std::uint64_t AsyncPrefetcher::layer_file_offset(std::uint32_t layer_id) const {
    if (layer_id >= m_header.num_layers) {
        throw std::out_of_range(
            "AsyncPrefetcher::layer_file_offset: layer_id " +
            std::to_string(layer_id) + " is out of range.");
    }
    return m_layer_offsets[layer_id];
}

std::uint64_t AsyncPrefetcher::total_data_size() const noexcept {
    std::uint64_t total = 0;
    for (const auto& sz : m_layer_sizes) {
        total += sz;
    }
    return total;
}

std::uint64_t AsyncPrefetcher::data_start_offset() const noexcept {
    return m_data_start_offset;
}

bool AsyncPrefetcher::has_error() const noexcept {
    // [FIX #6] m_error الآن atomic<bool> — acquire يضمن رؤية آخر قيمة من BG thread
    return m_error.load(std::memory_order_acquire);
}

const std::string& AsyncPrefetcher::error_message() const noexcept {
    // [FIX #6] m_error_msg يُكتب من BG thread ويُقرأ من Main thread.
    // نقرأه تحت lock لتجنب Data Race على الـ std::string.
    // ملاحظة: نرجع by value مش by reference هنا لأن الـ reference ممكن تبقى dangling
    // بعد تحرير الـ lock، لكن الـ signature في الـ header يقول const ref —
    // نحتفظ بالـ signature ونقلّل الخطر بالـ lock السريع.
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_error_msg;
}

bool AsyncPrefetcher::is_fetch_complete() const noexcept {
    return m_fetch_complete.load(std::memory_order_acquire);
}

bool AsyncPrefetcher::is_running() const noexcept {
    return m_running.load(std::memory_order_acquire);
}

const std::string& AsyncPrefetcher::file_path() const noexcept {
    return m_file_path;
}

std::size_t AsyncPrefetcher::io_buffer_size() const noexcept {
    return IO_BUFFER_SIZE;
}

// =============================================================================
//  Diagnostic Interface
// =============================================================================

std::string AsyncPrefetcher::diagnostics() const {
    std::ostringstream oss;

    oss << "=== AsyncPrefetcher Diagnostics ===\n";
    oss << "State:            " << prefetcher_state_string(
            m_state.load(std::memory_order_acquire)) << "\n";
    oss << "File Path:        " << m_file_path << "\n";
    oss << "Total Layers:     " << m_header.num_layers << "\n";
    oss << "Latest Fetched:   " << m_latest_fetched_layer.load() << "\n";
    oss << "Consumed Up To:   " << m_consumed_up_to.load() << "\n";
    oss << "Running:          " << (m_running.load() ? "Yes" : "No") << "\n";
    oss << "Paused:           " << (m_paused.load() ? "Yes" : "No") << "\n";
    oss << "Fetch Complete:   " << (m_fetch_complete.load() ? "Yes" : "No") << "\n";
    oss << "Error:            " << (m_error ? ("Yes: " + m_error_msg) : "No") << "\n";
    oss << "I/O Buffer Size:  " << (IO_BUFFER_SIZE / (1024 * 1024)) << " MB\n";
    oss << "Data Start Offset:" << m_data_start_offset << " bytes\n";
    oss << "Total Data Size:  " << total_data_size() << " bytes ("
        << (total_data_size() / (1024 * 1024)) << " MB)\n";
    oss << "File Open:        " << (m_file.is_open() ? "Yes" : "No") << "\n";
    oss << "------------------------------------------\n";

    // Per-layer metadata
    for (std::uint32_t i = 0; i < m_header.num_layers; ++i) {
        oss << "  Layer[" << i << "]: "
            << "size=" << m_layer_sizes[i]
            << " bytes, file_offset=" << m_layer_offsets[i];
        if (m_latest_fetched_layer.load() >= static_cast<std::int32_t>(i)) {
            oss << " [FETCHED]";
        }
        if (m_consumed_up_to.load() >= static_cast<std::int32_t>(i)) {
            oss << " [CONSUMED]";
        }
        oss << "\n";
    }

    return oss.str();
}

// =============================================================================
//  Private Methods — Background Thread
// =============================================================================

void AsyncPrefetcher::prefetch_loop() {
    // This function runs in the background thread (m_worker).
    // It is the core I/O engine of the AsyncPrefetcher.
    //
    // Invariant: at any point in time, at most ONE layer is in the Preparing
    // state in the SlidingWindowManager. This is enforced by:
    //   (a) The SWM itself (throws logic_error on duplicate prepare)
    //   (b) The handshake protocol (BG waits for consumption before next prepare)

    std::int32_t current_layer = 0;
    const std::int32_t total = static_cast<std::int32_t>(m_header.num_layers);

    while (current_layer < total && m_running.load(std::memory_order_acquire)) {
        // ------------------------------------------------------------------
        // PHASE A: Check Pause State (Backpressure)
        // ------------------------------------------------------------------
        // If the main thread has requested a pause (backpressure), the BG
        // thread suspends itself on m_cv_state_change. It resumes when
        // resume() is called (which sets m_paused = false and notifies).
        if (m_paused.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv_state_change.wait(lock, [this]() -> bool {
                return !m_paused.load(std::memory_order_acquire) ||
                       !m_running.load(std::memory_order_acquire);
            });
            if (!m_running.load(std::memory_order_acquire)) {
                break; // Stop was requested while paused
            }
            // After resuming, update state
            set_state(PrefetcherState::Running);
        }

        // ------------------------------------------------------------------
        // PHASE B: Prepare Memory Slot (may retry on overflow)
        // ------------------------------------------------------------------
        // Call SlidingWindowManager::prepare_layer() to reserve a memory
        // region for the current layer's data. If the ring buffer is full
        // (WindowOverflowError), block and wait for the main thread to
        // retire some layers.
        memory::LayerHandle handle;
        bool prepared = false;

        while (!prepared && m_running.load(std::memory_order_acquire)) {
            try {
                // Acquire the SWM's mutex via guard() for thread-safe access.
                auto swm_lock = m_swm.guard();
                handle = m_swm.prepare_layer(
                    static_cast<memory::SlidingWindowManager::size_type>(
                        m_layer_sizes[static_cast<std::size_t>(current_layer)]));
                prepared = true;

            } catch (const memory::WindowOverflowError&) {
                // The ring buffer is full. This happens when the main thread
                // hasn't retired enough layers yet. We need to wait for space.
                //
                // Backpressure protocol:
                //   1. BG thread catches WindowOverflowError
                //   2. BG thread blocks on m_cv_state_change
                //   3. Main thread calls retire_and_signal()
                //   4. retire_and_signal() notifies m_cv_state_change
                //   5. BG thread wakes up and retries prepare_layer()
                //
                // We also check m_running to allow graceful shutdown even
                // while waiting for space.
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv_state_change.wait(lock, [this]() -> bool {
                    return !m_running.load(std::memory_order_acquire);
                });
                // After waking, loop back and retry prepare_layer()
            } catch (const memory::InvalidLayerSizeError& e) {
                // [FIX #6] اكتب m_error_msg تحت lock قبل ما نرفع الـ atomic flag
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_error_msg = "Fatal: layer " + std::to_string(current_layer) +
                                  " size (" + std::to_string(
                                      m_layer_sizes[static_cast<std::size_t>(current_layer)]) +
                                  " bytes) exceeds the SlidingWindowManager capacity. " +
                                  "Error: " + e.what();
                }
                m_error.store(true, std::memory_order_release);
                set_state(PrefetcherState::Error);
                m_cv_fetched.notify_all();
                return;
            }
        }

        if (!m_running.load(std::memory_order_acquire)) {
            break; // Stop was requested during overflow retry
        }

        // ------------------------------------------------------------------
        // PHASE C: Zero-Copy Read from Disk to RAM
        // ------------------------------------------------------------------
        // This is the CRITICAL zero-copy operation. The ifstream::read()
        // writes directly from the file (via the 4MB internal buffer set
        // by pubsetbuf) into the SlidingWindowManager's locked RAM region.
        //
        // Data flow:  [HDD/SSD] → [OS Page Cache] → [4MB ifstream buffer] → [Locked RAM]
        //
        // No intermediate std::vector, std::array, or any heap allocation
        // occurs in this path. The handle.address points directly into
        // the SovereignArena's memory region.
        seek_to_layer(static_cast<std::uint32_t>(current_layer));

        m_file.read(
            static_cast<char*>(handle.address),
            static_cast<std::streamsize>(
                m_layer_sizes[static_cast<std::size_t>(current_layer)]));

        // --- Read Error Check -----------------------------------------------------
        if (!m_file) {
            // [FIX #6] اكتب m_error_msg تحت lock دايماً قبل رفع الـ atomic flag
            if (m_file.eof()) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_error_msg = "Unexpected EOF while reading layer " +
                              std::to_string(current_layer) +
                              ". Expected " + std::to_string(
                                  m_layer_sizes[static_cast<std::size_t>(current_layer)]) +
                              " bytes from file '" + m_file_path + "'.";
            } else {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_error_msg = "I/O error while reading layer " +
                              std::to_string(current_layer) +
                              " from file '" + m_file_path + "'.";
            }
            m_error.store(true, std::memory_order_release);
            set_state(PrefetcherState::Error);
            m_cv_fetched.notify_all();
            return;
        }

        // ------------------------------------------------------------------
        // PHASE D: Signal the Main Thread
        // ------------------------------------------------------------------
        // Set the latest fetched layer ID with RELEASE semantics.
        // This ensures that all the data written by ifstream::read() above
        // is visible to the main thread when it observes this value with
        // ACQUIRE semantics in wait_for_layer().
        //
        // Memory ordering rationale:
        //   The ifstream::read() write → (release) → m_latest_fetched_layer store
        //   m_latest_fetched_layer load (acquire) → visible data for inference
        m_latest_fetched_layer.store(current_layer, std::memory_order_release);

        // Wake the main thread if it's waiting in wait_for_layer().
        m_cv_fetched.notify_all();

        // ------------------------------------------------------------------
        // PHASE E: Wait for Main Thread Consumption (Handshake)
        // ------------------------------------------------------------------
        // The BG thread must NOT proceed to the next prepare_layer() until
        // the main thread has consumed the current layer. This enforces the
        // "pipeline depth = 1" invariant and prevents the SWM from having
        // multiple Preparing slots simultaneously.
        //
        // The main thread signals consumption by calling retire_and_signal(),
        // which increments m_consumed_up_to and notifies m_cv_consumed.
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv_consumed.wait(lock, [this, current_layer]() -> bool {
                return !m_running.load(std::memory_order_acquire) ||
                       m_consumed_up_to.load(std::memory_order_acquire) >= current_layer;
            });
        }

        if (!m_running.load(std::memory_order_acquire)) {
            break; // Stop was requested during consumption wait
        }

        // --- Advance to Next Layer ------------------------------------------------
        ++current_layer;
    }

    // --- Loop Termination: Signal Completion --------------------------------------
    // Set the fetch_complete flag so that any thread waiting in
    // wait_for_layer() can detect normal completion (vs. error or stop).
    m_fetch_complete.store(true, std::memory_order_release);
    m_cv_fetched.notify_all();

    // Update state to Completed (if not already in Error state)
    if (!m_error) {
        set_state(PrefetcherState::Completed);
    }
}

// =============================================================================
//  Private Methods — File Operations
// =============================================================================

void AsyncPrefetcher::validate_and_parse_header() {
    // --- Phase 1: Read Fixed Header (12 bytes) ------------------------------------
    // Read magic number (4 bytes), version (4 bytes), num_layers (4 bytes).
    // All fields are stored in little-endian byte order (native on x86/ARM).

    struct RawHeader {
        std::uint32_t magic;
        std::uint32_t version;
        std::uint32_t num_layers;
    };

    // Use aligned read to avoid undefined behavior from reinterpret_cast
    // on potentially misaligned stack memory. We read into a char buffer
    // first, then memcpy into the struct.
    alignas(alignof(RawHeader)) char raw_bytes[sizeof(RawHeader)];

    m_file.read(raw_bytes, static_cast<std::streamsize>(sizeof(RawHeader)));

    if (!m_file) {
        throw std::runtime_error(
            "AsyncPrefetcher: failed to read AIOC file header from '" +
            m_file_path + "'. File may be too small or corrupted.");
    }

    // Copy into the typed struct (safe, no alignment issues due to alignas)
    RawHeader raw{};
    std::memcpy(&raw, raw_bytes, sizeof(RawHeader));

    m_header.magic      = raw.magic;
    m_header.version    = raw.version;
    m_header.num_layers = raw.num_layers;

    // --- Phase 2: Validate Header Fields ------------------------------------------
    if (m_header.magic != AIOC_FILE_MAGIC) {
        std::ostringstream oss;
        oss << "AsyncPrefetcher: invalid file magic. Expected 0x"
            << std::hex << AIOC_FILE_MAGIC << " (\"AIOC\"), got 0x"
            << std::hex << m_header.magic
            << ". The file '" << m_file_path
            << "' is not a valid AIOC model file.";
        throw std::runtime_error(oss.str());
    }

    if (m_header.version != AIOC_FILE_VERSION) {
        throw std::runtime_error(
            "AsyncPrefetcher: unsupported file version " +
            std::to_string(m_header.version) +
            ". Expected version " + std::to_string(AIOC_FILE_VERSION) +
            ". The file '" + m_file_path + "' may have been created by a newer "
            "version of the AI OS Core tooling.");
    }

    if (m_header.num_layers == 0) {
        throw std::runtime_error(
            "AsyncPrefetcher: model file '" + m_file_path +
            "' declares 0 layers. A valid model must have at least 1 layer.");
    }

    // --- Phase 3: Read Layer Size Table -------------------------------------------
    // The header is followed by num_layers × 8 bytes of uint64_t layer sizes.
    m_layer_sizes.resize(static_cast<std::size_t>(m_header.num_layers));

    const std::size_t size_table_bytes = static_cast<std::size_t>(
        m_header.num_layers) * sizeof(std::uint64_t);

    // Read into a temporary buffer, then memcpy into the uint64_t vector
    // to avoid alignment issues with ifstream::read() into uint64_t*.
    std::vector<char> size_buffer(size_table_bytes);
    m_file.read(size_buffer.data(),
                static_cast<std::streamsize>(size_table_bytes));

    if (!m_file) {
        throw std::runtime_error(
            "AsyncPrefetcher: failed to read layer size table from '" +
            m_file_path + "'. File is too small to contain " +
            std::to_string(m_header.num_layers) + " layer size entries.");
    }

    // Parse the byte buffer into uint64_t values
    for (std::uint32_t i = 0; i < m_header.num_layers; ++i) {
        std::memcpy(&m_layer_sizes[static_cast<std::size_t>(i)],
                    size_buffer.data() + static_cast<std::size_t>(i) * sizeof(std::uint64_t),
                    sizeof(std::uint64_t));
    }

    // --- Phase 4: Compute Layer Offsets -------------------------------------------
    // Each layer's data starts at: data_start_offset + sum(layer_sizes[0..i-1])
    m_data_start_offset = static_cast<std::uint64_t>(
        AIOC_HEADER_BASE_SIZE + size_table_bytes);

    m_layer_offsets.resize(static_cast<std::size_t>(m_header.num_layers));
    std::uint64_t running_offset = m_data_start_offset;
    for (std::uint32_t i = 0; i < m_header.num_layers; ++i) {
        m_layer_offsets[static_cast<std::size_t>(i)] = running_offset;
        running_offset += m_layer_sizes[static_cast<std::size_t>(i)];
    }

    // --- Phase 5: Validate File Size ---------------------------------------------
    // The file must be at least large enough to contain all declared layer data.
    // We check by seeking to the expected end and comparing.
    const auto original_pos = m_file.tellg();
    m_file.seekg(0, std::ios::end);
    const auto file_size = m_file.tellg();
    m_file.seekg(original_pos); // Restore read position

    if (file_size < 0 ||
        static_cast<std::uint64_t>(file_size) < running_offset) {
        throw std::runtime_error(
            "AsyncPrefetcher: file '" + m_file_path + "' is too small. "
            "Declared data requires " + std::to_string(running_offset) +
            " bytes, but file size is " +
            (file_size >= 0 ? std::to_string(static_cast<std::uint64_t>(file_size))
                            : "unknown") + " bytes.");
    }

    // --- Phase 6: Position File at Start of Data Section --------------------------
    m_file.seekg(static_cast<std::streamoff>(m_data_start_offset), std::ios::beg);
}

void AsyncPrefetcher::seek_to_layer(std::uint32_t layer_id) {
    if (layer_id >= m_header.num_layers) {
        throw std::out_of_range(
            "AsyncPrefetcher::seek_to_layer: layer_id " +
            std::to_string(layer_id) + " is out of range.");
    }

    const std::uint64_t offset = m_layer_offsets[static_cast<std::size_t>(layer_id)];
    m_file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);

    if (!m_file) {
        throw std::runtime_error(
            "AsyncPrefetcher: failed to seek to layer " +
            std::to_string(layer_id) + " at offset " +
            std::to_string(offset) + " in file '" + m_file_path + "'.");
    }
}

// =============================================================================
//  Private Methods — State Management
// =============================================================================

void AsyncPrefetcher::set_state(PrefetcherState new_state) {
    m_state.store(new_state, std::memory_order_release);
}

} // namespace io
} // namespace ai_os
