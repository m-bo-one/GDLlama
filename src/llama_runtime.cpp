#include "llama_runtime.h"

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "llama.h"
#include "log.h"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace godot;

namespace {

std::atomic<bool> verbose_logs{false};
std::atomic<bool> backends_ready{false};
std::mutex backends_lock;

// The breadcrumb. A literal with static storage, so reading it from a handler on another
// thread while a worker replaces it is a torn read of nothing: either pointer is valid text.
std::atomic<const char *> current_operation{"nothing yet"};

// Raised by the first handler to fire. Two of the paths below can run for one death -- ggml's
// abort callback and then the abort signal -- and one line is the point of this.
std::atomic<bool> already_reported{false};

std::atomic<bool> handlers_installed{false};

// The last lines llama.cpp and ggml wrote at error or warning level. ggml prints the cause of
// a fault -- "CUDA error: out of memory", the device, the statement -- through its log and then
// aborts with a message naming only a file and a line, so without this ring the crash line
// carries the abort and not the reason. Fixed slots and no allocation: it is read while the
// process is dying.
//
// A slot's last byte is only ever written as the terminator snprintf puts there, so a reader
// racing a writer sees a mix of two messages and never runs off the end.
constexpr int KEPT_LINES = 12;
constexpr int KEPT_LENGTH = 256;
char kept_lines[KEPT_LINES][KEPT_LENGTH];
std::atomic<uint64_t> kept_written{0};

// The device whichever class is working put its model on, with what its memory read when it
// was noted. Sampled at a safe moment rather than from a handler, and stored as plain numbers.
std::atomic<ggml_backend_dev_t> current_device{nullptr};
std::atomic<uint64_t> device_free_mb{0};
std::atomic<uint64_t> device_total_mb{0};
char device_name[KEPT_LENGTH] = "";

// One log line into the ring, its trailing newline and padding dropped so the crash report
// reads as sentences rather than as a transcript. Whitespace alone is not a line.
void keep_line(const char *text) {
    if (text == nullptr) {
        return;
    }
    size_t begin = 0;
    while (text[begin] == ' ' || text[begin] == '\t' || text[begin] == '\n' || text[begin] == '\r') {
        begin++;
    }
    size_t end = begin;
    while (text[end] != '\0') {
        end++;
    }
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t' || text[end - 1] == '\n' ||
                                  text[end - 1] == '\r')) {
        end--;
    }
    if (end == begin) {
        return;
    }
    const uint64_t slot = kept_written.fetch_add(1) % KEPT_LINES;
    size_t length = end - begin;
    if (length > (size_t)KEPT_LENGTH - 1) {
        length = (size_t)KEPT_LENGTH - 1;
    }
    memcpy(kept_lines[slot], text + begin, length);
    kept_lines[slot][length] = '\0';
}

// llama.cpp's lines below warning level are dropped unless asked for. A continuation line
// follows the level of the line it continues, or a dropped message leaks its tail.
void quiet_log(ggml_log_level level, const char *text, void *) {
    static std::atomic<int> last_level{GGML_LOG_LEVEL_INFO};
    if (level != GGML_LOG_LEVEL_CONT) {
        last_level.store((int)level);
    }
    const int shown = last_level.load();
    if (shown == GGML_LOG_LEVEL_ERROR || shown == GGML_LOG_LEVEL_WARN) {
        keep_line(text);
    }
    if (shown == GGML_LOG_LEVEL_ERROR || shown == GGML_LOG_LEVEL_WARN || verbose_logs.load()) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

} // namespace

namespace llama_runtime {

void note_operation(const char *what) {
    current_operation.store(what == nullptr ? "nothing yet" : what);
}

const char *last_operation() {
    return current_operation.load();
}

void note_device(ggml_backend_dev_t device) {
    current_device.store(device);
    if (device == nullptr) {
        device_free_mb.store(0);
        device_total_mb.store(0);
        device_name[0] = '\0';
        return;
    }
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    device_free_mb.store((uint64_t)(free / (1024 * 1024)));
    device_total_mb.store((uint64_t)(total / (1024 * 1024)));
    snprintf(device_name, sizeof(device_name), "%s (%s)", ggml_backend_dev_name(device),
            ggml_backend_dev_description(device));
}

// Composed into a fixed buffer rather than a std::string: this runs while the process is
// dying, and a heap that has just been corrupted is the likeliest reason it is. The kept log
// lines go out with it, because the abort message names a file and a line and the reason is in
// what the library wrote just before.
void report_fatal(const char *reason, const char *detail) {
    if (already_reported.exchange(true)) {
        return;
    }
    char line[4096];
    int at = snprintf(line, sizeof(line),
            "LlamaChat extension: %s while %s.%s%s The process is going down; this line is the "
            "last thing it knows.",
            reason == nullptr ? "a fault" : reason, last_operation(),
            detail == nullptr || detail[0] == '\0' ? "" : " ",
            detail == nullptr ? "" : detail);
    if (at < 0) {
        at = 0;
    }
    // The card the working class chose and what was free on it when the operation began. Not
    // asked for again here: the query goes through the backend that is dying.
    if (device_name[0] != '\0' && at < (int)sizeof(line)) {
        at += snprintf(line + at, sizeof(line) - (size_t)at,
                "\nThe model was on %s, %llu MiB free of %llu MiB when this began.", device_name,
                (unsigned long long)device_free_mb.load(), (unsigned long long)device_total_mb.load());
    }
    const uint64_t written = kept_written.load();
    if (written > 0 && at > 0 && at < (int)sizeof(line)) {
        at += snprintf(line + at, sizeof(line) - (size_t)at, "\nWhat the library said last:");
        const uint64_t first = written > (uint64_t)KEPT_LINES ? written - (uint64_t)KEPT_LINES : 0;
        for (uint64_t i = first; i < written && at > 0 && at < (int)sizeof(line); i++) {
            at += snprintf(line + at, sizeof(line) - (size_t)at, "\n  %s",
                    kept_lines[i % (uint64_t)KEPT_LINES]);
        }
    }
    // stderr first and flushed: the engine may already be unable to take a call, and this is
    // the copy that reaches a console either way.
    fputs(line, stderr);
    fputc('\n', stderr);
    fflush(stderr);
    UtilityFunctions::push_error(String::utf8(line));
}

// ggml calls this with its own message and then aborts, so a GGML_ASSERT inside a backend --
// which is how the speech codec dies on Vulkan -- names its file, its line and its condition
// in the Godot log rather than taking the process with nothing written.
static void on_ggml_abort(const char *message) {
    report_fatal("ggml stopped", message);
}

static void on_terminate() {
    const char *detail = "";
    if (std::current_exception() != nullptr) {
        detail = "An exception left a thread with nobody to catch it.";
    }
    report_fatal("a thread ended in terminate", detail);
    std::abort();
}

static void on_abort_signal(int) {
    report_fatal("the process was aborted", "");
    std::signal(SIGABRT, SIG_DFL);
    std::raise(SIGABRT);
}

#ifdef _WIN32
static LONG WINAPI on_unhandled_exception(EXCEPTION_POINTERS *info) {
    char detail[128] = "";
    if (info != nullptr && info->ExceptionRecord != nullptr) {
        snprintf(detail, sizeof(detail), "Windows reports code 0x%08lX at 0x%p.",
                (unsigned long)info->ExceptionRecord->ExceptionCode,
                info->ExceptionRecord->ExceptionAddress);
    }
    report_fatal("the process faulted", detail);
    return EXCEPTION_CONTINUE_SEARCH;
}

// The C runtime calls this instead of failing fast when a parameter it was handed cannot be
// used. Installing it is what turns that class of 0xC0000409 -- a death no filter above ever
// sees -- back into a line somebody can read.
static void on_invalid_parameter(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int, uintptr_t) {
    report_fatal("the runtime was handed a parameter it could not use", "");
    std::abort();
}
#endif

// Installed once, from the library's initialisation, and never taken down: a handler that is
// removed while a worker is still running is a fault nobody reports.
void install_crash_reporting() {
    if (handlers_installed.exchange(true)) {
        return;
    }
    std::set_terminate(on_terminate);
    std::signal(SIGABRT, on_abort_signal);
    ggml_set_abort_callback(on_ggml_abort);
#ifdef _WIN32
    SetUnhandledExceptionFilter(on_unhandled_exception);
    _set_invalid_parameter_handler(on_invalid_parameter);
#endif
}

std::string to_std(const String &text) {
    const CharString bytes = text.utf8();
    return std::string(bytes.get_data(), (size_t)bytes.length());
}

String to_gd(const std::string &text) {
    return String::utf8(text.c_str(), (int)text.size());
}

std::string own_directory() {
#ifdef _WIN32
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCWSTR)&own_directory, &module);
    wchar_t buffer[MAX_PATH];
    const DWORD length = GetModuleFileNameW(module, buffer, MAX_PATH);
    std::wstring wide(buffer, length);
    const size_t slash = wide.find_last_of(L"\\/");
    if (slash != std::wstring::npos) {
        wide.resize(slash);
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int)wide.size(), nullptr, 0, nullptr, nullptr);
    std::string out((size_t)size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int)wide.size(), &out[0], size, nullptr, nullptr);
    return out;
#else
    Dl_info info;
    if (dladdr((void *)&own_directory, &info) && info.dli_fname != nullptr) {
        std::string path = info.dli_fname;
        const size_t slash = path.find_last_of('/');
        return slash == std::string::npos ? "." : path.substr(0, slash);
    }
    return ".";
#endif
}

bool ensure_backends() {
    std::lock_guard<std::mutex> hold(backends_lock);
    if (backends_ready.load()) {
        return true;
    }
    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(verbose_logs.load() ? LOG_LEVEL_INFO : LOG_LEVEL_WARN);
    llama_backend_init();
    const std::string folder = own_directory();
#ifdef _WIN32
    // This folder goes on the library search path for the length of the call: a backend here
    // imports its runtime from beside itself -- the CUDA one names three NVIDIA libraries --
    // and Windows does not look in a loaded library's own directory for what it imports.
    const int wide_size = MultiByteToWideChar(CP_UTF8, 0, folder.c_str(), (int)folder.size(), nullptr, 0);
    std::wstring wide((size_t)wide_size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, folder.c_str(), (int)folder.size(), &wide[0], wide_size);
    SetDllDirectoryW(wide.c_str());
    ggml_backend_load_all_from_path(folder.c_str());
    SetDllDirectoryW(nullptr);
#else
    ggml_backend_load_all_from_path(folder.c_str());
#endif
    if (ggml_backend_dev_count() == 0) {
        UtilityFunctions::push_error(
                "LlamaRuntime: no ggml backend library was found beside the extension in \"" + to_gd(folder) + "\".");
        return false;
    }
    backends_ready.store(true);
    return true;
}

void shutdown_backends() {
    std::lock_guard<std::mutex> hold(backends_lock);
    if (backends_ready.exchange(false)) {
        llama_backend_free();
    }
}

void set_verbose(bool on) {
    verbose_logs.store(on);
    common_log_set_verbosity_thold(on ? LOG_LEVEL_INFO : LOG_LEVEL_WARN);
}

bool is_verbose() {
    return verbose_logs.load();
}

ggml_log_callback log_callback() {
    return quiet_log;
}

std::string backend_name_of(ggml_backend_dev_t device) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    const char *name = reg == nullptr ? nullptr : ggml_backend_reg_name(reg);
    return name == nullptr ? std::string() : std::string(name);
}

const char *device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "accel";
        default:
            return "other";
    }
}

ggml_backend_dev_t best_device() {
    ggml_backend_dev_t best = nullptr;
    size_t best_total = 0;
    bool best_discrete = false;
    bool best_cuda = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const bool discrete = type == GGML_BACKEND_DEVICE_TYPE_GPU;
        const bool cuda = backend_name_of(device) == "CUDA";
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        // CUDA first, then a discrete GPU over an integrated one, then the larger memory: the
        // model is placed whole on one device rather than split across two.
        const bool better = best == nullptr || (cuda && !best_cuda) ||
                (cuda == best_cuda && discrete && !best_discrete) ||
                (cuda == best_cuda && discrete == best_discrete && total > best_total);
        if (better) {
            best = device;
            best_total = total;
            best_discrete = discrete;
            best_cuda = cuda;
        }
    }
    return best;
}

std::string describe_device(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return "cpu";
    }
    return std::string(ggml_backend_dev_name(device)) + " (" + ggml_backend_dev_description(device) + ")";
}

Array describe_devices() {
    Array out;
    if (!ensure_backends()) {
        return out;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        Dictionary entry;
        entry["name"] = String::utf8(ggml_backend_dev_name(device));
        entry["description"] = String::utf8(ggml_backend_dev_description(device));
        entry["backend"] = to_gd(backend_name_of(device));
        entry["type"] = device_type_name(ggml_backend_dev_type(device));
        entry["memory_total_mb"] = (int64_t)(total / (1024 * 1024));
        entry["memory_free_mb"] = (int64_t)(free / (1024 * 1024));
        out.push_back(entry);
    }
    return out;
}

} // namespace llama_runtime
