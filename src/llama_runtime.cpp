#include "llama_runtime.h"

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "llama.h"
// The staging header of llama.cpp, which is where the buffer accounting lives. It is under the
// library's src/ rather than its include/, so the build puts that folder on this target's path.
#include "llama-ext.h"
#include "log.h"

#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <set>
#include <string>


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

// Whether what stands at this position is the mark a virtual device carries -- "-v" and a number,
// which the CUDA backend appends to keep several devices over one card apart.
bool is_a_virtual_tail(const std::string &address, size_t at) {
    if (at + 2 >= address.size()) {
        return false;
    }
    for (size_t i = at + 2; i < address.size(); i++) {
        if (isdigit((unsigned char)address[i]) == 0) {
            return false;
        }
    }
    return true;
}

// One PCI address, in the one spelling this file compares. Three are in the wild and all three
// name the same card: the four-digit domain both backends write, the eight-digit one nvidia-smi
// prints, and a "-v<i>" tail on a virtual device. Lower case, trimmed, domain cut to four digits.
std::string canonical_address(const std::string &written) {
    std::string address;
    for (char one : written) {
        address += (char)tolower((unsigned char)one);
    }
    const size_t first = address.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return std::string();
    }
    address = address.substr(first, address.find_last_not_of(" \t\r\n") + 1 - first);
    const size_t tail = address.rfind("-v");
    if (tail != std::string::npos && is_a_virtual_tail(address, tail)) {
        address.erase(tail);
    }
    const size_t colon = address.find(':');
    if (colon != std::string::npos && colon > 4) {
        address.erase(0, colon - 4);
    }
    return address;
}

// The device whose PCI address is the one asked for, or null where none carries it. It is the
// only identity two libraries in one process can both produce: a name is shared by two cards of a
// model, and an index is a position each of them walks for itself.
//
// One card can be two devices here -- a CUDA one and a Vulkan one over the same hardware, both
// reporting the same address -- so the address chooses the card and this backend keeps its own
// preference within it: the first CUDA device at that address, else the first of any. That tie
// alone; the heap and discrete ranking best_device() applies has nothing to break here.
ggml_backend_dev_t device_at(const std::string &address) {
    const std::string wanted = canonical_address(address);
    if (wanted.empty()) {
        return nullptr;
    }
    ggml_backend_dev_t found = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(device, &props);
        if (props.device_id == nullptr || wanted != canonical_address(props.device_id)) {
            continue;
        }
        if (found == nullptr) {
            found = device;
            continue;
        }
        if (backend_name_of(found) != "CUDA" && backend_name_of(device) == "CUDA") {
            found = device;
        }
    }
    return found;
}

// Whether nothing has been said about this address yet, marking it as it answers. Keyed by the
// address rather than latched once for the process: two models told two different cards that are
// not there is two problems, and the second one silent reads as a load that got what it asked for.
bool is_the_first_word_about(const std::string &address) {
    static std::mutex said_lock;
    static std::set<std::string> said_about;
    std::lock_guard<std::mutex> held(said_lock);
    return said_about.insert(address).second;
}

ggml_backend_dev_t best_device(const std::string &wanted) {
    ggml_backend_dev_t named = device_at(wanted);
    if (named != nullptr) {
        return named;
    }
    const std::string address = canonical_address(wanted);
    if (!address.empty() && is_the_first_word_about(address)) {
        // Said once and not obeyed: the caller named a card this backend cannot see -- another
        // library's ranking, a driver reporting no PCI address -- and one ranked here is better
        // than none. A host reading this knows the two libraries are on different cards.
        UtilityFunctions::push_warning(
                "LlamaRuntime: no device at \"" + to_gd(wanted) + "\" (compared as \""
                + to_gd(address) + "\"), so one was ranked here instead. A caller that named it "
                "to another library is then on two cards.");
    }
    return best_device();
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

// The PCI address a device reports, in the one spelling this file compares. It is what a host
// holds against another library's answer to know the two are on one card, so it is canonical
// here: a virtual device's "-v<i>" tail would otherwise read as a card the other library lacks.
std::string address_of(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return std::string();
    }
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);
    return props.device_id == nullptr ? std::string() : canonical_address(props.device_id);
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

Dictionary memory_report_of(const llama_context *ctx) {
    int64_t weights = 0;
    int64_t kv = 0;
    int64_t compute = 0;
    int64_t host = 0;
    if (ctx != nullptr) {
        // One entry per buffer type the context allocated anything in: the device's own, and
        // the pinned host buffer beside it where the backend keeps one.
        for (const auto &entry : llama_get_memory_breakdown(ctx)) {
            const llama_memory_breakdown_data &held = entry.second;
            weights += (int64_t)held.model;
            kv += (int64_t)held.context;
            compute += (int64_t)held.compute;
            // Asked of the buffer type rather than of its device: a backend's pinned host
            // buffer belongs to the device and is ordinary memory, and counting it as the
            // card's would put half a gigabyte on a card that never held it.
            if (ggml_backend_buft_is_host(entry.first)) {
                host += (int64_t)held.total();
            }
        }
    }
    Dictionary out;
    out["weights_bytes"] = weights;
    out["kv_bytes"] = kv;
    out["compute_bytes"] = compute;
    out["host_bytes"] = host;
    return out;
}

Dictionary device_memory_of(ggml_backend_dev_t device) {
    Dictionary out;
    out["free_bytes"] = (int64_t)-1;
    out["total_bytes"] = (int64_t)-1;
    out["name"] = String();
    if (device == nullptr) {
        return out;
    }
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    out["name"] = to_gd(describe_device(device));
    if (total == 0) {
        return out;
    }
    out["free_bytes"] = (int64_t)free;
    out["total_bytes"] = (int64_t)total;
    return out;
}

} // namespace llama_runtime
