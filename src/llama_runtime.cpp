#include "llama_runtime.h"

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "llama.h"
#include "log.h"

#include <atomic>
#include <mutex>
#include <set>

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
std::set<std::string> opened_folders;

// llama.cpp's lines below warning level are dropped unless asked for. A continuation line
// follows the level of the line it continues, or a dropped message leaks its tail.
void quiet_log(ggml_log_level level, const char *text, void *) {
    static std::atomic<int> last_level{GGML_LOG_LEVEL_INFO};
    if (level != GGML_LOG_LEVEL_CONT) {
        last_level.store((int)level);
    }
    const int shown = last_level.load();
    if (shown == GGML_LOG_LEVEL_ERROR || shown == GGML_LOG_LEVEL_WARN || verbose_logs.load()) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

} // namespace

namespace llama_runtime {

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
    ggml_backend_load_all_from_path(folder.c_str());
    opened_folders.insert(folder);
    if (ggml_backend_dev_count() == 0) {
        UtilityFunctions::push_error(
                "LlamaRuntime: no ggml backend library was found beside the extension in \"" + to_gd(folder) + "\".");
        return false;
    }
    backends_ready.store(true);
    return true;
}

bool load_backend_folder(const std::string &folder) {
    if (folder.empty()) {
        return false;
    }
    if (!ensure_backends()) {
        return false;
    }
    std::lock_guard<std::mutex> hold(backends_lock);
    if (opened_folders.count(folder) != 0) {
        return false;
    }
    opened_folders.insert(folder);
    const size_t before = ggml_backend_dev_count();
#ifdef _WIN32
    const int wide_size = MultiByteToWideChar(CP_UTF8, 0, folder.c_str(), (int)folder.size(), nullptr, 0);
    std::wstring wide((size_t)wide_size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, folder.c_str(), (int)folder.size(), &wide[0], wide_size);
    SetDllDirectoryW(wide.c_str());
    ggml_backend_load_all_from_path(folder.c_str());
    SetDllDirectoryW(nullptr);
#else
    ggml_backend_load_all_from_path(folder.c_str());
#endif
    return ggml_backend_dev_count() > before;
}

void shutdown_backends() {
    std::lock_guard<std::mutex> hold(backends_lock);
    if (backends_ready.exchange(false)) {
        opened_folders.clear();
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
