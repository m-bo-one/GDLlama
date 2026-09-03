#ifndef LLAMA_RUNTIME_H
#define LLAMA_RUNTIME_H

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>

#include "ggml-backend.h"

#include <string>

// What every class in this library shares: the ggml backend registry, which is process-wide
// and must be opened exactly once per folder, and the two string conversions. Two classes with
// a registry each would load the same folder twice and register every device twice over.
namespace llama_runtime {

std::string to_std(const godot::String &text);
godot::String to_gd(const std::string &text);

// The folder this library was loaded from, which is where ggml looks for its backends. Asked
// of the OS rather than of the engine, so an exported game and the editor answer the same.
std::string own_directory();

// Opens the backends beside this library, once for the process. False when the folder holds
// none, which is the one failure a caller cannot go on from.
bool ensure_backends();

// Opens the backends in one more folder, once per folder, for a backend too large to ship in
// the addon -- the CUDA one, fetched beside the models. The folder is put on the DLL search
// path for the length of the call, because a backend there imports its runtime from beside
// itself and Windows does not look in a loaded library's own folder. Answers whether anything
// was opened; a folder that is not there is not a failure.
bool load_backend_folder(const std::string &folder);

// Frees what the backends hold. Called when the library is unloaded.
void shutdown_backends();

// Whether llama.cpp's own log lines below warning level reach the console.
void set_verbose(bool on);
bool is_verbose();

// The filter llama.cpp's log lines go through, so that a library with a logger of its own --
// mtmd keeps a second one -- can be pointed at the same one. Without it a game's console
// carries a line per encoded chunk.
ggml_log_callback log_callback();

// The best device for a whole model, or null when the machine has none: CUDA over any other
// backend, then a discrete GPU over an integrated one, then the largest memory. CUDA is
// preferred because it is the only backend that runs the speech codec's graph.
ggml_backend_dev_t best_device();

// One device's backend name ("CUDA", "Vulkan"), and the word its type is reported by.
std::string backend_name_of(ggml_backend_dev_t device);
const char *device_type_name(enum ggml_backend_dev_type type);

// How a chosen device is written into a timings dictionary: the ggml name and its description.
std::string describe_device(ggml_backend_dev_t device);

// Every device the backends found, one dictionary each.
godot::Array describe_devices();

} // namespace llama_runtime

#endif // LLAMA_RUNTIME_H
