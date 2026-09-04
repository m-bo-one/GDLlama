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

// Opens every backend beside this library, once for the process: the ones that ship with the
// addon and any that were dropped in later, the CUDA one included. That folder is put on the
// DLL search path for the length of the call, because a backend there imports its runtime from
// beside itself and Windows does not look in a loaded library's own folder. False when the
// folder holds no backend at all, which is the one failure a caller cannot go on from.
bool ensure_backends();

// Frees what the backends hold. Called when the library is unloaded.
void shutdown_backends();

// Whether llama.cpp's own log lines below warning level reach the console.
void set_verbose(bool on);
bool is_verbose();

// The filter llama.cpp's log lines go through, so that a library with a logger of its own --
// mtmd keeps a second one -- can be pointed at the same one. Without it a game's console
// carries a line per encoded chunk.
ggml_log_callback log_callback();

// The device a class put its model on, sampled for its free and total memory as it is noted.
// Called beside note_operation() at load and again as a turn begins, so the pair the crash
// line prints belongs to whoever was working. Sampled here rather than read from a handler:
// asking CUDA for its memory goes through the same check that is killing the process.
void note_device(ggml_backend_dev_t device);

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

// What this library was last doing, as a literal that outlives the call: "LlamaSpeech: making
// a sentence". Written before an operation that could take the process down and read by the
// handlers below, which run when nothing else can be asked. A breadcrumb rather than a log
// line, because one line per sentence in a game's console is not worth what it buys.
//
// It is the whole answer for a Windows fail-fast, which no handler ever sees: what survives
// such a death is the last thing the process wrote down before it.
void note_operation(const char *what);
const char *last_operation();

// One line naming this library, what it was doing and why it is dying, out through Godot's
// error printing so it reaches the log file, and through stderr first so it lands even when
// the engine can no longer be called. Safe from a handler: it allocates nothing.
void report_fatal(const char *reason, const char *detail);

// Installed once at library initialisation: a terminate handler, a Windows unhandled-exception
// filter, a CRT invalid-parameter handler -- which is what turns one class of fail-fast back
// into a report -- an abort signal handler, and ggml's own abort callback, so a GGML_ASSERT
// names itself in the log before the process goes. Without these a fault inside a backend is a
// process that disappears with nothing written anywhere.
void install_crash_reporting();

} // namespace llama_runtime

#endif // LLAMA_RUNTIME_H
