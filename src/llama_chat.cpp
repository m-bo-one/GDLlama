#include "llama_chat.h"

#include "llama_runtime.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "ggml-backend.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <set>

// The fences below are a try around a turn and a catch around a thread that could not
// start; compiled without exceptions they are dead code and either fault unwinds into the
// engine. Refused here rather than discovered on a machine that could not start the thread.
#if !defined(_CPPUNWIND) && !defined(__EXCEPTIONS) && !defined(__cpp_exceptions)
#error "llama_chat.cpp needs C++ exceptions: configure with GODOTCPP_DISABLE_EXCEPTIONS=OFF"
#endif

using namespace godot;

namespace {

// The slot a caller that names none answers in, which is the one a context opened for a single
// conversation has. Every other slot is asked for by index.
constexpr int FIRST_SLOT = 0;

// The index sentence, written once for the two places that refuse one. The number is how many
// sequences this context has, which is the range an index is accepted from and not how many
// seats a host counts: a caller passing a bad index needs the range that is actually taken.
String not_one_of_the_slots(int64_t slot, int count) {
    return vformat("LlamaChat: slot %d is not one of the %d this context was opened with; "
            "they are numbered from zero.", slot, count);
}

// What a refused decode says, by what the library means by each answer rather than by one
// sentence for all of them. Only `1` is the host's to act on, so only `1` advises a slot; `2`
// says what was thrown away, and the two negatives say the fault is not the conversation's.
String the_decode_refused(int answered, const char *what, int tokens, const String &held) {
    if (answered == 1) {
        return vformat("LlamaChat: %s could not be decoded; the context of %d tokens is full. "
                "The slots hold %s. Drop a finished conversation's slot with drop_slot(), or "
                "raise the context.", String(what), tokens, held);
    }
    if (answered == 2) {
        return vformat("LlamaChat: %s could not be decoded; the decode was aborted while the "
                "batch was being processed, so what this slot held has been cleared.",
                String(what));
    }
    if (answered == -1) {
        return vformat("LlamaChat: %s could not be decoded; the library refused the batch "
                "itself, which is this class's own fault and not the conversation's. Its own "
                "line is in the editor log.", String(what));
    }
    return vformat("LlamaChat: %s could not be decoded; llama_decode answered %d, and the "
            "library's own line is in the editor log.", String(what), answered);
}

using clock_type = std::chrono::steady_clock;

double ms_between(clock_type::time_point from, clock_type::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// Takes the busy flag in one atomic step or reports that another path holds it, and gives it
// back unless the turn was handed on. Reading the flag and raising it separately lets two
// callers both start a worker, and assigning a thread over a joinable one is std::terminate().
struct BusyGuard {
    std::atomic<bool> *held = nullptr;

    explicit BusyGuard(std::atomic<bool> &flag) {
        bool expected = false;
        if (flag.compare_exchange_strong(expected, true)) {
            held = &flag;
        }
    }

    ~BusyGuard() {
        if (held != nullptr) {
            held->store(false);
        }
    }

    BusyGuard(const BusyGuard &) = delete;
    BusyGuard &operator=(const BusyGuard &) = delete;

    bool taken() const { return held != nullptr; }

    // The flag stays raised and this stops owning it: the delivery on the main thread is
    // what lowers it, which is what keeps the model taken until the finish has gone out.
    void hand_on() { held = nullptr; }
};

using llama_runtime::to_gd;
using llama_runtime::to_std;

// A value as JSON text: a string is taken as already being one, anything else is encoded.
std::string json_text(const Variant &value) {
    if (value.get_type() == Variant::STRING) {
        return to_std(value);
    }
    if (value.get_type() == Variant::NIL) {
        return std::string();
    }
    return to_std(JSON::stringify(value));
}

// The text of a message's content, whether a string or the list of typed parts the OpenAI
// shape allows; only text parts are read.
std::string content_text(const Variant &content) {
    if (content.get_type() == Variant::STRING) {
        return to_std(content);
    }
    if (content.get_type() == Variant::ARRAY) {
        std::string out;
        const Array parts = content;
        for (int i = 0; i < parts.size(); i++) {
            if (parts[i].get_type() != Variant::DICTIONARY) {
                continue;
            }
            const Dictionary part = parts[i];
            if (String(part.get("type", "text")) == "text") {
                out += to_std(part.get("text", ""));
            }
        }
        return out;
    }
    return json_text(content);
}

bool read_messages(const Array &messages, std::vector<common_chat_msg> &out, String &error) {
    for (int i = 0; i < messages.size(); i++) {
        if (messages[i].get_type() != Variant::DICTIONARY) {
            error = vformat("Message %d is not a Dictionary.", i);
            return false;
        }
        const Dictionary message = messages[i];
        common_chat_msg msg;
        msg.role = to_std(message.get("role", "user"));
        if (msg.role != "system" && msg.role != "user" && msg.role != "assistant" && msg.role != "tool") {
            error = vformat("Message %d has the role \"%s\"; system, user, assistant and tool are the roles.",
                    i, to_gd(msg.role));
            return false;
        }
        msg.content = content_text(message.get("content", ""));
        if (message.has("tool_call_id")) {
            msg.tool_call_id = to_std(message["tool_call_id"]);
        }
        if (message.has("name")) {
            msg.tool_name = to_std(message["name"]);
        }
        if (message.has("reasoning_content")) {
            msg.reasoning_content = to_std(message["reasoning_content"]);
        }
        const Variant calls = message.get("tool_calls", Variant());
        if (calls.get_type() == Variant::ARRAY) {
            const Array list = calls;
            for (int j = 0; j < list.size(); j++) {
                if (list[j].get_type() != Variant::DICTIONARY) {
                    continue;
                }
                const Dictionary call = list[j];
                const Dictionary function = call.get("function", Dictionary());
                common_chat_tool_call made;
                made.id = to_std(call.get("id", ""));
                made.name = to_std(function.get("name", ""));
                made.arguments = json_text(function.get("arguments", "{}"));
                if (made.arguments.empty()) {
                    made.arguments = "{}";
                }
                msg.tool_calls.push_back(made);
            }
        }
        out.push_back(msg);
    }
    return true;
}

// A declaration in the OpenAI function shape, or the bare function object itself.
bool read_tools(const Array &tools, std::vector<common_chat_tool> &out, String &error) {
    for (int i = 0; i < tools.size(); i++) {
        if (tools[i].get_type() != Variant::DICTIONARY) {
            error = vformat("Tool %d is not a Dictionary.", i);
            return false;
        }
        Dictionary declaration = tools[i];
        if (declaration.has("function") && declaration["function"].get_type() == Variant::DICTIONARY) {
            declaration = declaration["function"];
        }
        common_chat_tool tool;
        tool.name = to_std(declaration.get("name", ""));
        if (tool.name.empty()) {
            error = vformat("Tool %d has no name.", i);
            return false;
        }
        tool.description = to_std(declaration.get("description", ""));
        Variant parameters = declaration.get("parameters", Variant());
        if (parameters.get_type() == Variant::NIL) {
            parameters = declaration.get("schema", Dictionary());
        }
        tool.parameters = json_text(parameters);
        if (tool.parameters.empty()) {
            tool.parameters = "{\"type\":\"object\",\"properties\":{}}";
        }
        out.push_back(tool);
    }
    return true;
}

// The cache types llama.cpp's own command line takes, by the names ggml gives them. Anything
// else is refused: a type the kernels have no path for opens a context that then fails to run.
bool cache_type_named(const String &word, ggml_type &named) {
    static const ggml_type allowed[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0,
        GGML_TYPE_IQ4_NL };
    for (const ggml_type type : allowed) {
        if (word == String(ggml_type_name(type))) {
            named = type;
            return true;
        }
    }
    return false;
}

// Where the bytes from `from` stop being whole UTF-8 sequences: the size, or the start of a
// sequence whose tail has not arrived. A piece cut inside a letter reads as a broken glyph.
size_t utf8_complete_end(const std::string &text, size_t from) {
    const size_t size = text.size();
    for (size_t back = 1; back <= 3 && size >= from + back; back++) {
        const size_t at = size - back;
        const unsigned char byte = (unsigned char)text[at];
        if ((byte & 0xC0) == 0x80) {
            continue;
        }
        size_t need = 1;
        if ((byte & 0xE0) == 0xC0) {
            need = 2;
        } else if ((byte & 0xF0) == 0xE0) {
            need = 3;
        } else if ((byte & 0xF8) == 0xF0) {
            need = 4;
        }
        return back >= need ? size : at;
    }
    return size;
}

// The one model file of a folder, so that a model is named by its folder the way a
// recogniser is, with its licence beside it. A folder holding none or several is refused
// with the sentence that says so, rather than the first one found being loaded in silence.
String single_gguf_in(const String &folder) {
    const PackedStringArray files = DirAccess::get_files_at(folder);
    PackedStringArray found;
    for (int i = 0; i < files.size(); i++) {
        if (files[i].to_lower().ends_with(".gguf")) {
            found.append(files[i]);
        }
    }
    if (found.size() != 1) {
        UtilityFunctions::push_error(vformat(
                "LlamaChat: the folder \"%s\" holds %d .gguf files; it has to hold exactly one.", folder, found.size()));
        return String();
    }
    return folder.path_join(found[0]);
}

bool ends_with(const std::string &text, const std::string &tail) {
    return tail.size() <= text.size() && text.compare(text.size() - tail.size(), tail.size(), tail) == 0;
}

// The tag at the end of the text with trailing blanks ignored. A template that opens the thought
// in its generation prompt puts a newline after the tag, and a plain end-of-string match misses
// it -- which leaves the thought uncounted and its ceiling unarmed.
bool ends_with_tag(const std::string &text, const std::string &tag) {
    if (tag.empty()) {
        return false;
    }
    size_t end = text.size();
    while (end > 0 && (unsigned char)text[end - 1] <= ' ') {
        end--;
    }
    return tag.size() <= end && text.compare(end - tag.size(), tag.size(), tag) == 0;
}

LlamaEvent piece_event(int64_t at, const String &text) {
    LlamaEvent event;
    event.kind = LlamaEvent::PIECE;
    event.at = at;
    event.text = text;
    return event;
}

LlamaEvent call_event(int64_t at, const String &id, const String &name, const String &arguments) {
    LlamaEvent event;
    event.kind = LlamaEvent::CALL;
    event.at = at;
    event.text = id;
    event.name = name;
    event.arguments = arguments;
    return event;
}

LlamaEvent finished_event(int64_t at, const String &reason, int prompt_tokens, int completion_tokens) {
    LlamaEvent event;
    event.kind = LlamaEvent::FINISHED;
    event.at = at;
    event.name = reason;
    event.prompt_tokens = prompt_tokens;
    event.completion_tokens = completion_tokens;
    return event;
}

LlamaEvent failed_event(int64_t at, const String &message) {
    LlamaEvent event;
    event.kind = LlamaEvent::FAILED;
    event.at = at;
    event.text = message;
    return event;
}

} // namespace

// Every sequence the library will ever open is given its storage here, once, and no load
// replaces it: a reader with no lock may then index an element that cannot have been freed,
// and how many of them are live is the atomic beside them rather than the vector's own size.
LlamaChat::LlamaChat() {
    const int ceiling = (int)llama_max_parallel_sequences();
    slots.reserve((size_t)ceiling);
    for (int i = 0; i < ceiling; i++) {
        slots.push_back(std::make_unique<LlamaSlot>());
    }
}

LlamaChat::~LlamaChat() {
    unload();
}

// The backends are opened once for the process and shared with every other class in this
// library, which is what keeps one folder from being registered twice over.
bool LlamaChat::ensure_backends() {
    return llama_runtime::ensure_backends();
}

void LlamaChat::shutdown_backends() {
    llama_runtime::shutdown_backends();
}

void LlamaChat::set_verbose(bool on) {
    llama_runtime::set_verbose(on);
}

// The knobs of the next load. Refused whole rather than in part: a dictionary half of which
// was taken would open a context nobody asked for and report the half that landed.
bool LlamaChat::set_load_options(const Dictionary &options) {
    bool wants_swa_full = swa_full;
    // Taken whole out of the Variant and range-checked below before it is narrowed: GDScript
    // holds an integer in 64 bits and the conversion to int truncates without a word.
    int64_t wants_slots = slots_asked.load();
    int wants_batch = n_batch.load();
    int wants_ubatch = n_ubatch.load();
    ggml_type wants_k = cache_type_k;
    ggml_type wants_v = cache_type_v;
    llama_flash_attn_type wants_fa = flash_attn_type;

    const Array keys = options.keys();
    for (int i = 0; i < keys.size(); i++) {
        const String key = keys[i];
        const Variant value = options[keys[i]];
        if (key == "swa_full") {
            wants_swa_full = value;
        } else if (key == "slots") {
            wants_slots = value;
        } else if (key == "n_batch") {
            wants_batch = value;
        } else if (key == "n_ubatch") {
            wants_ubatch = value;
        } else if (key == "cache_type_k" || key == "cache_type_v") {
            ggml_type named = GGML_TYPE_COUNT;
            if (!cache_type_named(String(value), named)) {
                UtilityFunctions::push_error("LlamaChat: \"" + String(value) + "\" is not a cache "
                        "type. f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0 and iq4_nl are.");
                return false;
            }
            (key == "cache_type_k" ? wants_k : wants_v) = named;
        } else if (key == "flash_attn") {
            const String word = String(value);
            if (word == "auto") {
                wants_fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
            } else if (word == "on") {
                wants_fa = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            } else if (word == "off") {
                wants_fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            } else {
                UtilityFunctions::push_error("LlamaChat: flash_attn is \"" + word
                        + "\"; auto, on and off are the words.");
                return false;
            }
        } else {
            UtilityFunctions::push_error("LlamaChat: \"" + key + "\" is not a load option. slots, "
                    "swa_full, n_batch, n_ubatch, cache_type_k, cache_type_v and flash_attn are.");
            return false;
        }
    }
    // The library opens at most llama_max_parallel_sequences() of them, and it says so by
    // throwing inside the load, after the weights have been read off the disk: refused here,
    // by name, so a host is told which number it asked for and what the ceiling is.
    const int64_t ceiling = (int64_t)llama_max_parallel_sequences();
    if (wants_slots < 1 || wants_slots > ceiling) {
        UtilityFunctions::push_error(vformat("LlamaChat: slots is %d, and a context holds at "
                "least one conversation and at most %d, which is the library's own ceiling. "
                "The context's tokens are the budget they share.", wants_slots, ceiling));
        return false;
    }
    if (wants_batch < 1 || wants_ubatch < 1 || wants_ubatch > wants_batch) {
        UtilityFunctions::push_error(vformat("LlamaChat: n_batch %d and n_ubatch %d are not a "
                "pair: both are above zero and the micro-batch is no wider than the batch.",
                wants_batch, wants_ubatch));
        return false;
    }

    swa_full = wants_swa_full;
    slots_asked.store((int)wants_slots);
    n_batch.store(wants_batch);
    n_ubatch.store(wants_ubatch);
    cache_type_k = wants_k;
    cache_type_v = wants_v;
    flash_attn_type = wants_fa;
    return true;
}

Dictionary LlamaChat::load_report() const {
    Dictionary out;
    out["swa_full"] = swa_full;
    out["slots"] = slots_asked.load();
    out["n_batch"] = n_batch.load();
    out["n_ubatch"] = n_ubatch.load();
    out["cache_type_k"] = String(ggml_type_name(cache_type_k));
    out["cache_type_v"] = String(ggml_type_name(cache_type_v));
    out["flash_attn"] = String(llama_flash_attn_type_name(flash_attn_type));
    out["context_size"] = context_tokens.load();
    std::string device_name;
    {
        std::lock_guard<std::mutex> hold(readers_lock);
        device_name = chosen_device;
    }
    out["device"] = to_gd(device_name);
    return out;
}

// The model and one context for it. A negative layer count puts every layer on the GPU with
// the most memory; zero keeps the whole model on the CPU. The load waits for a turn in
// flight, the way unload() does, and the context is warmed up so the first turn's time is
// the turn's own rather than the backend's pipeline compilation.
bool LlamaChat::load(const String &model_path, int n_ctx, int n_threads, int n_gpu_layers) {
    unload();
    llama_runtime::note_operation("LlamaChat was loading a model");
    if (!ensure_backends()) {
        return false;
    }
    const auto started = clock_type::now();

    String path = model_path;
    if (path.begins_with("res://") || path.begins_with("user://")) {
        path = ProjectSettings::get_singleton()->globalize_path(path);
    }
    if (DirAccess::dir_exists_absolute(path)) {
        path = single_gguf_in(path);
        if (path.is_empty()) {
            return false;
        }
    }

    common_params params;
    params.model.path = to_std(path);
    params.n_ctx = n_ctx > 0 ? n_ctx : 4096;
    params.n_batch = n_batch.load();
    params.n_ubatch = n_ubatch.load();
    // The sequences the context carries, and the one cache they all draw from. Left to itself
    // the library gives each sequence n_ctx divided by their number, which fixes in advance what
    // a conversation may ever hold; unified, n_ctx is a budget the occupied slots share, and a
    // long room stands beside a short exchange without either being sized for the other.
    params.n_parallel = slots_asked.load();
    params.kv_unified = true;
    params.n_gpu_layers = n_gpu_layers < 0 ? -1 : n_gpu_layers;
    params.fit_params = false;
    params.warmup = false;
    // The sliding-window layers keep the whole window rather than the model's own thousand.
    // Left short, llama.cpp empties them as soon as a turn generated more than the window
    // holds and says nothing, so the next turn reuses a prefix those layers no longer have.
    params.swa_full = swa_full;
    params.cache_type_k = cache_type_k;
    params.cache_type_v = cache_type_v;
    params.flash_attn_type = flash_attn_type;
    const int threads = n_threads > 0 ? n_threads : std::max(1, (int)std::thread::hardware_concurrency() / 2);
    params.cpuparams.n_threads = threads;
    params.cpuparams_batch.n_threads = threads;

    // The name is read on the main thread by load_report() and last_timings() while this runs,
    // so both ends of it are under the readers' lock and neither is under the model lock.
    {
        std::lock_guard<std::mutex> hold(readers_lock);
        chosen_device = "cpu";
    }
    device = nullptr;
    // Zero layers on the GPU has to mean the GPU is not used at all: left to itself, ggml
    // still hands a large prompt batch to a GPU that is present, and a measurement or a
    // machine whose driver misbehaves gets a "CPU" run that was not one.
    params.devices = { nullptr };
    if (params.n_gpu_layers != 0) {
        ggml_backend_dev_t best = llama_runtime::best_device(device_selector_);
        if (best != nullptr) {
            params.devices = { best, nullptr };
            // Read the file rather than map it: the weights are copied to the device and the
            // read buffer freed, so the process keeps no copy of the file in memory. Mapped,
            // the whole file stays in the working set, because Windows cannot unmap a part.
            params.load_mode = LLAMA_LOAD_MODE_NONE;
            const std::string named = llama_runtime::describe_device(best);
            {
                std::lock_guard<std::mutex> hold(readers_lock);
                chosen_device = named;
            }
            device = best;
        } else {
            params.n_gpu_layers = 0;
        }
    }
    // Noted before the weights are read, so a load that runs the card out of memory still says
    // which card it was and how much of it was free before anything was put on it.
    llama_runtime::note_device(device);

    std::lock_guard<std::mutex> hold(model_lock);
    try {
        runtime = common_init_from_params(params);
    } catch (const std::exception &e) {
        UtilityFunctions::push_error(String("LlamaChat: loading failed: ") + String::utf8(e.what()));
        runtime.reset();
        return false;
    }
    if (!runtime || runtime->model() == nullptr || runtime->context() == nullptr) {
        UtilityFunctions::push_error("LlamaChat: the model at \"" + path + "\" could not be loaded.");
        runtime.reset();
        return false;
    }
    model = runtime->model();
    ctx = runtime->context();
    vocab = llama_model_get_vocab(model);
    try {
        templates = common_chat_templates_init(model, "");
    } catch (const std::exception &e) {
        UtilityFunctions::push_error(String("LlamaChat: the model's chat template could not be read: ") + String::utf8(e.what()));
        templates.reset();
        runtime.reset();
        model = nullptr;
        ctx = nullptr;
        vocab = nullptr;
        return false;
    }
    // What one slot may hold, which with the one shared cache is the whole context: the ceiling
    // a prompt is measured against. Asked of the library rather than worked out, because it is
    // the library that decides whether the slots share the tokens or divide them.
    context_tokens.store((int)llama_n_ctx_seq(ctx));
    llama_runtime::note_operation("LlamaChat was warming the model up");
    warm_up(params.n_gpu_layers != 0);
    // After the warm-up and not before: the compute buffers are sized by the widest batch the
    // model is decoded at, and that is the batch the warm-up above has just put through it.
    remember_what_it_holds();
    // The storage below never moves; what changes is how many of it are live. A reader that
    // lands between the two stores reads zero and answers its empty value, and one either side
    // indexes an element that was built in the constructor and is never freed.
    opened.store(0, std::memory_order_release);
    const double took = ms_between(started, clock_type::now());
    load_ms.store(took);
    const int live = std::min((int)slots.size(), std::max(1, (int)llama_n_seq_max(ctx)));
    {
        std::lock_guard<std::mutex> readers(readers_lock);
        for (int i = 0; i < live; i++) {
            LlamaSlot &slot = *slots[i];
            slot.cached.clear();
            slot.kv_tokens.store(0);
            slot.drop_asked.store(false);
            slot.timings = LlamaTimings();
            slot.timings.load_ms = took;
        }
    }
    opened.store(live, std::memory_order_release);
    loaded.store(true);
    llama_runtime::note_operation("LlamaChat was waiting for a turn");
    return true;
}

// A backend builds its kernels for a batch shape the first time it meets one; on Vulkan that
// is seconds of pipeline compilation, which would otherwise land on the first turn. The
// widths a turn produces -- a full prompt batch, a short tail, one sampled token -- are
// decoded once here and thrown away, so the cost sits in load_ms where it belongs.
void LlamaChat::warm_up(bool every_width) {
    llama_token filler = llama_vocab_bos(vocab);
    if (filler == LLAMA_TOKEN_NULL) {
        filler = llama_vocab_eos(vocab);
    }
    if (filler == LLAMA_TOKEN_NULL) {
        filler = 0;
    }
    const int batch_width = n_batch.load();
    std::vector<int> widths = { std::min(batch_width, context_tokens.load() - 1), 1 };
    if (every_width) {
        widths = { std::min(batch_width, context_tokens.load() - 1), 128, 32, 8, 1 };
    }
    llama_batch batch = llama_batch_init(batch_width, 0, 1);
    for (const int width : widths) {
        common_batch_clear(batch);
        for (int i = 0; i < width; i++) {
            common_batch_add(batch, filler, (llama_pos)i, { (llama_seq_id)FIRST_SLOT }, i + 1 == width);
        }
        llama_decode(ctx, batch);
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    llama_batch_free(batch);
    llama_synchronize(ctx);
    llama_perf_context_reset(ctx);
}

// The worker is stopped and joined and then the lock is taken, in that order: the worker's
// turn holds the lock, so taking it first would wait on a thread that is waiting to be joined.
void LlamaChat::unload() {
    stop_asked.store(true);
    epoch.fetch_add(1);
    join_worker();
    {
        std::lock_guard<std::mutex> hold(model_lock);
        loaded.store(false);
        templates.reset();
        runtime.reset();
        model = nullptr;
        ctx = nullptr;
        vocab = nullptr;
        // The slots themselves stay, and so does the count of them: what the last turn of each
        // cost is still readable after the weights are gone, and what they held is not, because
        // the cache went with them.
        const int count = opened.load(std::memory_order_acquire);
        for (int i = 0; i < count; i++) {
            slots[i]->cached.clear();
            slots[i]->kv_tokens.store(0);
            slots[i]->drop_asked.store(false);
        }
        weights_bytes.store(0);
        kv_bytes.store(0);
        compute_bytes.store(0);
        host_bytes.store(0);
    }
    // What the joined worker left queued belongs to a model that is gone; the epoch would
    // drop it on delivery, and clearing it here keeps a stale finish from ever being read.
    {
        std::lock_guard<std::mutex> hold(events_lock);
        events.clear();
    }
    if (owed.exchange(false)) {
        busy.store(false);
    }
    stop_asked.store(false);
}

bool LlamaChat::is_loaded() const {
    return loaded.load();
}

bool LlamaChat::is_busy() const {
    return busy.load();
}

bool LlamaChat::generate(const Array &messages, const Array &tools, const Dictionary &options, int64_t slot) {
    BusyGuard guard(busy);
    if (!guard.taken()) {
        UtilityFunctions::push_error("LlamaChat: a turn is still running; wait for finished or cancel it.");
        return false;
    }
    if (!loaded.load()) {
        UtilityFunctions::push_error("LlamaChat: no model is loaded.");
        return false;
    }
    const int count = opened.load(std::memory_order_acquire);
    if (slot < 0 || slot >= (int64_t)count) {
        UtilityFunctions::push_error(not_one_of_the_slots(slot, count));
        return false;
    }

    LlamaTurn turn;
    // Narrowed here and nowhere earlier: the range above is what makes the cast lossless.
    turn.slot = (int)slot;
    String error;
    if (!read_messages(messages, turn.inputs.messages, error) || !read_tools(tools, turn.inputs.tools, error)) {
        UtilityFunctions::push_error("LlamaChat: " + error);
        return false;
    }
    if (turn.inputs.messages.empty()) {
        UtilityFunctions::push_error("LlamaChat: there is no message to answer.");
        return false;
    }
    turn.inputs.add_generation_prompt = true;
    turn.inputs.use_jinja = true;
    turn.inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    turn.inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    turn.inputs.enable_thinking = (bool)options.get("enable_thinking", false);
    turn.inputs.parallel_tool_calls = (bool)options.get("parallel_tool_calls", false);
    if (options.has("json_schema")) {
        turn.inputs.json_schema = json_text(options["json_schema"]);
    }
    turn.temperature = (float)(double)options.get("temperature", 0.7);
    turn.top_p = (float)(double)options.get("top_p", 0.8);
    turn.top_k = (int)options.get("top_k", 20);
    turn.min_p = (float)(double)options.get("min_p", 0.0);
    turn.max_tokens = (int)options.get("max_tokens", 512);
    if (turn.max_tokens <= 0) {
        turn.max_tokens = context_tokens.load();
    }
    turn.thinking_budget = (int)options.get("thinking_budget", 0);
    // Every penalty defaults to llama.cpp's own number, so a key nobody passed changes nothing.
    const common_params_sampling defaults;
    turn.penalty_repeat = (float)(double)options.get("penalty_repeat", defaults.penalty_repeat);
    turn.penalty_last_n = (int)options.get("penalty_last_n", defaults.penalty_last_n);
    turn.penalty_freq = (float)(double)options.get("penalty_freq", defaults.penalty_freq);
    turn.penalty_present = (float)(double)options.get("penalty_present", defaults.penalty_present);
    turn.dry_multiplier = (float)(double)options.get("dry_multiplier", defaults.dry_multiplier);
    turn.dry_base = (float)(double)options.get("dry_base", defaults.dry_base);
    turn.dry_allowed_length = (int)options.get("dry_allowed_length", defaults.dry_allowed_length);
    turn.dry_penalty_last_n = (int)options.get("dry_penalty_last_n", defaults.dry_penalty_last_n);
    if (options.has("seed")) {
        turn.seed = (uint32_t)(int64_t)options["seed"];
    }

    join_worker();
    stop_asked.store(false);
    // Marked owed before the thread exists: a delivery that raced this line would otherwise
    // lower the flag first and have the mark set over it afterwards.
    owed.store(true);
    // A thread that could not be started answers false with the flag given back, rather than
    // letting the system error unwind into the engine, which has no handler for one.
    try {
        worker = std::thread(&LlamaChat::work, this, std::move(turn), epoch.load());
    } catch (...) {
        owed.store(false);
        UtilityFunctions::push_error("LlamaChat: the worker thread could not be started.");
        return false;
    }
    guard.hand_on();
    return true;
}

// Never blocks: the flag is read by the worker before every token, and the finish it then
// delivers carries "cancelled". What was decoded stays in the context for the next turn.
void LlamaChat::cancel() {
    if (busy.load()) {
        stop_asked.store(true);
    }
}

// Queues one thing to say and asks the engine for a drain unless one is already on its
// way. The flag is what keeps a burst of a hundred pieces from queueing a hundred calls.
void LlamaChat::post(LlamaEvent event) {
    {
        std::lock_guard<std::mutex> hold(events_lock);
        events.push_back(std::move(event));
    }
    if (!drain_queued.exchange(true)) {
        callable_mp(this, &LlamaChat::deliver_pending).call_deferred();
    }
}

// The queue is swapped out under the lock and walked outside it, so a handler that starts
// the next turn from inside a finish does not run against a lock the worker wants.
void LlamaChat::deliver_pending() {
    drain_queued.store(false);
    std::vector<LlamaEvent> batch;
    {
        std::lock_guard<std::mutex> hold(events_lock);
        batch.swap(events);
    }
    for (const LlamaEvent &event : batch) {
        switch (event.kind) {
            case LlamaEvent::PIECE:
                deliver_piece(event.at, event.text);
                break;
            case LlamaEvent::CALL:
                deliver_call(event.at, event.text, event.name, event.arguments);
                break;
            case LlamaEvent::FINISHED:
                deliver_finished(event.at, event.name, event.prompt_tokens, event.completion_tokens);
                break;
            case LlamaEvent::FAILED:
                deliver_failed(event.at, event.text);
                break;
        }
    }
}

bool LlamaChat::wait_for_turn(int timeout_ms) {
    const auto deadline = clock_type::now() + std::chrono::milliseconds(std::max(0, timeout_ms));
    for (;;) {
        deliver_pending();
        if (!busy.load()) {
            return true;
        }
        if (clock_type::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

Dictionary LlamaChat::last_timings(int64_t slot) const {
    Dictionary out;
    const int count = opened.load(std::memory_order_acquire);
    // Nothing open has no sequence to name, and the empty reading is the answer there rather
    // than a refusal; an index the open context has not is refused whole, as a turn is.
    if (count == 0) {
        return out;
    }
    if (slot < 0 || slot >= (int64_t)count) {
        UtilityFunctions::push_error(not_one_of_the_slots(slot, count));
        return out;
    }
    // Copied out under the lock and read from the copy: the worker writes the eleven fields one
    // by one, and a reading taken across a write is two turns reported as one.
    LlamaTimings timings;
    std::string device_name;
    {
        std::lock_guard<std::mutex> hold(readers_lock);
        timings = slots[(int)slot]->timings;
        device_name = chosen_device;
    }
    out["load_ms"] = timings.load_ms;
    out["template_ms"] = timings.template_ms;
    out["prompt_ms"] = timings.prompt_ms;
    out["first_piece_ms"] = timings.first_piece_ms;
    out["generate_ms"] = timings.generate_ms;
    out["total_ms"] = timings.total_ms;
    out["prompt_tokens"] = timings.prompt_tokens;
    out["reused_tokens"] = timings.reused_tokens;
    out["decoded_tokens"] = timings.decoded_tokens;
    out["completion_tokens"] = timings.completion_tokens;
    out["reasoning_tokens"] = timings.reasoning_tokens;
    const double prompt_seconds = timings.prompt_ms / 1000.0;
    const double generate_seconds = timings.generate_ms / 1000.0;
    out["prompt_tokens_per_second"] = prompt_seconds > 0.0 ? timings.decoded_tokens / prompt_seconds : 0.0;
    out["tokens_per_second"] = generate_seconds > 0.0 ? timings.completion_tokens / generate_seconds : 0.0;
    out["device"] = to_gd(device_name);
    out["context_size"] = context_tokens.load();
    return out;
}

int LlamaChat::context_size() const {
    return context_tokens.load();
}

int LlamaChat::cached_tokens(int64_t slot) const {
    const int count = opened.load(std::memory_order_acquire);
    if (count == 0) {
        return 0;
    }
    if (slot < 0 || slot >= (int64_t)count) {
        UtilityFunctions::push_error(not_one_of_the_slots(slot, count));
        return 0;
    }
    return slots[(int)slot]->kv_tokens.load();
}

// How many conversations the open context holds, or how many the next load asks for where none
// is open. The count outlives an unload so the last turn's cost stays readable, so what is
// loaded decides which of the two is answered rather than whether anything was ever opened.
int LlamaChat::slot_count() const {
    const int count = opened.load(std::memory_order_acquire);
    if (!loaded.load() || count == 0) {
        return slots_asked.load();
    }
    return count;
}

// What every slot carrying tokens holds, as one phrase, so a sentence about a context that is
// full says which conversation to close rather than only that something filled it.
String LlamaChat::what_the_slots_hold(int count) const {
    String said;
    for (int i = 0; i < count; i++) {
        const int held = slots[i]->kv_tokens.load();
        if (held > 0) {
            said += (said.is_empty() ? String() : String(", "))
                    + vformat("slot %d: %d tokens", i, held);
        }
    }
    return said.is_empty() ? String("nothing") : said;
}

// Clears one slot now where nothing is decoding, and leaves the clear owed where something is:
// the cache may not be touched beside a running worker, and a caller that waited for one would
// hold the frame for a whole turn. The next turn honours what is owed before it decodes.
String LlamaChat::drop_slot(int64_t slot) {
    const int count = opened.load(std::memory_order_acquire);
    // Before the first load there is no sequence to name, and the index sentence would say a
    // slot is not one of the zero this context has while slot_count() answers what was asked for.
    if (count == 0) {
        return String("LlamaChat: no model is loaded.");
    }
    if (slot < 0 || slot >= (int64_t)count) {
        return not_one_of_the_slots(slot, count);
    }
    std::unique_lock<std::mutex> hold(model_lock, std::try_to_lock);
    if (!hold.owns_lock()) {
        slots[slot]->drop_asked.store(true);
        return String();
    }
    clear_slot((int)slot);
    return String();
}


void LlamaChat::clear_slot(int slot) {
    if (ctx != nullptr) {
        llama_memory_seq_rm(llama_get_memory(ctx), (llama_seq_id)slot, -1, -1);
    }
    slots[slot]->cached.clear();
    slots[slot]->kv_tokens.store(0);
    slots[slot]->drop_asked.store(false);
}

// What was given back while a turn held the model. Run in front of a turn, so nothing is
// compared against a prefix the host has let go of, and again behind it, so a seat given back
// during the last turn of a scene is not held until a turn that never comes.
void LlamaChat::drain_owed_drops(int count) {
    for (int i = 0; i < count; i++) {
        if (slots[i]->drop_asked.load()) {
            clear_slot(i);
        }
    }
}

Array LlamaChat::describe_devices() {
    return llama_runtime::describe_devices();
}

// Read off the context once, inside the load's own lock, and kept in the four atomics. Doing it
// here rather than on every call is what lets a caller ask while a turn is running.
void LlamaChat::remember_what_it_holds() {
    const Dictionary held = llama_runtime::memory_report_of(ctx);
    weights_bytes.store((int64_t)held["weights_bytes"]);
    kv_bytes.store((int64_t)held["kv_bytes"]);
    compute_bytes.store((int64_t)held["compute_bytes"]);
    host_bytes.store((int64_t)held["host_bytes"]);
}

Dictionary LlamaChat::memory_report() const {
    Dictionary out;
    out["weights_bytes"] = weights_bytes.load();
    out["kv_bytes"] = kv_bytes.load();
    out["compute_bytes"] = compute_bytes.load();
    out["host_bytes"] = host_bytes.load();
    return out;
}

// The device this model went on, or the one a model would go on where none is loaded: a host
// drawing a memory line before the first load still has a card to name.
Dictionary LlamaChat::device_memory() {
    ggml_backend_dev_t asked = device;
    // The backends are opened here rather than assumed: before the first load nothing else has
    // opened them, and an unopened registry has no devices at all to answer about.
    if (asked == nullptr && !loaded.load() && llama_runtime::ensure_backends()) {
        asked = llama_runtime::best_device(device_selector_);
    }
    return llama_runtime::device_memory_of(asked);
}

// The worker's whole life. An exception out of the turn is turned into a failure delivered
// on the main thread, where it would otherwise end the process.
void LlamaChat::work(LlamaTurn turn, int64_t at) {
    llama_runtime::note_operation("LlamaChat was running a turn");
    llama_runtime::note_device(device);
    try {
        run_turn(turn, at);
    } catch (const std::exception &e) {
        post(failed_event(at, String("LlamaChat: running the turn failed: ") + String::utf8(e.what())));
    } catch (...) {
        post(failed_event(at, String("LlamaChat: running the turn failed with an error that "
                                     "carries no words. The editor log carries whatever the "
                                     "library wrote before it.")));
    }
    llama_runtime::note_operation("LlamaChat was waiting for a turn");
}

// One turn in one slot: render, tokenize, drop that sequence past the first token that differs
// from what it holds, decode the rest, then sample under the template's grammar until the model
// stops, the budget runs out or cancel() is seen. Every piece of visible text goes out as soon
// as its last letter is whole; calls go out once the reply has ended cleanly.
void LlamaChat::run_turn(const LlamaTurn &turn, int64_t at) {
    std::lock_guard<std::mutex> hold(model_lock);
    const int count = opened.load(std::memory_order_acquire);
    if (!loaded.load() || at != epoch.load() || turn.slot < 0 || turn.slot >= count) {
        post(failed_event(at, "LlamaChat: the model was unloaded before the turn started."));
        return;
    }
    // What was dropped while a turn held the model is cleared here, before anything is compared
    // against a slot's tokens: a prefix kept out of a cache the host has given back is a reply
    // answering somebody else's conversation.
    drain_owed_drops(count);
    LlamaSlot &slot = *slots[turn.slot];
    const llama_seq_id sequence = (llama_seq_id)turn.slot;
    const int context_size_now = context_tokens.load();
    const auto started = clock_type::now();
    LlamaTimings cost;
    cost.load_ms = load_ms.load();

    // What a turn leaves behind, written under the readers' lock because the main thread reads
    // the eleven fields while this thread writes them, and the drop owed to a seat given back
    // mid-turn honoured here rather than in front of a turn that may never be asked for.
    auto leave = [&]() {
        {
            std::lock_guard<std::mutex> readers(readers_lock);
            slot.timings = cost;
        }
        drain_owed_drops(count);
    };

    // A refused turn writes down what it reached before it says so: what the last turn in this
    // slot cost is read right after a refusal, and a row left from the turn before is then a
    // measurement of something else, reported as this one's.
    auto fail = [&](const String &message) {
        cost.total_ms = ms_between(started, clock_type::now());
        leave();
        post(failed_event(at, message));
    };

    common_chat_params chat;
    try {
        chat = common_chat_templates_apply(templates.get(), turn.inputs);
    } catch (const std::exception &e) {
        fail(String("LlamaChat: the chat template refused the messages: ") + String::utf8(e.what()));
        return;
    }

    const llama_tokens prompt = common_tokenize(vocab, chat.prompt, true, true);
    cost.template_ms = ms_between(started, clock_type::now());
    // Written down in front of both gates, so a turn refused over its size leaves a row naming
    // that size. Left below them, the row says the turn cost nothing while its own sentence
    // names the thousands of tokens it was refused over.
    cost.prompt_tokens = (int)prompt.size();
    if (prompt.empty()) {
        fail("LlamaChat: the rendered prompt is empty.");
        return;
    }
    // No count here: a prompt bigger than the whole context is refused whatever the neighbours
    // hold, and how many sequences there are is not the number that makes this one actionable.
    if ((int)prompt.size() + 1 > context_size_now) {
        fail(vformat("LlamaChat: the prompt is %d tokens and the context holds %d, which every "
                "slot of it shares.", (int)prompt.size(), context_size_now));
        return;
    }
    // The slots draw on one budget, so what this turn may have is the context less what the
    // others are holding. Measured in front of the decode: inside it the library writes its own
    // line to stderr and the failure a host is handed carries no number at all. It cannot move
    // while this turn runs -- a drop only raises a flag, and no other turn decodes beside it --
    // so the generation ceiling below is measured against this same number.
    int counted = 0;
    for (int i = 0; i < count; i++) {
        if (i != turn.slot) {
            counted += slots[i]->kv_tokens.load();
        }
    }
    const int held_elsewhere = counted;
    if ((int)prompt.size() + 1 + held_elsewhere > context_size_now) {
        fail(vformat("LlamaChat: the prompt is %d tokens, the other slot(s) hold %d, and the "
                "context holds %d, which they share. Give a finished conversation's slot back, "
                "or raise the context.",
                (int)prompt.size(), held_elsewhere, context_size_now));
        return;
    }

    // The prefix already decoded is kept; when the whole prompt is there the last token is
    // decoded again, because a turn needs the logits of its final token to sample from.
    size_t keep = 0;
    while (keep < slot.cached.size() && keep < prompt.size() && slot.cached[keep] == prompt[keep]) {
        keep++;
    }
    if (keep == prompt.size()) {
        keep--;
    }
    // This slot's own tokens and no others: clearing the whole cache here would empty the
    // conversations beside it, which would then pay for their whole prompt again in silence.
    llama_memory_t memory = llama_get_memory(ctx);
    // A recurrent or hybrid memory refuses a partial removal, and a whole sequence never does.
    // Left unanswered, the slot would decode from a position the memory does not agree with and
    // every later turn in it would be refused the same way until somebody dropped it.
    if (!llama_memory_seq_rm(memory, sequence, (llama_pos)keep, -1)) {
        clear_slot(turn.slot);
        keep = 0;
    }
    slot.cached.resize(keep);
    slot.kv_tokens.store((int)slot.cached.size());
    // Counted from the keep that survived the removal above, so a turn that fell back to
    // nothing does not report a prefix it never kept.
    cost.reused_tokens = (int)keep;
    cost.decoded_tokens = (int)(prompt.size() - keep);

    const int batch_width = n_batch.load();
    llama_batch batch = llama_batch_init(batch_width, 0, 1);
    struct BatchFree {
        llama_batch &batch;
        ~BatchFree() { llama_batch_free(batch); }
    } batch_free{ batch };

    const auto prompt_started = clock_type::now();
    for (size_t from = keep; from < prompt.size();) {
        if (stop_asked.load()) {
            cost.total_ms = ms_between(started, clock_type::now());
            leave();
            post(finished_event(at, String("cancelled"), (int)prompt.size(), 0));
            return;
        }
        const size_t to = std::min(prompt.size(), from + (size_t)batch_width);
        common_batch_clear(batch);
        for (size_t i = from; i < to; i++) {
            common_batch_add(batch, prompt[i], (llama_pos)i, { sequence }, i + 1 == prompt.size());
        }
        const int answered = llama_decode(ctx, batch);
        if (answered != 0) {
            // 1 is answered before a single ubatch is processed and the memory is left as it
            // was, so what this slot holds is still exactly what its book says; an abort or a
            // fatal error leaves behind the ubatches that ran, and only a whole-sequence
            // removal can be sure of clearing those.
            if (answered != 1) {
                clear_slot(turn.slot);
            }
            fail(the_decode_refused(answered, "the prompt", context_size_now,
                    what_the_slots_hold(count)));
            return;
        }
        for (size_t i = from; i < to; i++) {
            slot.cached.push_back(prompt[i]);
        }
        slot.kv_tokens.store((int)slot.cached.size());
        from = to;
    }
    cost.prompt_ms = ms_between(prompt_started, clock_type::now());

    // The sampler carries the template's grammar: lazy, opened by the trigger words, so
    // speech is free and a call is constrained to the declared schemas.
    common_params_sampling sampling;
    sampling.seed = turn.seed;
    sampling.temp = turn.temperature;
    sampling.top_p = turn.top_p;
    sampling.top_k = turn.top_k;
    sampling.min_p = turn.min_p;
    sampling.penalty_repeat = turn.penalty_repeat;
    sampling.penalty_last_n = turn.penalty_last_n;
    sampling.penalty_freq = turn.penalty_freq;
    sampling.penalty_present = turn.penalty_present;
    sampling.dry_multiplier = turn.dry_multiplier;
    sampling.dry_base = turn.dry_base;
    sampling.dry_allowed_length = turn.dry_allowed_length;
    sampling.dry_penalty_last_n = turn.dry_penalty_last_n;
    if (!chat.grammar.empty()) {
        const common_grammar_type kind =
                turn.inputs.tools.empty() ? COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT : COMMON_GRAMMAR_TYPE_TOOL_CALLS;
        sampling.grammar = common_grammar(kind, chat.grammar);
        sampling.grammar_lazy = chat.grammar_lazy;
        sampling.grammar_triggers = chat.grammar_triggers;
    }
    sampling.generation_prompt = chat.generation_prompt;
    std::set<llama_token> preserved;
    for (const std::string &word : chat.preserved_tokens) {
        const llama_tokens ids = common_tokenize(vocab, word, false, true);
        if (ids.size() == 1) {
            preserved.insert(ids[0]);
        }
    }
    sampling.preserved_tokens = preserved;

    // The thought's ceiling and the count below, armed from the template's own thinking markers
    // the way the server arms it: the budget spent, the first end marker is forced and the turn
    // goes on into the visible answer. A template with no markers arms nothing, and neither does
    // a turn with thinking off, whatever empty thought the template makes the model write.
    const std::string thinking_start = turn.inputs.enable_thinking ? chat.thinking_start_tag : std::string();
    std::vector<std::string> thinking_end;
    for (const std::string &tag : chat.thinking_end_tags) {
        if (!tag.empty()) {
            thinking_end.push_back(tag);
        }
    }
    // After a tool result the template opens the thought itself, tag and newline both inside the
    // generation prompt. The budget is armed from the prompt's own trailing tokens rather than
    // from the tag tokenized alone, so the seeding matches and counting starts with this turn.
    const bool prompt_opens_thought = ends_with_tag(chat.generation_prompt, thinking_start);
    llama_tokens opened_by;
    if (prompt_opens_thought) {
        const llama_tokens prompt_ids = common_tokenize(vocab, chat.generation_prompt, false, true);
        const size_t reach = std::min<size_t>(prompt_ids.size(), 8);
        std::string tail;
        for (size_t take = 1; take <= reach; take++) {
            tail = common_token_to_piece(vocab, prompt_ids[prompt_ids.size() - take], true) + tail;
            if (ends_with_tag(tail, thinking_start)) {
                opened_by.assign(prompt_ids.end() - take, prompt_ids.end());
                break;
            }
        }
    }
    if (turn.thinking_budget > 0 && !thinking_start.empty() && !thinking_end.empty()) {
        sampling.reasoning_budget_tokens = turn.thinking_budget;
        sampling.reasoning_budget_start =
                opened_by.empty() ? common_tokenize(vocab, thinking_start, false, true) : opened_by;
        for (const std::string &tag : thinking_end) {
            sampling.reasoning_budget_end.push_back(common_tokenize(vocab, tag, false, true));
        }
        sampling.reasoning_budget_forced = sampling.reasoning_budget_end.front();
    }

    common_sampler_ptr sampler(common_sampler_init(model, sampling));
    if (!sampler) {
        fail("LlamaChat: the sampler could not be built from the template's grammar.");
        return;
    }

    common_chat_parser_params parsing(chat);
    parsing.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    parsing.parse_tool_calls = true;
    if (!chat.parser.empty()) {
        try {
            parsing.parser.load(chat.parser);
        } catch (const std::exception &e) {
            fail(String("LlamaChat: the template's parser could not be loaded: ") + String::utf8(e.what()));
            return;
        }
    }

    std::string generated;
    std::string visible;
    size_t sent = 0;
    common_chat_msg parsed;
    std::vector<std::string> ids_cache;
    auto mint_id = [this]() { return "call_" + std::to_string(++call_serial); };
    bool first_piece = true;

    // What the parser now takes the reply to be, diffed against the last reading so that only
    // new visible text is queued; a reading that fails keeps the previous one.
    auto reread = [&](bool is_partial) {
        try {
            common_chat_msg reading = common_chat_parse(generated, is_partial, parsing);
            if (reading.empty()) {
                return;
            }
            reading.set_tool_call_ids(ids_cache, mint_id);
            for (const common_chat_msg_diff &diff : common_chat_msg_diff::compute_diffs(parsed, reading)) {
                visible += diff.content_delta;
            }
            parsed = reading;
        } catch (const std::exception &) {
            if (!is_partial) {
                parsed.content = generated;
                visible = generated;
            }
        }
    };

    auto flush = [&]() {
        const size_t end = utf8_complete_end(visible, sent);
        if (end > sent) {
            if (first_piece) {
                first_piece = false;
                cost.first_piece_ms = ms_between(started, clock_type::now());
            }
            post(piece_event(at, String::utf8(visible.data() + sent, (int)(end - sent))));
            sent = end;
        }
    };

    // The thought is counted from the token after the start marker to the one that completes an
    // end marker, forced or reached on its own. Counted whether or not a budget was given, so a
    // caller can see what an unbounded thought cost before deciding on a ceiling.
    int reasoning_tokens = 0;
    // A template whose generation prompt already opens the thought leaves nothing to see in the
    // output; the sampler is armed from that prompt too, so the count starts open with it.
    bool inside_thought = prompt_opens_thought;
    bool thought_over = false;

    std::string reason = "stop";
    int produced = 0;
    const auto generate_started = clock_type::now();
    for (;;) {
        if (stop_asked.load()) {
            reason = "cancelled";
            break;
        }
        // The neighbours are counted here as the entry gate counts them: with one shared cache
        // the room left is the context less what they hold, and a loop bounded by the context
        // alone runs into a decode that refuses, after the host has already had the pieces.
        if (produced >= turn.max_tokens
                || (int)slot.cached.size() + 1 + held_elsewhere >= context_size_now) {
            reason = "length";
            break;
        }
        const llama_token token = common_sampler_sample(sampler.get(), ctx, -1);
        common_sampler_accept(sampler.get(), token, true);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        produced++;
        generated += common_token_to_piece(vocab, token, preserved.count(token) > 0);
        if (!thought_over && !thinking_start.empty()) {
            if (inside_thought) {
                reasoning_tokens++;
                for (const std::string &tag : thinking_end) {
                    if (ends_with(generated, tag)) {
                        thought_over = true;
                        break;
                    }
                }
            } else if (ends_with(generated, thinking_start)) {
                inside_thought = true;
            }
        }
        reread(true);
        flush();

        bool stopped = false;
        for (const std::string &stop : chat.additional_stops) {
            if (!stop.empty() && ends_with(generated, stop)) {
                stopped = true;
            }
        }
        common_batch_clear(batch);
        common_batch_add(batch, token, (llama_pos)slot.cached.size(), { sequence }, true);
        const int answered = llama_decode(ctx, batch);
        if (answered != 0) {
            // As above: only an abort or a fatal error leaves cells this slot's book has not
            // recorded, and only those are worth the whole prefix a clear costs.
            if (answered != 1) {
                clear_slot(turn.slot);
            }
            fail(the_decode_refused(answered, "a generated token", context_size_now,
                    what_the_slots_hold(count)));
            return;
        }
        slot.cached.push_back(token);
        slot.kv_tokens.store((int)slot.cached.size());
        if (stopped) {
            break;
        }
    }
    cost.generate_ms = ms_between(generate_started, clock_type::now());

    // The final reading is whole only for a reply the model ended itself. A cut reply keeps
    // its partial reading, and its unfinished calls are never announced.
    reread(reason != "stop");
    flush();
    if (reason == "stop") {
        for (const common_chat_tool_call &call : parsed.tool_calls) {
            post(call_event(at, to_gd(call.id), to_gd(call.name), to_gd(call.arguments)));
        }
        if (!parsed.tool_calls.empty()) {
            reason = "tool_calls";
        }
    }

    cost.completion_tokens = produced;
    cost.reasoning_tokens = reasoning_tokens;
    cost.total_ms = ms_between(started, clock_type::now());
    leave();
    post(finished_event(at, to_gd(reason), (int)prompt.size(), produced));
}

// The deliveries, on the main thread. A turn whose model was unloaded or replaced while it
// ran is dropped: the flag it would clear belongs to whatever was started after it.
void LlamaChat::deliver_piece(int64_t at, const String &text) {
    if (at != epoch.load()) {
        return;
    }
    emit_signal("piece_arrived", text);
}

void LlamaChat::deliver_call(int64_t at, const String &id, const String &name, const String &arguments) {
    if (at != epoch.load()) {
        return;
    }
    emit_signal("tool_called", id, name, arguments);
}

void LlamaChat::deliver_finished(int64_t at, const String &reason, int prompt_tokens, int completion_tokens) {
    if (at != epoch.load() || !busy.load()) {
        return;
    }
    owed.store(false);
    busy.store(false);
    emit_signal("finished", reason, prompt_tokens, completion_tokens);
}

void LlamaChat::deliver_failed(int64_t at, const String &message) {
    if (at != epoch.load() || !busy.load()) {
        return;
    }
    owed.store(false);
    busy.store(false);
    emit_signal("failed", message);
}

void LlamaChat::join_worker() {
    if (worker.joinable()) {
        worker.join();
    }
}

// The card a host names for every library in the process. It is read at the load and never
// after: a model already on a device stays where it was put.
void LlamaChat::set_device_selector(const godot::String &address) {
    device_selector_ = address.utf8().get_data();
}

godot::String LlamaChat::get_device_selector() const {
    return godot::String(device_selector_.c_str());
}

// The card this model really went to, as the address every backend that reports one writes the
// same way. Empty before a load and from a model on the processor, which is on no card.
godot::String LlamaChat::device_identity() const {
    return godot::String(llama_runtime::address_of(device).c_str());
}

void LlamaChat::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_load_options", "options"), &LlamaChat::set_load_options);
    ClassDB::bind_method(D_METHOD("load_report"), &LlamaChat::load_report);
    ClassDB::bind_method(D_METHOD("set_device_selector", "address"), &LlamaChat::set_device_selector);
    ClassDB::bind_method(D_METHOD("get_device_selector"), &LlamaChat::get_device_selector);
    ClassDB::bind_method(D_METHOD("device_identity"), &LlamaChat::device_identity);
    ClassDB::bind_method(D_METHOD("load", "model_path", "n_ctx", "n_threads", "n_gpu_layers"), &LlamaChat::load);
    ClassDB::bind_method(D_METHOD("unload"), &LlamaChat::unload);
    ClassDB::bind_method(D_METHOD("is_loaded"), &LlamaChat::is_loaded);
    ClassDB::bind_method(D_METHOD("is_busy"), &LlamaChat::is_busy);
    ClassDB::bind_method(D_METHOD("generate", "messages", "tools", "options", "slot"),
            &LlamaChat::generate, DEFVAL(FIRST_SLOT));
    ClassDB::bind_method(D_METHOD("cancel"), &LlamaChat::cancel);
    ClassDB::bind_method(D_METHOD("deliver_pending"), &LlamaChat::deliver_pending);
    ClassDB::bind_method(D_METHOD("wait_for_turn", "timeout_ms"), &LlamaChat::wait_for_turn);
    ClassDB::bind_method(D_METHOD("slot_count"), &LlamaChat::slot_count);
    ClassDB::bind_method(D_METHOD("drop_slot", "slot"), &LlamaChat::drop_slot);
    ClassDB::bind_method(D_METHOD("last_timings", "slot"), &LlamaChat::last_timings,
            DEFVAL(FIRST_SLOT));
    ClassDB::bind_method(D_METHOD("context_size"), &LlamaChat::context_size);
    ClassDB::bind_method(D_METHOD("cached_tokens", "slot"), &LlamaChat::cached_tokens,
            DEFVAL(FIRST_SLOT));
    ClassDB::bind_method(D_METHOD("memory_report"), &LlamaChat::memory_report);
    ClassDB::bind_method(D_METHOD("device_memory"), &LlamaChat::device_memory);
    ClassDB::bind_method(D_METHOD("describe_devices"), &LlamaChat::describe_devices);
    ClassDB::bind_static_method("LlamaChat", D_METHOD("set_verbose", "on"), &LlamaChat::set_verbose);

    ADD_SIGNAL(MethodInfo("piece_arrived", PropertyInfo(Variant::STRING, "text")));
    ADD_SIGNAL(MethodInfo("tool_called", PropertyInfo(Variant::STRING, "id"), PropertyInfo(Variant::STRING, "name"),
            PropertyInfo(Variant::STRING, "arguments_json")));
    ADD_SIGNAL(MethodInfo("finished", PropertyInfo(Variant::STRING, "reason"), PropertyInfo(Variant::INT, "prompt_tokens"),
            PropertyInfo(Variant::INT, "completion_tokens")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::STRING, "message")));
}
