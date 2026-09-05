#ifndef LLAMA_CHAT_H
#define LLAMA_CHAT_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace godot {

// One turn as the worker receives it: the messages and tools already in llama.cpp's own
// shape, and the sampling knobs. Gathered on the caller's thread so the worker never reads
// a Variant, which is what keeps the engine's reference counting off that thread.
struct LlamaTurn {
    common_chat_templates_inputs inputs;
    float temperature = 0.7f;
    float top_p = 0.8f;
    int top_k = 20;
    float min_p = 0.0f;
    int max_tokens = 512;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

// What the last turn cost. Written by the worker before the finish is delivered and read on
// the main thread after it; read during a turn they are the turn before.
struct LlamaTimings {
    double load_ms = 0.0;
    double template_ms = 0.0;
    double prompt_ms = 0.0;
    double first_piece_ms = 0.0;
    double generate_ms = 0.0;
    double total_ms = 0.0;
    int prompt_tokens = 0;
    int reused_tokens = 0;
    int decoded_tokens = 0;
    int completion_tokens = 0;
};

// One thing the worker has to say, kept on the object until the main thread hands it out.
// A piece fills text; a call fills text with its id, then name and arguments; a finish
// fills name with the reason and the two counts; a failure fills text with the sentence.
struct LlamaEvent {
    enum Kind { PIECE, CALL, FINISHED, FAILED };
    Kind kind = PIECE;
    int64_t at = 0;
    String text;
    String name;
    String arguments;
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

// A chat with one local model: the model's own chat template renders the history and the
// tool declarations, the template's grammar constrains a call, and the template's parser
// separates what the model says from what it calls. One llama_context lives as long as the
// model is loaded, and a turn decodes only the tokens that differ from the ones already in it.
//
// generate() hands the turn to one worker thread and refuses while one runs. The worker
// queues what it has to say on the object; a deferred call drains the queue on the main
// thread after each burst, and deliver_pending() drains it for a caller that draws no
// frames. unload(), load() and the destructor wait for the worker.
class LlamaChat : public RefCounted {
    GDCLASS(LlamaChat, RefCounted)

    // The one worker, the flag that says the model is taken, and whether the worker owns
    // that flag. The flag is raised in one atomic step and lowered by the delivery of the
    // finish on the main thread, or by unload() when that delivery is never coming.
    std::thread worker;
    std::atomic<bool> busy{false};
    std::atomic<bool> owed{false};

    // What the worker has said and nobody has handed out yet, and whether a deferred drain
    // is already on its way: one drain per burst of pieces rather than one call per piece.
    std::mutex events_lock;
    std::vector<LlamaEvent> events;
    std::atomic<bool> drain_queued{false};

    // Read once per token by the worker; raised by cancel() and by unload().
    std::atomic<bool> stop_asked{false};

    // Which model a turn was started against. A delivery whose epoch has moved on is dropped
    // rather than reported for a model that is gone.
    std::atomic<int64_t> epoch{0};

    // Held by every turn and by whatever frees or replaces the model.
    std::mutex model_lock;
    std::atomic<bool> loaded{false};

    common_init_result_ptr runtime;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    common_chat_templates_ptr templates;

    // The tokens whose entries the context holds, in order. The next turn's prompt is
    // compared against them and only the tail past the first difference is decoded.
    std::vector<llama_token> cached;
    std::atomic<int> kv_tokens{0};
    int context_tokens = 0;
    int n_batch = 512;

    // Identifiers for calls the template leaves unnamed, unique for the life of the object.
    int64_t call_serial = 0;

    // The device this model was put on, as ggml names it, or "cpu". Per object rather than
    // per process: a second model may be loaded on another device beside this one. The handle
    // beside it is what the crash line reads its free memory from; null means the processor.
    std::string chosen_device;
    ggml_backend_dev_t device = nullptr;

    // What the library allocated for this model, taken once at the end of the load and kept.
    // Read outside the model lock on purpose: asking the context during a turn would hold the
    // caller for the whole turn, and none of the four moves once the load has returned.
    std::atomic<int64_t> weights_bytes{0};
    std::atomic<int64_t> kv_bytes{0};
    std::atomic<int64_t> compute_bytes{0};
    std::atomic<int64_t> host_bytes{0};

    LlamaTimings timings;

protected:
    static void _bind_methods();

public:
    LlamaChat() = default;
    ~LlamaChat();

    // A .gguf file, or the folder holding exactly one; an OS path or a res:// or user:// one.
    bool load(const String &model_path, int n_ctx, int n_threads, int n_gpu_layers);
    void unload();
    bool is_loaded() const;
    bool is_busy() const;

    // Messages in the OpenAI chat shape, tools as OpenAI function declarations, and the
    // options temperature, top_p, top_k, min_p, max_tokens, seed, enable_thinking,
    // parallel_tool_calls, json_schema. False when the model is not loaded, a turn is
    // running, or the messages do not read.
    bool generate(const Array &messages, const Array &tools, const Dictionary &options);
    void cancel();

    // Hands out every signal the worker has queued, on the calling thread, in order. The
    // engine calls it deferred after each burst; a caller that draws no frames calls it.
    void deliver_pending();

    // Blocks the calling thread, draining as it waits, until the turn in flight has been
    // handed out or the wait runs out; answers whether the turn is over. For a caller with
    // no frames -- a headless test, a tool -- and never for a game, which has frames.
    bool wait_for_turn(int timeout_ms);

    Dictionary last_timings() const;
    int context_size() const;
    int cached_tokens() const;

    // What this model holds, as the library counted it when the load finished: weights_bytes,
    // kv_bytes, compute_bytes, and host_bytes for the part of the three that is ordinary memory
    // rather than the card's. Zeroes with nothing loaded. Answered without waiting for a turn.
    Dictionary memory_report() const;

    // The free and total memory of the device this model was put on, or of the best device on
    // the machine before anything is loaded: free_bytes, total_bytes and name. Both numbers are
    // -1 where the machine has no device or its backend will not say, and both are about the
    // whole card rather than this process's share of it.
    Dictionary device_memory();

    // The ggml devices the backends found, one dictionary each, after the backends are
    // loaded. Opening them here when they are not yet is what a caller uses to see the GPU.
    Array describe_devices();

    // Whether llama.cpp's own log lines below warning level reach the console.
    static void set_verbose(bool on);

    // Frees what the backends hold. Called when the library is unloaded.
    static void shutdown_backends();

private:
    static bool ensure_backends();

    void remember_what_it_holds();
    void warm_up(bool every_width);
    void work(LlamaTurn turn, int64_t at);
    void run_turn(const LlamaTurn &turn, int64_t at);
    void post(LlamaEvent event);
    void deliver_piece(int64_t at, const String &text);
    void deliver_call(int64_t at, const String &id, const String &name, const String &arguments);
    void deliver_finished(int64_t at, const String &reason, int prompt_tokens, int completion_tokens);
    void deliver_failed(int64_t at, const String &message);
    void join_worker();
};

} // namespace godot

#endif // LLAMA_CHAT_H
