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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace godot {

// One turn as the worker receives it: the messages and tools already in llama.cpp's own
// shape, and the sampling knobs. Gathered on the caller's thread so no Variant a caller owns
// ever crosses to the worker, and no signal is ever emitted from it.
struct LlamaTurn {
    common_chat_templates_inputs inputs;
    float temperature = 0.7f;
    float top_p = 0.8f;
    int top_k = 20;
    float min_p = 0.0f;
    int max_tokens = 512;
    // How many tokens the thought may spend before its end marker is forced; 0 or less leaves
    // it unbounded. Without it a long thought reaches max_tokens and the turn says nothing.
    int thinking_budget = 0;
    // Which slot of the context this turn is decoded into. Each slot keeps its own tokens and
    // its own positions, so a turn reads and writes nothing of the conversations beside it.
    int slot = 0;
    // What a repetition costs this turn, under llama.cpp's own names. The penalties sampler and
    // DRY are already in the chain; at these defaults both pass everything through, so a caller
    // that asks for nothing gets the sampling it had.
    float penalty_repeat = 1.0f;
    int penalty_last_n = 64;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    float dry_multiplier = 0.0f;
    float dry_base = 1.75f;
    int dry_allowed_length = 2;
    int dry_penalty_last_n = 64;
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
    // The part of completion_tokens the thought took, counted between the template's thinking
    // markers. Zero for a template that has none and for a turn that never opened a thought.
    int reasoning_tokens = 0;
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

// One slot of the context: the tokens that sequence holds, in order, how many of them the
// cache carries, whether a clear is owed, and what the last turn run there cost. Held behind a
// pointer so the list of them never has to move an atomic, and never has to move at all.
struct LlamaSlot {
    std::vector<llama_token> cached;
    std::atomic<int> kv_tokens{0};
    // Raised by drop_slot() while a turn holds the model, and honoured before the next turn
    // decodes anything. Without it a drop would either block the caller or race the worker.
    std::atomic<bool> drop_asked{false};
    LlamaTimings timings;
};

// A chat with one local model: the model's own chat template renders the history and the
// tool declarations, the template's grammar constrains a call, and the template's parser
// separates what the model says from what it calls. One llama_context lives as long as the
// model is loaded, and a turn decodes only the tokens that differ from the ones already in it.
//
// The context carries several sequences, one per slot, and several conversations answer through
// it without reading each other's prompts: a slot keeps its own tokens, its own positions and
// its own reuse, and the context's tokens are one budget the occupied slots draw from.
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

    // One entry per sequence the library will ever open, built once in the constructor and
    // never cleared, resized or reallocated: slots.size() is llama_max_parallel_sequences()
    // and says nothing about this context. A turn compares its prompt against its own slot's
    // tokens and decodes only the tail past the first difference.
    //
    // How many of them are live is `opened`, and every reader below takes one snapshot of it
    // and bounds itself by that copy. The readers take no model lock -- a turn holds it for its
    // whole length and a reader that waited would hold the frame with it -- so one that lands
    // mid-load reads zero and answers its empty value; one that lands either side indexes an
    // element nothing in this class ever frees.
    std::vector<std::unique_ptr<LlamaSlot>> slots;
    std::atomic<int> opened{0};

    // Read by the main thread while the worker writes it: what every turn leaves in its slot's
    // timings, and the device name the load writes. Small enough to be held across eleven field
    // copies and one string copy, and never held across a decode or across a load.
    mutable std::mutex readers_lock;

    // How many sequences the next load opens the context with, and how many the open one has.
    // The slots share one cache of the context's tokens, so context_tokens below is the whole
    // of it and what a conversation may hold is bounded by what the others are holding.
    //
    // The four are written on the caller's thread or the loader's and read on the worker's and
    // the main one, so they are atomics rather than plain ints: a reading taken across a write
    // here is a context of the wrong size.
    std::atomic<int> slots_asked{1};

    std::atomic<int> context_tokens{0};
    std::atomic<int> n_batch{512};
    std::atomic<int> n_ubatch{512};

    // A sliding window that keeps every position rather than the model's own thousand. On a
    // model with such layers the short window silently empties whenever a turn generated more
    // tokens than it holds, and the prefix the next turn keeps is then a prefix of nothing.
    bool swa_full = true;

    // What the cache is stored as, and whether flash attention is forced either way. A
    // quantized V needs flash attention; asked for without it the context refuses to open.
    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    // Identifiers for calls the template leaves unnamed, unique for the life of the object.
    int64_t call_serial = 0;

    // The device this model was put on, as ggml names it, or "cpu". Per object rather than
    // per process: a second model may be loaded on another device beside this one. The handle
    // beside it is what the crash line reads its free memory from; null means the processor.
    //
    // Written by the loader thread and read by the main one, so both ends are under
    // readers_lock: a std::string assignment that reallocates under a reader is a read of
    // freed memory, which no atomic on the four counts below would have caught.
    std::string chosen_device;
    ggml_backend_dev_t device = nullptr;

    // What the library allocated for this model, taken once at the end of the load and kept.
    // Read outside the model lock on purpose: asking the context during a turn would hold the
    // caller for the whole turn, and none of the four moves once the load has returned.
    std::atomic<int64_t> weights_bytes{0};
    std::atomic<int64_t> kv_bytes{0};
    std::atomic<int64_t> compute_bytes{0};
    std::atomic<int64_t> host_bytes{0};

    // What the load itself took, copied into every slot's timings so a turn reports it too.
    // Written on the loader thread and read on the worker's and the main one, so it is an
    // atomic of its own rather than a field of the struct the lock above covers.
    std::atomic<double> load_ms{0.0};

protected:
    static void _bind_methods();

public:
    // The PCI address of the card this model is to take -- "0000:c1:00.0" -- set before the load
    // and read at it. Empty is this library's own ranking, which is what a host that runs nothing
    // else beside it wants; a host running a second library names one card for both.
    void set_device_selector(const godot::String &address);
    godot::String get_device_selector() const;

    // The PCI address of the device this model is actually on, or "" where it is on
    // none. A host holds it against the other library's answer: one address is one card.
    godot::String device_identity() const;

    // Builds the storage for every sequence the library will ever open, so no load ever has to
    // replace it under a reader. What it costs is one allocation per entry, once per object.
    LlamaChat();
    ~LlamaChat();

    // What the next load opens the context with, beyond its four numbers: slots, swa_full,
    // n_batch, n_ubatch, cache_type_k, cache_type_v ("f16", "q8_0", …) and flash_attn
    // ("auto", "on", "off"). A key that is absent leaves that knob where it stands, and a
    // value that is not one of the words is refused with a sentence rather than guessed at.
    // Read at the next load and never during one, so a model already open does not move.
    bool set_load_options(const Dictionary &options);

    // What the load was ASKED for -- the six above and the four numbers -- and not what the
    // library resolved: flash_attn stays "auto" where the probe decided it, and llama.cpp
    // exposes no getter for the resolved value. Its own log line is where that is read.
    Dictionary load_report() const;

    // A .gguf file, or the folder holding exactly one; an OS path or a res:// or user:// one.
    bool load(const String &model_path, int n_ctx, int n_threads, int n_gpu_layers);
    void unload();
    bool is_loaded() const;
    bool is_busy() const;

    // Messages in the OpenAI chat shape, tools as OpenAI function declarations, and the
    // options temperature, top_p, top_k, min_p, max_tokens, thinking_budget, seed,
    // enable_thinking, parallel_tool_calls, json_schema and the penalties above. The slot is
    // which conversation of the context answers; false when the model is not loaded, a turn is
    // running, the slot is not one the context has, or the messages do not read.
    bool generate(const Array &messages, const Array &tools, const Dictionary &options, int64_t slot = 0);
    void cancel();

    // How many conversations the context holds at once. Which conversation owns which slot is
    // the host's own book, kept outside this class, and nothing here can answer it.
    int slot_count() const;

    // Clears one slot's tokens out of the cache, so the next turn there starts from nothing.
    // Answers "" when it was done or is owed, and the sentence saying why when nothing is
    // loaded or the slot is not one the context has. A slot dropped while a turn runs is
    // cleared behind that turn, or in front of the next one. The index is taken whole: a number
    // GDScript holds as 64 bits narrowed here would clear a conversation nobody named.
    String drop_slot(int64_t slot);

    // Hands out every signal the worker has queued, on the calling thread, in order. The
    // engine calls it deferred after each burst; a caller that draws no frames calls it.
    void deliver_pending();

    // Blocks the calling thread, draining as it waits, until the turn in flight has been
    // handed out or the wait runs out; answers whether the turn is over. For a caller with
    // no frames -- a headless test, a tool -- and never for a game, which has frames.
    bool wait_for_turn(int timeout_ms);

    // What the last turn of one slot cost, and how many tokens of it the cache holds. The
    // context size is the whole context, which is what a prompt is measured against: the slots
    // share those tokens rather than each being given a fixed part of them.
    //
    // Both take the index whole, as generate() and drop_slot() do, and refuse one the context
    // has not with the same sentence: a number GDScript holds as 64 bits narrowed here would
    // answer with the row and the cache of a conversation nobody named. Before a load there is
    // no sequence to name and both answer their empty value without a word.
    Dictionary last_timings(int64_t slot = 0) const;
    int context_size() const;
    int cached_tokens(int64_t slot = 0) const;

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

    // The card a host named for every library in the process, or empty for this one's own choice.
    std::string device_selector_;
    static bool ensure_backends();

    void remember_what_it_holds();
    // What each occupied sequence carries, for a sentence about a full context to name. The
    // count is the caller's own snapshot of `opened`, so one sentence reads one list.
    godot::String what_the_slots_hold(int count) const;
    // Clears one slot out of the cache. Called with the model lock held, and never otherwise:
    // touching the cache beside a decoding worker is what it is there to prevent.
    void clear_slot(int slot);
    // Clears every slot whose drop was asked for while a turn held the model. Called with the
    // lock held, in front of a turn and again behind it, so a seat given back during the last
    // turn of a scene is cleared without waiting for a turn that may never come.
    void drain_owed_drops(int count);
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
