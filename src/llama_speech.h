#ifndef LLAMA_SPEECH_H
#define LLAMA_SPEECH_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

namespace godot {

// One sentence as the worker receives it. Gathered on the caller's thread so the worker never
// reads a Variant, which is what keeps the engine's reference counting off that thread.
struct LlamaSpeechTurn {
    std::string text;
    std::string language;
    std::string reference;
    int32_t top_k = 0;
    float top_p = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    int max_frames = 0;
};

// What the last sentence cost. Written by the worker before the finish is delivered and read
// on the main thread after it; read during a sentence they are the sentence before.
struct LlamaSpeechTimings {
    double load_ms = 0.0;
    double speaker_ms = 0.0;
    double prompt_ms = 0.0;
    double generate_ms = 0.0;
    double vocoder_ms = 0.0;
    double total_ms = 0.0;
    double audio_ms = 0.0;
    int frames = 0;
    int samples = 0;
};

// One thing the worker has to say, kept on the object until the main thread hands it out. A
// synthesis fills samples and rate; a failure fills text with the sentence.
struct LlamaSpeechEvent {
    enum Kind { SYNTHESISED, FAILED };
    Kind kind = SYNTHESISED;
    int64_t at = 0;
    String text;
    PackedFloat32Array samples;
    int rate = 0;
};

// Speech from one local model pair: a backbone GGUF that samples the semantic codes and an
// mmproj GGUF whose code predictor and vocoder turn them into PCM. Both are in this process,
// and the whole of a sentence's sound is delivered as float samples at the codec's own rate.
//
// speak() hands the sentence to one worker thread and refuses while one runs. The worker
// queues what it has to say on the object; a deferred call drains the queue on the main
// thread, and deliver_pending() drains it for a caller that draws no frames. unload(),
// load() and the destructor wait for the worker.
class LlamaSpeech : public RefCounted {
    GDCLASS(LlamaSpeech, RefCounted)

    // The one worker, the flag that says the model is taken, and whether the worker owns that
    // flag. The flag is raised in one atomic step and lowered by the delivery of the finish on
    // the main thread, or by unload() when that delivery is never coming.
    std::thread worker;
    std::atomic<bool> busy{false};
    std::atomic<bool> owed{false};

    // What the worker has said and nobody has handed out yet, and whether a deferred drain is
    // already on its way.
    std::mutex events_lock;
    std::vector<LlamaSpeechEvent> events;
    std::atomic<bool> drain_queued{false};

    // Read once per frame by the worker; raised by cancel() and by unload().
    std::atomic<bool> stop_asked{false};

    // Which model a sentence was started against. A delivery whose epoch has moved on is
    // dropped rather than reported for a model that is gone.
    std::atomic<int64_t> epoch{0};

    // Held by every sentence and by whatever frees or replaces the model.
    std::mutex model_lock;
    std::atomic<bool> loaded{false};

    common_init_result_ptr runtime;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    common_sampler *sampler = nullptr;
    mtmd::context_ptr mctx;

    // The reference clip already through the speaker encoder's front end, and the path it was
    // read from. One clip is decoded and resampled once however many sentences it speaks.
    std::string speaker_path;
    mtmd::bitmap_ptr speaker;

    int context_tokens = 0;
    int n_batch = 512;
    int sample_rate = 0;

    // Where the backbone was put, as ggml names it, and where the codec ended up. The two
    // differ whenever the chosen backend cannot run the codec's graph. The handle beside them
    // is what the crash line reads its free memory from; null means the processor.
    std::string chosen_device;
    std::string codec_device;
    ggml_backend_dev_t device = nullptr;

    // What the library allocated for the backbone, taken once at the end of the load and kept.
    // The codec is a second runtime of its own and is not in these: a caller that wants it
    // whole adds the projector file beside them. Read outside the lock, as the timings are.
    std::atomic<int64_t> weights_bytes{0};
    std::atomic<int64_t> kv_bytes{0};
    std::atomic<int64_t> compute_bytes{0};
    std::atomic<int64_t> host_bytes{0};

    LlamaSpeechTimings timings;

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

    LlamaSpeech() = default;
    ~LlamaSpeech();

    // A folder holding exactly one backbone .gguf and one mmproj-*.gguf; an OS path or a
    // res:// or user:// one. A negative layer count puts every layer on the best GPU the
    // machine has, zero keeps the whole model on the CPU.
    bool load(const String &model_folder, int n_ctx, int n_threads, int n_gpu_layers);
    void unload();
    bool is_loaded() const;
    bool is_busy() const;

    // One sentence, in the two-letter language code the codec knows, optionally in the voice
    // of a reference clip. False when the model is not loaded, a sentence is running, or the
    // text is empty.
    bool speak(const String &text, const String &language, const String &reference);
    void cancel();

    // Hands out every signal the worker has queued, on the calling thread, in order. The
    // engine calls it deferred; a caller that draws no frames calls it.
    void deliver_pending();

    // Blocks the calling thread, draining as it waits, until the sentence in flight has been
    // handed out or the wait runs out; answers whether the sentence is over.
    bool wait_for_speech(int timeout_ms);

    Dictionary last_timings() const;
    int output_rate() const;
    int context_size() const;

    // What the backbone holds, as the library counted it when the load finished: weights_bytes,
    // kv_bytes, compute_bytes, and host_bytes for the part of the three that is ordinary memory
    // rather than the card's. The codec's own buffers are not among them -- it is a runtime of
    // its own, and llama.cpp accounts for nothing outside its context.
    Dictionary memory_report() const;

    // The free and total memory of the device this model was put on, or of the best device on
    // the machine before anything is loaded: free_bytes, total_bytes and name, with -1 for both
    // numbers where no backend will say. It is the whole card and not this process's share.
    Dictionary device_memory();

    // The ggml devices the backends found, one dictionary each, after the backends are loaded.
    Array describe_devices();

    // Whether llama.cpp's own log lines below warning level reach the console.
    static void set_verbose(bool on);

    // What a reference clip may be, answered by the half that actually decodes one so a caller
    // never keeps a second copy of the list: the container suffixes, and the test on a clip's
    // first bytes. Static -- a caller asks before there is a model to ask.
    static PackedStringArray readable_clip_formats();
    static bool is_readable_clip(const PackedByteArray &clip);

private:

    // The card a host named for every library in the process, or empty for this one's own choice.
    std::string device_selector_;
    void remember_what_it_holds();
    void warm_up();
    void work(LlamaSpeechTurn turn, int64_t at);
    void run_turn(const LlamaSpeechTurn &turn, int64_t at);
    bool take_reference(const std::string &path, String &error);
    void post(LlamaSpeechEvent event);
    void deliver_synthesised(int64_t at, const PackedFloat32Array &samples, int rate);
    void deliver_failed(int64_t at, const String &message);
    void join_worker();
};

} // namespace godot

#endif // LLAMA_SPEECH_H
