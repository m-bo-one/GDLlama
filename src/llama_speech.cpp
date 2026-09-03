#include "llama_speech.h"

#include "llama_runtime.h"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "ggml-backend.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>

// The same fence llama_chat.cpp carries: the worker thread is wrapped in a try/catch, and
// compiled without exceptions that is dead code and a fault unwinds into the engine.
#if !defined(_CPPUNWIND) && !defined(__EXCEPTIONS) && !defined(__cpp_exceptions)
#error "llama_speech.cpp needs C++ exceptions: configure with GODOTCPP_DISABLE_EXCEPTIONS=OFF"
#endif

using namespace godot;
using llama_runtime::to_gd;
using llama_runtime::to_std;

namespace {

using clock_type = std::chrono::steady_clock;

// How many frames one sentence may become before the loop gives up. At twelve frames a second
// this is a minute of speech, which no sentence is; a model that will not stop is cut here
// rather than filling memory.
constexpr int MAX_FRAMES = 720;

// The one sequence the context holds, and the sampling the report measured the model with.
constexpr llama_seq_id SEQUENCE = 0;
constexpr int32_t DEFAULT_TOP_K = 50;
constexpr float DEFAULT_TOP_P = 1.0f;

double ms_between(clock_type::time_point from, clock_type::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// Takes the busy flag in one atomic step or reports that another path holds it, and gives it
// back unless the sentence was handed on to a worker.
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

    void hand_on() { held = nullptr; }
};

LlamaSpeechEvent synthesised_event(int64_t at, PackedFloat32Array samples, int rate) {
    LlamaSpeechEvent event;
    event.kind = LlamaSpeechEvent::SYNTHESISED;
    event.at = at;
    event.samples = std::move(samples);
    event.rate = rate;
    return event;
}

LlamaSpeechEvent failed_event(int64_t at, const String &message) {
    LlamaSpeechEvent event;
    event.kind = LlamaSpeechEvent::FAILED;
    event.at = at;
    event.text = message;
    return event;
}

} // namespace

LlamaSpeech::~LlamaSpeech() {
    unload();
}

// The two GGUFs of one folder: the backbone, and the mmproj holding the code predictor and
// the vocoder. Named apart by the mmproj prefix llama.cpp's own conversion writes, so a
// folder with anything else in it is refused rather than guessed at.
static bool split_pair(const String &folder, String &backbone, String &mmproj, String &error) {
    Ref<DirAccess> dir = DirAccess::open(folder);
    if (dir.is_null()) {
        error = "the folder \"" + folder + "\" could not be opened";
        return false;
    }
    PackedStringArray backbones;
    PackedStringArray projectors;
    const PackedStringArray names = dir->get_files();
    for (int64_t i = 0; i < names.size(); i++) {
        const String name = names[i];
        const String lower = name.to_lower();
        if (!lower.ends_with(".gguf")) {
            continue;
        }
        if (lower.begins_with("mmproj")) {
            projectors.push_back(name);
        } else {
            backbones.push_back(name);
        }
    }
    if (backbones.size() != 1 || projectors.size() != 1) {
        error = "the folder \"" + folder + "\" holds " + String::num_int64(backbones.size()) +
                " backbone and " + String::num_int64(projectors.size()) +
                " mmproj-*.gguf files, and it has to hold exactly one of each";
        return false;
    }
    backbone = folder.path_join(backbones[0]);
    mmproj = folder.path_join(projectors[0]);
    return true;
}

// The backbone, its context and the mmproj beside it. A negative layer count puts the backbone
// on the best GPU the machine has; zero keeps everything on the CPU. The mmproj follows the
// backbone onto the GPU only where its graph runs there, which today means CUDA alone.
bool LlamaSpeech::load(const String &model_folder, int n_ctx, int n_threads, int n_gpu_layers) {
    unload();
    llama_runtime::note_operation("LlamaSpeech was loading a speech model");
    if (!llama_runtime::ensure_backends()) {
        return false;
    }
    const auto started = clock_type::now();

    String folder = model_folder;
    if (folder.begins_with("res://") || folder.begins_with("user://")) {
        folder = ProjectSettings::get_singleton()->globalize_path(folder);
    }
    String backbone_path;
    String mmproj_path;
    String problem;
    if (!split_pair(folder, backbone_path, mmproj_path, problem)) {
        UtilityFunctions::push_error("LlamaSpeech: " + problem + ".");
        return false;
    }

    common_params params;
    params.model.path = to_std(backbone_path);
    params.n_ctx = n_ctx > 0 ? n_ctx : 1024;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_gpu_layers = n_gpu_layers < 0 ? -1 : n_gpu_layers;
    params.fit_params = false;
    params.warmup = false;
    // The pipeline reads the backbone's hidden state for every frame, not its logits alone.
    params.embedding = true;
    params.sampling.top_k = DEFAULT_TOP_K;
    params.sampling.top_p = DEFAULT_TOP_P;
    const int threads = n_threads > 0 ? n_threads : std::max(1, (int)std::thread::hardware_concurrency() / 2);
    params.cpuparams.n_threads = threads;
    params.cpuparams_batch.n_threads = threads;

    chosen_device = "cpu";
    codec_device = "cpu";
    params.devices = { nullptr };
    ggml_backend_dev_t best = nullptr;
    if (params.n_gpu_layers != 0) {
        best = llama_runtime::best_device();
        if (best != nullptr) {
            params.devices = { best, nullptr };
            // Read the file rather than map it, for the reason llama_chat.cpp gives: mapped,
            // the whole file stays in the process's working set on Windows.
            params.load_mode = LLAMA_LOAD_MODE_NONE;
            chosen_device = llama_runtime::describe_device(best);
        } else {
            params.n_gpu_layers = 0;
        }
    }

    std::lock_guard<std::mutex> hold(model_lock);
    try {
        runtime = common_init_from_params(params);
    } catch (const std::exception &e) {
        UtilityFunctions::push_error(String("LlamaSpeech: loading failed: ") + String::utf8(e.what()));
        runtime.reset();
        return false;
    }
    if (!runtime || runtime->model() == nullptr || runtime->context() == nullptr) {
        UtilityFunctions::push_error("LlamaSpeech: the model at \"" + backbone_path + "\" could not be loaded.");
        runtime.reset();
        return false;
    }
    model = runtime->model();
    ctx = runtime->context();
    vocab = llama_model_get_vocab(model);
    sampler = runtime->sampler(SEQUENCE);
    if (sampler == nullptr) {
        UtilityFunctions::push_error("LlamaSpeech: the backbone gave no sampler.");
        runtime.reset();
        model = nullptr;
        ctx = nullptr;
        vocab = nullptr;
        return false;
    }

    // mtmd keeps a logger of its own, and left alone it writes a line per encoded chunk into
    // the game's console. Pointed at the same filter llama.cpp's lines go through.
    mtmd_helper_log_set(llama_runtime::log_callback(), nullptr);

    // Only the CUDA backend runs the codec's graph: on Vulkan it dies inside GET_ROWS, so the
    // mmproj stays on the CPU there and the device word says so. Changing this without
    // measuring the assert again is how a game starts crashing on somebody else's card.
    const bool codec_on_gpu = best != nullptr && llama_runtime::backend_name_of(best) == "CUDA";
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = codec_on_gpu;
    mtmd_params.device = codec_on_gpu ? best : nullptr;
    mtmd_params.n_threads = threads;
    mtmd_params.print_timings = false;
    mtmd_params.warmup = true;
    // The codec's own load, warm-up included: the one step whose graph is known to abort on a
    // backend that cannot run it, so the breadcrumb names it before it is entered.
    llama_runtime::note_operation("LlamaSpeech was loading the speech codec");
    mctx.reset(mtmd_init_from_file(to_std(mmproj_path).c_str(), model, mtmd_params));
    if (!mctx) {
        UtilityFunctions::push_error("LlamaSpeech: the mmproj at \"" + mmproj_path + "\" could not be loaded.");
        runtime.reset();
        model = nullptr;
        ctx = nullptr;
        vocab = nullptr;
        sampler = nullptr;
        return false;
    }
    const mtmd_gen_audio_info info = mtmd_gen_audio_get_info(mctx.get());
    if (info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
        UtilityFunctions::push_error(
                "LlamaSpeech: the mmproj at \"" + mmproj_path + "\" carries no audio generation head.");
        mctx.reset();
        runtime.reset();
        model = nullptr;
        ctx = nullptr;
        vocab = nullptr;
        sampler = nullptr;
        return false;
    }
    sample_rate = info.sample_rate;
    codec_device = codec_on_gpu ? chosen_device : std::string("cpu");

    context_tokens = (int)llama_n_ctx(ctx);
    llama_runtime::note_operation("LlamaSpeech was warming the model up");
    warm_up();
    timings = LlamaSpeechTimings();
    timings.load_ms = ms_between(started, clock_type::now());
    loaded.store(true);
    llama_runtime::note_operation("LlamaSpeech was waiting for something to say");
    return true;
}

// A backend builds its kernels for a batch shape the first time it meets one, which on Vulkan
// is seconds of pipeline compilation. The widths a sentence produces are decoded once here so
// the cost sits in load_ms rather than in the first sentence.
void LlamaSpeech::warm_up() {
    llama_token filler = llama_vocab_bos(vocab);
    if (filler == LLAMA_TOKEN_NULL) {
        filler = llama_vocab_eos(vocab);
    }
    if (filler == LLAMA_TOKEN_NULL) {
        filler = 0;
    }
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (const int width : { std::min(n_batch, context_tokens - 1), 32, 1 }) {
        if (width <= 0) {
            continue;
        }
        common_batch_clear(batch);
        for (int i = 0; i < width; i++) {
            common_batch_add(batch, filler, (llama_pos)i, { SEQUENCE }, i + 1 == width);
        }
        llama_decode(ctx, batch);
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    llama_batch_free(batch);
    llama_synchronize(ctx);
}

// The worker is stopped and joined and then the lock is taken, in that order: the worker's
// sentence holds the lock, so taking it first would wait on a thread waiting to be joined.
void LlamaSpeech::unload() {
    stop_asked.store(true);
    epoch.fetch_add(1);
    join_worker();
    {
        std::lock_guard<std::mutex> hold(model_lock);
        loaded.store(false);
        speaker.reset();
        speaker_path.clear();
        mctx.reset();
        runtime.reset();
        model = nullptr;
        ctx = nullptr;
        vocab = nullptr;
        sampler = nullptr;
    }
    {
        std::lock_guard<std::mutex> hold(events_lock);
        events.clear();
    }
    if (owed.exchange(false)) {
        busy.store(false);
    }
    stop_asked.store(false);
}

bool LlamaSpeech::is_loaded() const {
    return loaded.load();
}

bool LlamaSpeech::is_busy() const {
    return busy.load();
}

bool LlamaSpeech::speak(const String &text, const String &language, const String &reference) {
    BusyGuard guard(busy);
    if (!guard.taken()) {
        UtilityFunctions::push_error("LlamaSpeech: a sentence is still being made; wait for it or cancel it.");
        return false;
    }
    if (!loaded.load()) {
        UtilityFunctions::push_error("LlamaSpeech: no model is loaded.");
        return false;
    }
    const String spoken = text.strip_edges();
    if (spoken.is_empty()) {
        UtilityFunctions::push_error("LlamaSpeech: there is nothing to say.");
        return false;
    }

    LlamaSpeechTurn turn;
    turn.text = to_std(spoken);
    turn.language = to_std(language.strip_edges());
    turn.reference = to_std(reference.strip_edges());
    turn.top_k = DEFAULT_TOP_K;
    turn.top_p = DEFAULT_TOP_P;
    turn.max_frames = MAX_FRAMES;

    join_worker();
    stop_asked.store(false);
    owed.store(true);
    try {
        worker = std::thread(&LlamaSpeech::work, this, std::move(turn), epoch.load());
    } catch (...) {
        owed.store(false);
        UtilityFunctions::push_error("LlamaSpeech: the worker thread could not be started.");
        return false;
    }
    guard.hand_on();
    return true;
}

// Never blocks: the flag is read by the worker before every frame, and what has been made so
// far is thrown away rather than delivered as a sentence nobody will hear the end of.
void LlamaSpeech::cancel() {
    if (busy.load()) {
        stop_asked.store(true);
    }
}

void LlamaSpeech::post(LlamaSpeechEvent event) {
    {
        std::lock_guard<std::mutex> hold(events_lock);
        events.push_back(std::move(event));
    }
    if (!drain_queued.exchange(true)) {
        callable_mp(this, &LlamaSpeech::deliver_pending).call_deferred();
    }
}

void LlamaSpeech::deliver_pending() {
    drain_queued.store(false);
    std::vector<LlamaSpeechEvent> batch;
    {
        std::lock_guard<std::mutex> hold(events_lock);
        batch.swap(events);
    }
    for (const LlamaSpeechEvent &event : batch) {
        switch (event.kind) {
            case LlamaSpeechEvent::SYNTHESISED:
                deliver_synthesised(event.at, event.samples, event.rate);
                break;
            case LlamaSpeechEvent::FAILED:
                deliver_failed(event.at, event.text);
                break;
        }
    }
}

bool LlamaSpeech::wait_for_speech(int timeout_ms) {
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

Dictionary LlamaSpeech::last_timings() const {
    Dictionary out;
    out["load_ms"] = timings.load_ms;
    out["speaker_ms"] = timings.speaker_ms;
    out["prompt_ms"] = timings.prompt_ms;
    out["generate_ms"] = timings.generate_ms;
    out["vocoder_ms"] = timings.vocoder_ms;
    out["total_ms"] = timings.total_ms;
    out["audio_ms"] = timings.audio_ms;
    out["frames"] = timings.frames;
    out["samples"] = timings.samples;
    // How long the sentence took against how long it sounds. Below one is faster than real
    // time, which is the number a game is bought or lost on.
    out["real_time_factor"] = timings.audio_ms > 0.0 ? timings.total_ms / timings.audio_ms : 0.0;
    out["device"] = to_gd(chosen_device);
    out["codec_device"] = to_gd(codec_device);
    out["sample_rate"] = sample_rate;
    out["context_size"] = context_tokens;
    return out;
}

int LlamaSpeech::output_rate() const {
    return sample_rate;
}

int LlamaSpeech::context_size() const {
    return context_tokens;
}

Array LlamaSpeech::describe_devices() {
    return llama_runtime::describe_devices();
}

void LlamaSpeech::set_verbose(bool on) {
    llama_runtime::set_verbose(on);
}

bool LlamaSpeech::load_backend_folder(const String &folder) {
    String path = folder;
    if (path.begins_with("res://") || path.begins_with("user://")) {
        path = ProjectSettings::get_singleton()->globalize_path(path);
    }
    if (path.is_empty() || !DirAccess::dir_exists_absolute(path)) {
        return false;
    }
    return llama_runtime::load_backend_folder(to_std(path));
}

// The reference clip decoded, resampled and kept as the bitmap the speaker encoder reads. One
// path is read once however many sentences it speaks: the decode is milliseconds, but a clip
// re-read per sentence is a file opened in the middle of a conversation.
bool LlamaSpeech::take_reference(const std::string &path, String &error) {
    if (path.empty()) {
        speaker.reset();
        speaker_path.clear();
        return true;
    }
    if (speaker && speaker_path == path) {
        return true;
    }
    mtmd_helper_bitmap_wrapper wrapper =
            mtmd_helper_bitmap_init_from_file(mctx.get(), path.c_str(), false, mtmd_helper_init_opt_default());
    if (wrapper.bitmap == nullptr) {
        error = "the reference clip \"" + to_gd(path) + "\" could not be read as audio";
        return false;
    }
    speaker.reset(wrapper.bitmap);
    speaker_path = path;
    return true;
}

// The worker's whole life. An exception out of the sentence is turned into a failure delivered
// on the main thread, where it would otherwise reach the engine and end the process. The
// breadcrumb is what a fault too hard to catch leaves behind instead.
void LlamaSpeech::work(LlamaSpeechTurn turn, int64_t at) {
    llama_runtime::note_operation("LlamaSpeech was making a sentence");
    try {
        run_turn(turn, at);
    } catch (const std::exception &e) {
        post(failed_event(at,
                String("LlamaSpeech: making the sentence failed: ") + String::utf8(e.what())));
    } catch (...) {
        post(failed_event(at, String("LlamaSpeech: making the sentence failed with an error "
                                     "that carries no words. The editor log carries whatever "
                                     "the library wrote before it.")));
    }
    llama_runtime::note_operation("LlamaSpeech was waiting for something to say");
}

// One sentence: the prompt through the backbone, then a frame at a time until the model says
// it has finished speaking, then the accumulated PCM handed over whole.
void LlamaSpeech::run_turn(const LlamaSpeechTurn &turn, int64_t at) {
    std::lock_guard<std::mutex> hold(model_lock);
    if (at != epoch.load() || !loaded.load()) {
        post(failed_event(at, "LlamaSpeech: the model was given back before the sentence ran."));
        return;
    }

    const auto started = clock_type::now();
    String problem;
    if (!take_reference(turn.reference, problem)) {
        post(failed_event(at, "LlamaSpeech: " + problem + "."));
        return;
    }
    const auto after_speaker = clock_type::now();

    // Every sentence starts from an empty context: the prompt is an embedding stream built
    // around this text, so there is no prefix of it that a later sentence could reuse.
    llama_memory_clear(llama_get_memory(ctx), true);
    common_sampler_reset(sampler);

    mtmd_helper::gen_audio gen(ctx, mctx.get());
    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id = SEQUENCE;
    inp.prompt = turn.text.c_str();
    inp.prompt_len = turn.text.size();
    inp.speaker_ref = speaker.get();
    inp.lang = turn.language.empty() ? nullptr : turn.language.c_str();
    inp.top_k = turn.top_k;
    inp.top_p = turn.top_p;
    inp.seed = turn.seed;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

    if (gen.set_input(&inp) != 0) {
        post(failed_event(at, "LlamaSpeech: the sentence could not be prepared. A language the codec does not "
                              "know, or a reference clip it could not encode, both end here."));
        return;
    }

    for (;;) {
        if (stop_asked.load()) {
            return;
        }
        const int32_t left = gen.step_prompt(n_batch);
        if (left < 0) {
            post(failed_event(at, "LlamaSpeech: the sentence's prompt could not be read by the model."));
            return;
        }
        if (left == 0) {
            break;
        }
    }
    const auto after_prompt = clock_type::now();

    llama_token sampled = common_sampler_sample(sampler, ctx, -1);
    common_sampler_accept(sampler, sampled, true);
    const float *h_state = llama_get_embeddings_ith(ctx, -1);

    int frames = 0;
    bool stop = false;
    while (!stop && frames < turn.max_frames) {
        if (stop_asked.load()) {
            return;
        }
        const float *h_next = nullptr;
        if (gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
            post(failed_event(at, "LlamaSpeech: the sentence broke off while its sound was being made."));
            return;
        }
        if (h_next == nullptr) {
            break;
        }
        frames++;
        h_state = h_next;
        sampled = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, sampled, true);
    }
    const auto after_frames = clock_type::now();

    int32_t rate = 0;
    const char *data = nullptr;
    size_t data_len = 0;
    int64_t made = 0;
    if (gen.get_output(&rate, &data, &data_len, &made) != 0) {
        post(failed_event(at, "LlamaSpeech: the sound of the sentence could not be read back."));
        return;
    }
    if (stop_asked.load()) {
        return;
    }

    PackedFloat32Array samples;
    if (made > 0 && data != nullptr) {
        samples.resize((int64_t)made);
        memcpy(samples.ptrw(), data, (size_t)made * sizeof(float));
    }

    const auto ended = clock_type::now();
    timings.speaker_ms = ms_between(started, after_speaker);
    timings.prompt_ms = ms_between(after_speaker, after_prompt);
    timings.generate_ms = ms_between(after_prompt, after_frames);
    timings.vocoder_ms = ms_between(after_frames, ended);
    timings.total_ms = ms_between(started, ended);
    timings.frames = frames;
    timings.samples = (int)made;
    timings.audio_ms = rate > 0 ? (double)made * 1000.0 / (double)rate : 0.0;

    post(synthesised_event(at, std::move(samples), rate));
}

// The deliveries, on the main thread. A sentence whose model was unloaded or replaced while it
// ran is dropped: the flag it would clear belongs to whatever was started after it.
void LlamaSpeech::deliver_synthesised(int64_t at, const PackedFloat32Array &samples, int rate) {
    if (at != epoch.load() || !busy.load()) {
        return;
    }
    owed.store(false);
    busy.store(false);
    emit_signal("synthesised", samples, rate);
}

void LlamaSpeech::deliver_failed(int64_t at, const String &message) {
    if (at != epoch.load() || !busy.load()) {
        return;
    }
    owed.store(false);
    busy.store(false);
    emit_signal("failed", message);
}

void LlamaSpeech::join_worker() {
    if (worker.joinable()) {
        worker.join();
    }
}

void LlamaSpeech::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load", "model_folder", "n_ctx", "n_threads", "n_gpu_layers"), &LlamaSpeech::load);
    ClassDB::bind_method(D_METHOD("unload"), &LlamaSpeech::unload);
    ClassDB::bind_method(D_METHOD("is_loaded"), &LlamaSpeech::is_loaded);
    ClassDB::bind_method(D_METHOD("is_busy"), &LlamaSpeech::is_busy);
    ClassDB::bind_method(D_METHOD("speak", "text", "language", "reference"), &LlamaSpeech::speak);
    ClassDB::bind_method(D_METHOD("cancel"), &LlamaSpeech::cancel);
    ClassDB::bind_method(D_METHOD("deliver_pending"), &LlamaSpeech::deliver_pending);
    ClassDB::bind_method(D_METHOD("wait_for_speech", "timeout_ms"), &LlamaSpeech::wait_for_speech);
    ClassDB::bind_method(D_METHOD("last_timings"), &LlamaSpeech::last_timings);
    ClassDB::bind_method(D_METHOD("output_rate"), &LlamaSpeech::output_rate);
    ClassDB::bind_method(D_METHOD("context_size"), &LlamaSpeech::context_size);
    ClassDB::bind_method(D_METHOD("describe_devices"), &LlamaSpeech::describe_devices);
    ClassDB::bind_static_method("LlamaSpeech", D_METHOD("set_verbose", "on"), &LlamaSpeech::set_verbose);
    ClassDB::bind_static_method(
            "LlamaSpeech", D_METHOD("load_backend_folder", "folder"), &LlamaSpeech::load_backend_folder);

    ADD_SIGNAL(MethodInfo("synthesised", PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "samples"),
            PropertyInfo(Variant::INT, "rate")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::STRING, "message")));
}
