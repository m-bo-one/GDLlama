# LlamaChat

A Godot 4.5+ GDExtension that runs a chat with one local GGUF model through llama.cpp, in
the process, on a worker thread. It exposes one class, `LlamaChat`, whose job is the part a
chat client cannot do from GDScript: render the history and the tool declarations with the
model's own chat template, constrain a call with the template's grammar, separate what the
model says from what it calls while the tokens stream, and keep the context between turns so
that a turn decodes only what changed.

The lineage is the GDLlama / godot-llm extension; nothing of that node remains. The build
skeleton is what was kept.

## What is pinned

| Submodule | Where | Commit |
|---|---|---|
| `llama.cpp` | `llama.cpp/` | tag **b10786** (2026-09-03) |
| `godot-cpp` | `godot-cpp/` | branch **4.5**, `27d9dd23c83871e0619fca5dc2cddfbfd69e926a` (2026-08-25) |

`compatibility_minimum` in the `.gdextension` is 4.5; a 4.5-built extension loads in 4.7.

## The class

```gdscript
var chat := LlamaChat.new()
chat.load("res://models/Qwen3-4B-Q4_K_M.gguf", 8192, 8, -1)   # path, n_ctx, threads, gpu layers (-1: all)

chat.piece_arrived.connect(func(text: String) -> void: print(text))
chat.tool_called.connect(func(id: String, name: String, arguments_json: String) -> void: ...)
chat.finished.connect(func(reason: String, prompt_tokens: int, completion_tokens: int) -> void: ...)
chat.failed.connect(func(message: String) -> void: push_error(message))

chat.generate(messages, tools, {"temperature": 0.7, "top_p": 0.8, "max_tokens": 256,
		"enable_thinking": false})
```

- `load(model_path, n_ctx, n_threads, n_gpu_layers) -> bool` — an OS path or a `res://` /
  `user://` one. `n_gpu_layers = -1` puts every layer on the largest GPU, `0` keeps the model
  on the CPU. One `llama_context` lives as long as the model is loaded. The KV cache is f16.
- `generate(messages, tools, options) -> bool` — messages in the OpenAI chat shape (`role`,
  `content`; `tool_calls` on an assistant turn; `tool_call_id` on a `tool` turn), tools as
  OpenAI function declarations, options `temperature`, `top_p`, `top_k`, `min_p`,
  `max_tokens`, `seed`, `enable_thinking`, `parallel_tool_calls`, `json_schema`. Refuses
  while a turn runs.
- `cancel()` — never blocks; the running turn ends with `finished("cancelled", ...)`.
- `unload()`, `is_loaded()`, `is_busy()`, `context_size()`, `cached_tokens()`,
  `last_timings()`, `describe_devices()`, `LlamaChat.set_verbose(on)`.
- Signals, all on the main thread: `piece_arrived(text)` — visible text only, whole UTF-8
  letters; `tool_called(id, name, arguments_json)` — after the reply ended cleanly;
  `finished(reason, prompt_tokens, completion_tokens)` with `stop`, `length`, `cancelled` or
  `tool_calls`; `failed(message)`.

`last_timings()` answers `load_ms`, `prompt_ms`, `first_piece_ms`, `generate_ms`,
`total_ms`, `prompt_tokens`, `reused_tokens` (the prefix already in the context),
`decoded_tokens` (what this turn actually decoded), `completion_tokens`,
`tokens_per_second`, `prompt_tokens_per_second`, `device`, `context_size`.

## Building on Windows (MSVC + Ninja)

Prerequisites: Visual Studio 2022 Build Tools with the C++ workload, CMake 3.22+, Ninja
(the copy inside the Build Tools works: pass it as `-DCMAKE_MAKE_PROGRAM=...`), Python 3
for godot-cpp's binding generator. No Vulkan SDK.

From a `vcvars64` prompt in the repository root, submodules initialised:

```console
cmake --preset windows-msvc-debug -DLLAMA_CHAT_VULKAN_DLL=<path to ggml-vulkan.dll>
cmake --build --preset windows-msvc-debug
cmake --install build/windows-msvc-debug
```

and the same with `windows-msvc-release` for the release library. Both are optimised
builds; the preset picks godot-cpp's `template_debug` or `template_release` API, which is
what the editor and an exported game respectively load.

`cmake --install` writes `install/addons/llm/`: the `.gdextension`, and in `bin/` the
extension library, `llama.dll`, `llama-common.dll`, `ggml.dll`, `ggml-base.dll`, the
`ggml-cpu-*.dll` variants ggml picks from at run time, and — when
`LLAMA_CHAT_VULKAN_DLL` was given — `ggml-vulkan.dll`.

### Why the Vulkan backend is prebuilt

ggml's Vulkan backend compiles its shaders at build time with `glslc` from the Vulkan SDK.
Instead of requiring the SDK, this build enables ggml's dynamically loaded backends
(`GGML_BACKEND_DL`) so every backend is a separate library found in the extension's folder
at run time, and takes `ggml-vulkan.dll` from llama.cpp's own release archive of the **same
tag** (`llama-b10786-bin-win-vulkan-x64.zip`). That library imports only `ggml-base.dll` and
the C runtime, both of which this build provides at the same tag; the extension itself and
the core libraries import nothing from Vulkan, so a machine without a Vulkan driver falls
back to the CPU variants. When the tag moves, the archive moves with it.

## Licence

MIT, see `LICENSE`. llama.cpp and ggml are MIT (The ggml authors); godot-cpp is MIT (Godot
Engine contributors). A shipped build carries their notices.
