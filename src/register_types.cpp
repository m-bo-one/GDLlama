#include "register_types.h"

#include "llama_chat.h"
#include "llama_runtime.h"
#include "llama_speech.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include "llama.h"

namespace godot {

// The classes are registered and nothing else happens here: the ggml backends are opened by
// the first load(), so a project that never loads a model never pays for a Vulkan instance.
void initialize_llama_chat_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    // Before either class exists: a fault while a model is loading is as silent as one during
    // a turn, and the handlers cost nothing until something goes wrong.
    llama_runtime::install_crash_reporting();
    GDREGISTER_CLASS(LlamaChat);
    GDREGISTER_CLASS(LlamaSpeech);
}

void uninitialize_llama_chat_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    LlamaChat::shutdown_backends();
}

extern "C" {

GDExtensionBool GDE_EXPORT llama_chat_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
    godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(initialize_llama_chat_module);
    init_obj.register_terminator(uninitialize_llama_chat_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}

} // extern "C"

} // namespace godot
