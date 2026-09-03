#ifndef LLAMA_CHAT_REGISTER_TYPES_H
#define LLAMA_CHAT_REGISTER_TYPES_H

#include <godot_cpp/core/class_db.hpp>

namespace godot {

void initialize_llama_chat_module(ModuleInitializationLevel p_level);
void uninitialize_llama_chat_module(ModuleInitializationLevel p_level);

} // namespace godot

#endif // LLAMA_CHAT_REGISTER_TYPES_H
