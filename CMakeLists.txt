# The GDExtension around llama.cpp: one shared library, LlamaChat, beside the llama.cpp and
# ggml shared libraries it links, with every ggml backend built as its own library and picked
# at run time. Vulkan is not built here: its shaders need the Vulkan SDK, and the prebuilt
# ggml-vulkan of the same llama.cpp tag is dropped beside the others instead (see README.md).
cmake_minimum_required(VERSION 3.22)
project(llama_chat C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# One C runtime, dynamically linked, shared with the prebuilt Vulkan backend and with the
# rest of the llama.cpp release binaries. A static runtime here would give ggml two heaps.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if (MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
endif()

# ---------------------------------------------------------------------------------------------
# godot-cpp, static. Exceptions stay on: the worker thread fences llama.cpp with try/catch,
# and the chat template code throws on a malformed message.
# ---------------------------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF)
set(GODOTCPP_TARGET "template_debug" CACHE STRING "template_debug or template_release")
set(GODOTCPP_DISABLE_EXCEPTIONS OFF CACHE BOOL "" FORCE)
add_subdirectory(godot-cpp)

# ---------------------------------------------------------------------------------------------
# llama.cpp, shared, with dynamically loaded backends and every CPU variant, so that the same
# build runs on any x86-64 and picks the widest instruction set the machine has.
# ---------------------------------------------------------------------------------------------
set(BUILD_SHARED_LIBS ON)
set(GGML_BACKEND_DL ON)
set(GGML_CPU_ALL_VARIANTS ON)
set(GGML_NATIVE OFF)
set(GGML_OPENMP OFF)
set(GGML_VULKAN OFF)
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_APP OFF)
set(LLAMA_OPENSSL OFF)
set(LLAMA_SUBPROCESS OFF)
set(LLAMA_ALL_WARNINGS OFF)
add_subdirectory(llama.cpp)

# ---------------------------------------------------------------------------------------------
# The extension
# ---------------------------------------------------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_NAME}" LLAMA_CHAT_PLATFORM)
if (LLAMA_CHAT_PLATFORM STREQUAL "darwin")
  set(LLAMA_CHAT_PLATFORM "macos")
endif()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(LLAMA_CHAT_ARCH "x86_64")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(LLAMA_CHAT_ARCH "arm64")
else()
  set(LLAMA_CHAT_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_library(llama_chat SHARED
  src/llama_chat.cpp
  src/llama_runtime.cpp
  src/register_types.cpp
)
target_link_libraries(llama_chat PRIVATE godot::cpp llama-common llama)
set_target_properties(llama_chat PROPERTIES
  PREFIX ""
  OUTPUT_NAME "libllama_chat.${LLAMA_CHAT_PLATFORM}.${GODOTCPP_TARGET}.${LLAMA_CHAT_ARCH}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
if (APPLE)
  set_target_properties(llama_chat PROPERTIES INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH TRUE)
elseif (UNIX)
  set_target_properties(llama_chat PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# ---------------------------------------------------------------------------------------------
# Install: the folder a Godot project takes as addons/llm. Everything the extension loads at
# run time lands in bin/, the prebuilt Vulkan backend included when its path is given.
# ---------------------------------------------------------------------------------------------
set(LLAMA_CHAT_VULKAN_DLL "" CACHE FILEPATH "prebuilt ggml-vulkan shared library to ship beside the others")
set(LLAMA_CHAT_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install/addons/llm")

# llama.cpp installs its own headers and libraries under the prefix whatever the caller wants;
# pointed into the build tree so that the only install that reaches the source tree is the
# addon folder below, whose destinations are absolute.
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install-prefix" CACHE PATH "" FORCE)

install(TARGETS llama_chat
  RUNTIME DESTINATION "${LLAMA_CHAT_INSTALL_DIR}/bin"
  LIBRARY DESTINATION "${LLAMA_CHAT_INSTALL_DIR}/bin"
)
install(DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
  DESTINATION "${LLAMA_CHAT_INSTALL_DIR}/bin"
  FILES_MATCHING
    PATTERN "ggml*${CMAKE_SHARED_LIBRARY_SUFFIX}"
    PATTERN "llama*${CMAKE_SHARED_LIBRARY_SUFFIX}"
    PATTERN "libggml*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    PATTERN "libllama*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
)
if (LLAMA_CHAT_VULKAN_DLL)
  install(FILES "${LLAMA_CHAT_VULKAN_DLL}" DESTINATION "${LLAMA_CHAT_INSTALL_DIR}/bin")
endif()
install(FILES "plugin/llama_chat.gdextension" DESTINATION "${LLAMA_CHAT_INSTALL_DIR}")
