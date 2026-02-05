cmake_minimum_required(VERSION 3.25)

find_package(cmake-bare REQUIRED PATHS node_modules/cmake-bare)
find_package(cmake-napi REQUIRED PATHS node_modules/cmake-napi)

project(bare_llama C CXX)

# Build llama.cpp as a static library
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_HTTPLIB OFF CACHE BOOL "" FORCE)

# llguidance requires Rust/Cargo and cannot cross-compile,
# so disable it when the target architecture differs from the host
if(CMAKE_CROSSCOMPILING)
  set(LLAMA_LLGUIDANCE OFF CACHE BOOL "" FORCE)
else()
  set(LLAMA_LLGUIDANCE ON CACHE BOOL "" FORCE)
endif()

if(ANDROID)
  set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(vendor/llama.cpp)

# Create the bare addon
add_bare_module(bare_llama_bare)

target_sources(
  ${bare_llama_bare}
  PRIVATE
    binding.cpp
)

target_include_directories(
  ${bare_llama_bare}
  PRIVATE
    vendor/llama.cpp/include
    vendor/llama.cpp/ggml/include
    vendor/llama.cpp/common
)

target_link_libraries(
  ${bare_llama_bare}
  PRIVATE
    llama
    common
)

# Create the Node.js NAPI addon
add_napi_module(bare_llama_node)

target_sources(
  ${bare_llama_node}
  PRIVATE
    binding.cpp
)

target_compile_definitions(
  ${bare_llama_node}
  PRIVATE
    NAPI_VERSION=9
)

target_include_directories(
  ${bare_llama_node}
  PRIVATE
    vendor/llama.cpp/include
    vendor/llama.cpp/ggml/include
    vendor/llama.cpp/common
)

target_link_libraries(
  ${bare_llama_node}
  PRIVATE
    llama
    common
)

resolve_node_module(bare-compat-napi compat)

target_include_directories(
  ${bare_llama_node}
  PRIVATE
    "${compat}/include"
)
