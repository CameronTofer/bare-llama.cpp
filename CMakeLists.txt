cmake_minimum_required(VERSION 3.25)

find_package(cmake-bare REQUIRED PATHS node_modules/cmake-bare)

project(bare_llama C CXX)

# Build llama.cpp as a static library
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_LLGUIDANCE ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

if(ANDROID)
  set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(vendor/llama.cpp)

# Create the bare addon
add_bare_module(llama)

target_sources(
  ${llama}
  PRIVATE
    binding.cpp
)

target_include_directories(
  ${llama}
  PRIVATE
    vendor/llama.cpp/include
    vendor/llama.cpp/ggml/include
    vendor/llama.cpp/common
)

target_link_libraries(
  ${llama}
  PRIVATE
    llama
    common
)
