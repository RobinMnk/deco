cmake_minimum_required(VERSION 3.18)
project(Deco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

option(DECO_ENABLE_CUDA "Build CUDA expander decomposition" OFF)
option(DECO_BUILD_CLI "Build command-line executable" ON)
option(DECO_BUILD_TESTS "Build tests" OFF)

add_subdirectory(lib)

if(DECO_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_subdirectory(src/algorithms/expander_decomposition/cuda)
endif()

if(DECO_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(DECO_BUILD_CLI)
    add_executable(Deco cli/deco.cpp)
    target_link_libraries(Deco PRIVATE DecoLib)

    if(DECO_ENABLE_CUDA)
        target_link_libraries(Deco PRIVATE DecoExpanderDecompositionCuda)
        target_compile_definitions(Deco PRIVATE DECO_ENABLE_CUDA)
    endif()
endif()
