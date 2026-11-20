cmake_minimum_required(VERSION 3.18)
project(Deco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

option(DECO_ENABLE_CUDA "Build CUDA expander decomposition" ON)
option(DECO_BUILD_CLI "Build command-line executable" ON)
option(DECO_BUILD_TESTS "Build tests" ON)

add_subdirectory(lib)

if(DECO_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "--std=c++20")
    set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "--std=c++20")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_subdirectory(lib/algorithms/expander_decomposition/cuda/internal)
endif()

if(DECO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DECO_BUILD_CLI)
    add_executable(Deco cli/deco.cpp)

    target_compile_options(Deco PRIVATE
            $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:
            -O3
            -march=native
            -mtune=native
            -ffast-math
            >
            $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:
            /O2
            /Ob2
            /fp:fast
            >
    )

    target_link_libraries(Deco PRIVATE DecoLib)

    if(DECO_ENABLE_CUDA)
        target_link_libraries(Deco PRIVATE DecoExpanderDecompositionCuda)
    endif()
endif()
