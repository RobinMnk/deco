cmake_minimum_required(VERSION 3.18)
project(Deco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# The CUDA backend is built if a working CUDA compiler is found
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(DECO_CUDA_DEFAULT ON)
else()
    set(DECO_CUDA_DEFAULT OFF)
endif()

# The command-line tool and the tests are built if Deco is the top-level project, not when another project includes it
# (PROJECT_IS_TOP_LEVEL needs CMake 3.21)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(DECO_TOP_LEVEL ON)
else()
    set(DECO_TOP_LEVEL OFF)
endif()

option(DECO_ENABLE_CUDA "Build the CUDA backend" ${DECO_CUDA_DEFAULT})
option(DECO_BUILD_CLI "Build the command-line tool" ${DECO_TOP_LEVEL})
option(DECO_BUILD_TESTS "Build the tests" ${DECO_TOP_LEVEL})
message(STATUS "Deco: CUDA backend ${DECO_ENABLE_CUDA}, command-line tool ${DECO_BUILD_CLI}, tests ${DECO_BUILD_TESTS}")

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
