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
option(DECO_BUILD_PYTHON "Build the Python module (pip install sets this)" OFF)
message(STATUS "Deco: CUDA backend ${DECO_ENABLE_CUDA}, command-line tool ${DECO_BUILD_CLI}, tests ${DECO_BUILD_TESTS}, Python module ${DECO_BUILD_PYTHON}")

if(DECO_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # the CUDA library is linked into the Python module, a shared library
endif()

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

# Optimization flags of the command-line tool and the Python module, which give the same results
function(deco_optimize target)
    target_compile_options(${target} PRIVATE
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
endfunction()

if(DECO_BUILD_CLI)
    add_executable(Deco cli/deco.cpp)
    deco_optimize(Deco)

    target_link_libraries(Deco PRIVATE DecoLib)

    if(DECO_ENABLE_CUDA)
        target_link_libraries(Deco PRIVATE DecoExpanderDecompositionCuda)
    endif()

    if(DECO_BUILD_PYTHON)
        install(TARGETS Deco RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR})   # pip puts it onto the PATH
    endif()
endif()

# The module deco._core of the Python package (python/), built by pip install with scikit-build-core and nanobind
if(DECO_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core NOMINSIZE python/src/module.cpp)   # NOMINSIZE: optimized for speed, not size
    deco_optimize(_core)
    target_link_libraries(_core PRIVATE DecoLib)

    if(DECO_ENABLE_CUDA)
        target_link_libraries(_core PRIVATE DecoExpanderDecompositionCuda)
    endif()

    install(TARGETS _core LIBRARY DESTINATION deco)
endif()
