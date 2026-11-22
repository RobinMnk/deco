file(GLOB ALL_CUDA_FILES CONFIGURE_DEPENDS *.cu *.cuh *.h)

add_library(DecoExpanderDecompositionCuda STATIC)
target_sources(DecoExpanderDecompositionCuda PRIVATE ${ALL_CUDA_FILES})
set_target_properties(DecoExpanderDecompositionCuda PROPERTIES CUDA_ARCHITECTURES "80;86")
if(DECO_BUILD_PYTHON)
    # static CUDA runtime: Python does not look for the DLLs of a module on the PATH
    set_target_properties(DecoExpanderDecompositionCuda PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    target_link_libraries(DecoExpanderDecompositionCuda PUBLIC DecoLib CUDA::cudart_static)
else()
    target_link_libraries(DecoExpanderDecompositionCuda PUBLIC DecoLib CUDA::cudart)
endif()
# programs that link the CUDA library get the CUDA backend in deco.h
target_compile_definitions(DecoExpanderDecompositionCuda PUBLIC DECO_ENABLE_CUDA)
target_include_directories(DecoExpanderDecompositionCuda
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/lib/algorithms/expander_decomposition/cuda
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(DecoExpanderDecompositionCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        --use_fast_math
        -Xptxas=-O3,-v
        -lineinfo
        >
        $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        >
)
