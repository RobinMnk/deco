file(GLOB ALL_CUDA_FILES CONFIGURE_DEPENDS *.cu *.cuh *.h)

add_library(DecoExpanderDecompositionCuda STATIC)
target_sources(DecoExpanderDecompositionCuda PRIVATE ${ALL_CUDA_FILES})
set_target_properties(DecoExpanderDecompositionCuda PROPERTIES CUDA_ARCHITECTURES "80;86")
target_link_libraries(DecoExpanderDecompositionCuda PUBLIC DecoLib CUDA::cudart)
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
