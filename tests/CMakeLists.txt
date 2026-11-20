include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

function(configure_test_target target_name source_file)
    add_executable(${target_name} ${source_file})

    target_link_libraries(${target_name}
            PRIVATE
            DecoLib
            GTest::gtest_main
    )

    if (MSVC)
        target_compile_definitions(${target_name} PRIVATE _CRT_DECLARE_NONSTDC_NAMES=1)
    endif()

    if (MINGW)
        target_link_options(${target_name} PRIVATE -static-libgcc -static-libstdc++)
    endif()

    include(GoogleTest)
    gtest_add_tests(TARGET ${target_name})
endfunction()


configure_test_target(graphTest graphTest.cpp)
configure_test_target(partitionTest partitionTest.cpp)
configure_test_target(algorithmsTest algorithmsTest.cpp)
configure_test_target(decoTest decoTest.cpp)

if(DECO_ENABLE_CUDA)
    configure_test_target(CudaTest cudaTest.cpp)
    target_link_libraries(CudaTest PRIVATE DecoExpanderDecompositionCuda)
    target_link_libraries(decoTest PRIVATE DecoExpanderDecompositionCuda)   # its CUDA cases run if a device is present
endif()
