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

# Smoke tests of the command-line tool on two 5-cliques joined by one edge: every command runs and finds the two cliques
if(DECO_BUILD_CLI)
    set(DATA ${CMAKE_CURRENT_SOURCE_DIR}/data)
    add_test(NAME cli.decompose COMMAND Deco decompose ${DATA}/two_cliques.mtx)
    add_test(NAME cli.decompose.json COMMAND Deco decompose ${DATA}/two_cliques_integer.mtx --json)
    add_test(NAME cli.sparsifier COMMAND Deco sparsifier ${DATA}/two_cliques_real.mtx)
    add_test(NAME cli.ncut COMMAND Deco ncut ${DATA}/two_cliques.mtx -k 2)
    set_tests_properties(cli.decompose cli.ncut PROPERTIES PASS_REGULAR_EXPRESSION "numClusters: 2")
    set_tests_properties(cli.decompose.json PROPERTIES PASS_REGULAR_EXPRESSION "\"numClusters\":2,\"cutEdges\":1,")
    set_tests_properties(cli.sparsifier PROPERTIES PASS_REGULAR_EXPRESSION "level 1: clusters 2, edges 1")

    # wrong use ends with an error
    add_test(NAME cli.ncut-without-k COMMAND Deco ncut ${DATA}/two_cliques.mtx)
    add_test(NAME cli.missing-file COMMAND Deco decompose ${DATA}/missing.mtx)
    set_tests_properties(cli.ncut-without-k cli.missing-file PROPERTIES WILL_FAIL TRUE)
endif()
