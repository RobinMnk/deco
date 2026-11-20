add_library(DecoLib INTERFACE)
target_compile_features(DecoLib INTERFACE cxx_std_20)   # also for programs that link DecoLib

target_include_directories(DecoLib
        INTERFACE
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)
