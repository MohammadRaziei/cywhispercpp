cmake_minimum_required(VERSION 3.15)
project(whisper_cpp_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if WHISPER_CPP_INSTALL_PREFIX is provided
if(NOT DEFINED WHISPER_CPP_INSTALL_PREFIX)
    message(FATAL_ERROR "WHISPER_CPP_INSTALL_PREFIX must be defined. Usage: cmake -DWHISPER_CPP_INSTALL_PREFIX=/path/to/whisper_cpp_build")
endif()

# Verify the install prefix exists
if(NOT EXISTS ${WHISPER_CPP_INSTALL_PREFIX})
    message(FATAL_ERROR "WHISPER_CPP_INSTALL_PREFIX '${WHISPER_CPP_INSTALL_PREFIX}' does not exist")
endif()

message(STATUS "Using WHISPER_CPP_INSTALL_PREFIX: ${WHISPER_CPP_INSTALL_PREFIX}")

# Include directories
include_directories(
    ${WHISPER_CPP_INSTALL_PREFIX}/include
)

# Add executable
add_executable(test_whisper test_whisper.cpp)

# Link libraries statically
target_link_directories(test_whisper PRIVATE ${WHISPER_CPP_INSTALL_PREFIX}/lib)

# Use static linking to embed libraries in the binary
target_link_libraries(test_whisper 
    # -Wl,-Bstatic
    ggml-base
    ggml-cpu
    ggml
    whisper
    # -Wl,-Bdynamic
    pthread
    dl
)

# Set output directory
set_target_properties(test_whisper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
