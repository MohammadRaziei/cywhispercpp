cmake_minimum_required(VERSION 3.21)
project(whisper-cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

# Additional optimization flags for GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -fomit-frame-pointer -fno-stack-protector")
endif()

# Set build type to Release for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Find Python and Cython
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)

# Add whisper.cpp submodule
add_subdirectory(src/third_party/whisper.cpp)

# Set Python module name
set(MODULE_NAME whisper_cpp)

# Configure Cython module
cython_transpile(src/whisper_cpp/whisper.pyx LANGUAGE CXX OUTPUT_VARIABLE whisper_cpp_file)

# Create Python module
python_add_library(${MODULE_NAME} MODULE "${whisper_cpp_file}" WITH_SOABI)

# Include directories
target_include_directories(${MODULE_NAME} PRIVATE
    ${Python_INCLUDE_DIRS}
    src/third_party/whisper.cpp
)

# Link whisper.cpp library
target_link_libraries(${MODULE_NAME} PRIVATE whisper)

# Install the module
install(TARGETS ${MODULE_NAME} DESTINATION ${SKBUILD_PROJECT_NAME}/whisper)
install(TARGETS whisper DESTINATION ${SKBUILD_PROJECT_NAME}/whisper)
