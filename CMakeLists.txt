cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME})
# Set Python module name (should match the .pyx filename)
string(REPLACE "-" "_" MODULE_NAME ${SKBUILD_PROJECT_NAME})

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




# Configure Cython module
cython_transpile(src/${MODULE_NAME}/${MODULE_NAME}.pyx LANGUAGE CXX OUTPUT_VARIABLE whisper_cpp_file)

# Create Python module
python_add_library(${MODULE_NAME} MODULE "${whisper_cpp_file}" WITH_SOABI)


include(ExternalProject)

# Set whisper.cpp install directory
set(WHISPER_CPP_INSTALL_DIR ${CMAKE_BINARY_DIR}/whisper_cpp/install)

# Define external project for whisper.cpp
ExternalProject_Add(whisper_cpp_project
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/third_party/whisper.cpp
    BINARY_DIR ${CMAKE_BINARY_DIR}/whisper_cpp/build
    INSTALL_DIR ${WHISPER_CPP_INSTALL_DIR}

    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=ON
        -DWHISPER_BUILD_TESTS=OFF
        -DWHISPER_BUILD_EXAMPLES=OFF
        -DWHISPER_BUILD_SERVER=OFF

    BUILD_ALWAYS OFF
    UPDATE_DISCONNECTED TRUE
    STEP_TARGETS build install
    LOG_CONFIGURE OFF
    LOG_BUILD OFF
    LOG_INSTALL OFF
    COMMENT "Building and installing whisper.cpp using ExternalProject_Add"
)

# Create interface library for whisper.cpp
add_library(whisper_interface INTERFACE)
add_dependencies(whisper_interface whisper_cpp_project)

target_include_directories(whisper_interface INTERFACE
    ${WHISPER_CPP_INSTALL_DIR}/include
)

target_link_directories(whisper_interface INTERFACE
    ${WHISPER_CPP_INSTALL_DIR}/lib
    ${WHISPER_CPP_INSTALL_DIR}/lib64
)

target_link_libraries(whisper_interface INTERFACE
    ggml-base ggml-cpu ggml whisper
)

# Link whisper interface to the main module
target_link_libraries(${MODULE_NAME} PRIVATE whisper_interface)

# Ensure whisper.cpp is built before the main module
add_dependencies(${MODULE_NAME} whisper_cpp_project)

# Install the Python module
install(TARGETS ${MODULE_NAME}
    DESTINATION ${SKBUILD_PROJECT_NAME}/whisper
)

# Copy whisper.cpp installation to Python module directory
install(DIRECTORY ${WHISPER_CPP_INSTALL_DIR}/
    DESTINATION ${SKBUILD_PROJECT_NAME}/whisper/cpp
)
