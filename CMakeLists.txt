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


# Add custom target to install whisper.cpp separately
add_custom_target(install_whisper_cpp
    # COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/installed
    COMMAND ${CMAKE_COMMAND} -B ${CMAKE_BINARY_DIR}/whisper_cpp
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/whisper_cpp/installed 
        -DBUILD_SHARED_LIBS=ON 
        -DWHISPER_BUILD_TESTS=OFF 
        -DWHISPER_BUILD_EXAMPLES=OFF 
        -DWHISPER_BUILD_SERVER=OFF 
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/whisper_cpp -j4 --target install
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src/third_party/whisper.cpp
    COMMENT "Installing whisper.cpp to ${CMAKE_BINARY_DIR}/whisper_cpp/installed"
)

add_library(whisper_interface INTERFACE)
target_include_directories(whisper_interface INTERFACE ${CMAKE_BINARY_DIR}/whisper_cpp/installed/include)
target_link_directories(whisper_interface INTERFACE ${CMAKE_BINARY_DIR}/whisper_cpp/installed/lib)
target_link_libraries(whisper_interface INTERFACE ggml-base ggml-cpu ggml)
target_link_libraries(whisper_interface INTERFACE whisper)

# Add dependency so whisper.cpp installation happens during build
add_dependencies(${MODULE_NAME} install_whisper_cpp)

# Link whisper.cpp library
target_link_libraries(${MODULE_NAME} PRIVATE whisper_interface)



# Install the Python module
install(TARGETS ${MODULE_NAME} DESTINATION ${SKBUILD_PROJECT_NAME}/whisper)

# Copy everything from build/installed_whisper_cpp to ${SKBUILD_PROJECT_NAME}/whisper/cpp
install(DIRECTORY ${CMAKE_BINARY_DIR}/whisper_cpp/installed/
    DESTINATION ${SKBUILD_PROJECT_NAME}/whisper/cpp
)

