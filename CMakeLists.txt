cmake_minimum_required(VERSION 3.21)
project(whisper-cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add whisper.cpp submodule
add_subdirectory(src/third_party/whisper.cpp)

# Find Python and Cython
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)

# Set Python module name
set(MODULE_NAME whisper_cpp)

# Configure Cython module
cython_transpile(src/whisper_cpp/whisper.pyx LANGUAGE CXX OUTPUT_VARIABLE whisper_c)

# Create Python module
python_add_library(${MODULE_NAME} MODULE "${whisper_c}" WITH_SOABI)

# Include directories
target_include_directories(${MODULE_NAME} PRIVATE
    ${Python_INCLUDE_DIRS}
    src/third_party/whisper.cpp
)

# Link whisper.cpp library
target_link_libraries(${MODULE_NAME} PRIVATE
    whisper
    ${Python_LIBRARIES}
)

# Install the module
install(TARGETS ${MODULE_NAME} DESTINATION whisper_cpp)
