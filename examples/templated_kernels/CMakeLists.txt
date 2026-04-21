cmake_minimum_required(VERSION 3.12)
project(TemplatedKernelsExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(templated_kernels templated_kernels.cpp)

# Include gpuLite headers (assuming examples are in gpuLite/examples/)
target_include_directories(templated_kernels PRIVATE ../..)

# Link required system libraries
if(WIN32)
    # Windows doesn't need explicit dynamic loading libraries
else()
    target_link_libraries(templated_kernels PRIVATE ${CMAKE_DL_LIBS})
endif()

# Set output directory
set_target_properties(templated_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_test(NAME templated_kernels COMMAND templated_kernels)
