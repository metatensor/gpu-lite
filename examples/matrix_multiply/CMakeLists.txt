cmake_minimum_required(VERSION 3.12)
project(MatrixMultiplyExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(matrix_multiply matrix_multiply.cpp)

# Include gpuLite headers (assuming examples are in gpuLite/examples/)
target_include_directories(matrix_multiply PRIVATE ../..)

# Link required system libraries
if(WIN32)
    # Windows doesn't need explicit dynamic loading libraries
else()
    target_link_libraries(matrix_multiply PRIVATE ${CMAKE_DL_LIBS})
endif()

# Set output directory
set_target_properties(matrix_multiply PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_test(NAME matrix_multiply COMMAND matrix_multiply)
