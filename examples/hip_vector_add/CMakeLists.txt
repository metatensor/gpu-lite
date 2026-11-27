cmake_minimum_required(VERSION 3.12)
project(HipVectorAddExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(hip_vector_add hip_vector_add.cpp)

# Include gpulite headers (assuming examples are in gpulite/examples/)
target_include_directories(hip_vector_add PRIVATE ../..)

# Link required system libraries
if(WIN32)
    # Windows doesn't need explicit dynamic loading libraries
else()
    target_link_libraries(hip_vector_add PRIVATE ${CMAKE_DL_LIBS})
endif()

# Set output directory
set_target_properties(hip_vector_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_test(NAME hip_vector_add COMMAND hip_vector_add)
# the example exits with 77 when the HIP libraries are not available
set_tests_properties(hip_vector_add PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(hip_vector_add PROPERTIES LABELS "rocm")