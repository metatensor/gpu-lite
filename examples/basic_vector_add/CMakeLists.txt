cmake_minimum_required(VERSION 3.12)
project(VectorAddExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(vector_add vector_add.cpp)

# Include gpulite headers (assuming examples are in gpulite/examples/)
target_include_directories(vector_add PRIVATE ../..)

# Link required system libraries
if(WIN32)
    # Windows doesn't need explicit dynamic loading libraries
else()
    target_link_libraries(vector_add PRIVATE ${CMAKE_DL_LIBS})
endif()

# Set output directory
set_target_properties(vector_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_test(NAME vector_add COMMAND vector_add)
