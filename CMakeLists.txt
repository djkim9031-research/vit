cmake_minimum_required(VERSION 3.25.0)
project(vit LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

# Find OpenMP package
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.h" "src/*.cpp")
add_executable(vit ${SOURCES})
target_link_libraries(vit PRIVATE OpenMP::OpenMP_CXX)

# Set the output directory for the main program executable
set_target_properties(vit PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Find gtest package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} src)

# Function to add a test executable
function(add_gtest_executable test_name)
    add_executable(${test_name} test/${test_name}.cpp)
    target_link_libraries(${test_name} ${GTEST_LIBRARIES} pthread)
    # Set the output directory for test executables
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add test executables
add_gtest_executable(activations)
add_gtest_executable(residual)
add_gtest_executable(conv2d)
add_gtest_executable(matmul)