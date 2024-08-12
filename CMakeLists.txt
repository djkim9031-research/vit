cmake_minimum_required(VERSION 3.25.0)
project(vit LANGUAGES C CXX CUDA)

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
add_gtest_executable(embeddings)
add_gtest_executable(attention)
add_gtest_executable(layernorm)
add_gtest_executable(softmax)

# CUDA project setup
find_package(MPI REQUIRED)
file(GLOB_RECURSE CUDA_SOURCES "cuda/*cuh" "cuda/*.cu")
add_executable(vit_cuda ${CUDA_SOURCES})
include_directories(/usr/include/)
set_target_properties(vit_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda LINKER_LANGUAGE CUDA)
target_include_directories(vit_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${MPI_INCLUDE_PATH})
target_link_libraries(vit_cuda nccl ${MPI_LIBRARIES})

# nccl installation
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# dpkg -i cuda-keyring_1.0-1_all.deb
# apt-get update
# apt install libnccl2-2.22.3-1_cuda12.2 libnccl-dev=2.22.3-1+cuda12.2

# openmpi installation
# apt-get install openmpi-bin libopenmpi-dev
# 
