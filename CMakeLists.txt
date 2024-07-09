cmake_minimum_required(VERSION 3.25.0)
project(vit LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(vit ${SOURCES})
target_link_libraries(vit PRIVATE OpenMP::OpenMP_CXX)