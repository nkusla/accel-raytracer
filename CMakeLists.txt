cmake_minimum_required(VERSION 3.18)
project(accel-raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# GLM (requires: apt install libglm-dev)
find_package(glm REQUIRED)

# Core library (shared code)
add_subdirectory(src/core)

# CPU backends (sequential + OpenMP from same sources)
add_subdirectory(src/cpu)

# CUDA backend
add_subdirectory(src/cuda)
