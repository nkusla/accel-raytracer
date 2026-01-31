set(CMAKE_CUDA_STANDARD 17)

file(GLOB CUDA_SOURCES "*.cu")
add_executable(raytracer-cuda ${CUDA_SOURCES})

target_compile_definitions(raytracer-cuda PRIVATE GLM_FORCE_CUDA)
target_link_libraries(raytracer-cuda PRIVATE core)
