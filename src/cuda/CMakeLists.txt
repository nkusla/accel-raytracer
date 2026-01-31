set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52 61 75 86)

file(GLOB CUDA_SOURCES "*.cu")
add_executable(raytracer-cuda ${CUDA_SOURCES})

target_compile_definitions(raytracer-cuda PRIVATE 
    GLM_FORCE_CUDA
    GLM_FORCE_PURE
)
target_compile_options(raytracer-cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(raytracer-cuda PRIVATE core)
