set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_ARCHITECTURES 52 61 75 86)

file(GLOB CUDA_SOURCES "*.cu")
file(GLOB CORE_SOURCES "../core/*.cpp")

# Force nvcc to compile all sources (including .cpp files)
set_source_files_properties(${CORE_SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(raytracer-cuda ${CUDA_SOURCES} ${CORE_SOURCES})

set_target_properties(raytracer-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(raytracer-cuda PRIVATE ../core)
target_compile_definitions(raytracer-cuda PRIVATE
    GLM_FORCE_CUDA
    GLM_FORCE_PURE
)
target_compile_options(raytracer-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(raytracer-cuda PRIVATE glm::glm)