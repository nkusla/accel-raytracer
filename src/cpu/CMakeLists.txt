find_package(OpenMP REQUIRED)

file(GLOB CPU_SOURCES "*.cpp")

# Sequential (no OpenMP)
add_executable(raytracer-seq ${CPU_SOURCES})
target_link_libraries(raytracer-seq PRIVATE core)

# OpenMP (same sources, with pragmas)
add_executable(raytracer-omp ${CPU_SOURCES})
target_link_libraries(raytracer-omp PRIVATE core OpenMP::OpenMP_CXX)
